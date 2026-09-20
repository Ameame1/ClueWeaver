"""Interpreter agent — given a question and the segments selected by the Finder,
outputs concise rationale followed by a single answer letter.

Output schema (strict):
    <reason>concise rationale, cite paragraph numbers like [N]</reason>
    <answer>A</answer>   (or B, C, D)
"""
from __future__ import annotations
import os
import re
from collections import Counter
from dataclasses import dataclass

from src.agent.finder import FinderDecision
from src.agent.segmenter import Segment, bm25_top_paragraphs, dense_top_paragraphs
from src.llm_client import LLMClient


INTERPRETER_SYSTEM = (
    "You are the final Interpreter in a dual-agent long-narrative QA pipeline.\n"
    "You answer multiple-choice questions using ONLY the supplied evidence. Evidence paragraphs are numbered like [N].\n"
    "\n"
    "RULES:\n"
    "  - First identify the exact fact the question asks for; do not drift to a related event.\n"
    "  - Match by meaning, not wording. If evidence implies an option with equivalent wording, treat it as support.\n"
    "  - Check the question polarity. If it asks which statement is false, except, or not true, choose the contradicted or least-supported option.\n"
    "  - If it asks to deduce or infer the truth, combine clues; a coherent inferred answer can beat an isolated name/object mention.\n"
    "  - Compare every option against direct evidence. Mark unsupported options as unsupported, not contradicted.\n"
    "  - Missing evidence for an option does NOT prove it wrong. Only explicit contradiction rules it out.\n"
    "  - Prefer the option with the strongest positive support in the evidence.\n"
    "  - If evidence is incomplete, still choose exactly one option by support, contradiction, and consistency with the question polarity.\n"
    "  - Cite paragraph numbers like [478] for every decisive fact.\n"
    "  - Names may be translated (Aveline=Elena, Polo=Poirot). Match by role, not spelling.\n"
    "  - Keep <reason> under 120 words.\n"
    "\n"
    "OUTPUT — only these two XML tags, nothing else:\n"
    "<reason>concise option audit with [N] citations, under 120 words</reason>\n"
    "<answer>A</answer>"
)

INTERPRETER_USER_TEMPLATE = """Question: {question}

Options:
  (A) {opt_a}
  (B) {opt_b}
  (C) {opt_c}
  (D) {opt_d}

Evidence (kept segments, in order):
{evidence_block}

Write a compact option audit under 120 words, then output exactly one answer letter. Output only the two XML tags."""


INTERPRETER_VERIFIER_SYSTEM = (
    "You are a strict answer Verifier for a long-narrative multiple-choice QA pipeline.\n"
    "The previous Interpreter answer may be wrong. Re-evaluate the question, options, evidence, and previous answer.\n"
    "\n"
    "VERIFICATION RULES:\n"
    "  - Use ONLY the supplied evidence.\n"
    "  - If the previous reasoning says one option is directly supported but the answer tag names another option, correct the answer tag.\n"
    "  - Match the level of the question. For why/cause questions, prefer the option that explains why the asked event happened, not a generic motive or suspect.\n"
    "  - Match by meaning, not exact wording. Do not reject a semantically equivalent option just because the evidence uses different words.\n"
    "  - If evidence says an object was destroyed, hidden, wiped, made mysterious, or removed to prevent discovery, that supports options like eliminating clues/evidence.\n"
    "  - A concrete event-chain link beats a broad motive/access clue. Example: evidence that poisoning was tied to a cliff incident beats generic 'enemies' if the question asks why poisoning happened.\n"
    "  - If the question asks false, except, or not true, choose the option contradicted by the evidence or least supported.\n"
    "  - If the question asks to deduce/infer, combine clues and prefer the coherent inferred answer over isolated word overlap.\n"
    "  - Keep <reason> under 100 words and cite decisive paragraph numbers like [478].\n"
    "\n"
    "OUTPUT — only these two XML tags, nothing else:\n"
    "<reason>verified concise option audit with [N] citations</reason>\n"
    "<answer>A</answer>"
)

INTERPRETER_VERIFIER_USER_TEMPLATE = """Question: {question}

Options:
  (A) {opt_a}
  (B) {opt_b}
  (C) {opt_c}
  (D) {opt_d}

Evidence:
{evidence_block}

Previous answer:
{previous_answer}

Previous reasoning:
{previous_reason}

Verify the answer from scratch. If another option is better supported for the exact question, change it. Output only the two XML tags."""


INTERPRETER_BINARY_SYSTEM = (
    "You are the final Interpreter in a dual-agent long-narrative claim-verification pipeline.\n"
    "You will be given a claim about a book and the evidence segments selected by an earlier Finder step. "
    "The book is paragraph-numbered like [N].\n"
    "\n"
    "REQUIREMENTS:\n"
    "  - Use ONLY the evidence provided.\n"
    "  - When citing a fact, name the paragraph like [478].\n"
    "  - Check every part of the claim: people, relation, action, cause, timing, and location.\n"
    "  - Decide TRUE when the core event, participants, and relation are supported, even if the claim is paraphrased.\n"
    "  - Do not answer FALSE just because wording, sentence focus, or minor phrasing differs from the evidence.\n"
    "  - Do not add extra requirements to the claim. If the claim says an object was missing jewels, do not require proof of who stole them unless the claim says so.\n"
    "  - Decide FALSE if any essential part is contradicted, or if the supplied evidence does not support it.\n"
    "  - Related events are not enough; the core claim must match the evidence.\n"
    "  - Keep <reason> under 120 words.\n"
    "\n"
    "OUTPUT FORMAT — exactly two XML blocks and NOTHING else:\n"
    "  <reason>concise rationale with [N] citations</reason>\n"
    "  <answer>TRUE</answer>     or     <answer>FALSE</answer>"
)

INTERPRETER_BINARY_USER_TEMPLATE = """Claim to verify: {question}

Evidence (kept segments, in order):
{evidence_block}

Audit the claim against the cited evidence in under 120 words, then output TRUE or FALSE. Output only the two XML tags."""


INTERPRETER_BINARY_VERIFIER_SYSTEM = (
    "You are a strict Verifier for binary long-narrative claim verification.\n"
    "The previous answer may be wrong. Re-check only what the claim actually says using the supplied evidence.\n"
    "\n"
    "RULES:\n"
    "  - Use ONLY the supplied evidence.\n"
    "  - TRUE if the core event, participants, and object/state match, even if the wording is paraphrased.\n"
    "  - Do not add extra requirements. If the claim says jewels were missing, do not require proof of who stole them or why.\n"
    "  - Later explanations of cause do not make an earlier factual state FALSE unless they contradict the stated event.\n"
    "  - FALSE only when an essential claim element is contradicted or unsupported.\n"
    "  - Keep <reason> under 100 words and cite paragraph numbers.\n"
    "\n"
    "OUTPUT FORMAT — exactly two XML blocks and NOTHING else:\n"
    "  <reason>verified audit with [N] citations</reason>\n"
    "  <answer>TRUE</answer>     or     <answer>FALSE</answer>"
)

INTERPRETER_BINARY_VERIFIER_USER_TEMPLATE = """Claim: {question}

Evidence:
{evidence_block}

Previous answer:
{previous_answer}

Previous reasoning:
{previous_reason}

Verify from scratch. Output only the two XML tags."""


RE_INTERPRETER_PARSE = re.compile(
    r"<reason>(.+?)</reason>\s*<answer>\s*([ABCD])\s*</answer>",
    re.S | re.I,
)
RE_INTERPRETER_BINARY_PARSE = re.compile(
    r"<reason>(.+?)</reason>\s*<answer>\s*(TRUE|FALSE)\s*</answer>",
    re.S | re.I,
)


@dataclass
class InterpreterOutput:
    answer: str  # "A"/"B"/"C"/"D" or "" if parse failed
    reason: str
    raw: str
    n_evidence_segments: int = 0
    evidence_chars: int = 0
    evidence: str = ""


STOP_WORDS = set(
    "a an the of in on at by to from for with as is are was were be been being have has had "
    "do does did will would shall should may might can could and or but not no nor so than "
    "that this these those i you he she it we they him her us them my your his its our their "
    "what which who whom whose why when where how does did was were about into onto over under "
    "after before during between among one two three four following option options true false claim "
    "question answer"
    .split()
)
TOKEN_NORMALIZATION = {
    "deliberate": "deliberate",
    "deliberately": "deliberate",
    "intentional": "deliberate",
    "intentionally": "deliberate",
    "purpose": "deliberate",
    "purposely": "deliberate",
    "mistake": "accidental",
    "mistaken": "accidental",
    "negligence": "accidental",
    "negligent": "accidental",
    "accident": "accidental",
    "accidental": "accidental",
    "accidentally": "accidental",
    "forgot": "accidental",
    "forget": "accidental",
    "poison": "poison",
    "poisoned": "poison",
    "poisoning": "poison",
    "fingerprint": "fingerprint",
    "fingerprints": "fingerprint",
    "murder": "kill",
    "murdered": "kill",
    "murderer": "kill",
    "killer": "kill",
    "killed": "kill",
    "kill": "kill",
    "pushed": "push",
    "push": "push",
    "fall": "fall",
    "fell": "fall",
    "fallen": "fall",
}


RE_WORD = re.compile(r"[A-Za-z][A-Za-z'-]{2,}")
RE_PARA = re.compile(r"\[(\d+)\]")
RE_PARA_LINE = re.compile(r"^\[(\d+)\]\s*(.*)")
RE_BRIDGE_TEXT = re.compile(
    r"\b(because|therefore|therefore|so|means|must|clue|purpose|intentional|intentionally|fingerprint|"
    r"exactly|of course not|that's what|this means|which means)\b",
    re.I,
)
RE_VERIFY_TRIGGER = re.compile(
    r"\b(why|reason|because|false|not true|except|most likely|deduce|deduction|infer|true story|real cause)\b"
    r"|object.*switch|switch.*object|\bwho pushed\b",
    re.I,
)
RE_AUTO_RETRIEVAL_ORDERED = re.compile(
    r"\breason\b|\bpurpose\b|\btechnique\b|\bmethod\b|\bhow did\b|\bwho (killed|pushed|wrote)\b",
    re.I,
)
RE_AUTO_RETRIEVAL_ORDERED_AVOID = re.compile(
    r"\b(false|not true|except|deduce|deduction|true story|real cause|real name|identity|profession|perpetrator)\b",
    re.I,
)


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _tokenize(text: str) -> Counter:
    tokens = []
    for m in RE_WORD.finditer(text):
        word = m.group(0).lower()
        if word in STOP_WORDS:
            continue
        tokens.append(TOKEN_NORMALIZATION.get(word, word))
    return Counter(tokens)


def _query_counter(question: str, options: dict[str, str]) -> Counter:
    text = question
    if options:
        text += " " + " ".join(options.get(k, "") for k in "ABCD")
    return _tokenize(text)


def _dice_score(query: Counter, text: str) -> float:
    if not query:
        return 0.0
    doc = _tokenize(text)
    if not doc:
        return 0.0
    common = sum((query & doc).values())
    return 2.0 * common / (sum(query.values()) + sum(doc.values()) + 1.0)


def _parse_numbered_paragraphs(text: str) -> list[tuple[int, str]]:
    out: list[tuple[int, str]] = []
    current_id = None
    current_lines: list[str] = []
    for line in text.splitlines():
        m = RE_PARA_LINE.match(line)
        if m:
            if current_id is not None:
                out.append((current_id, "\n".join(current_lines).strip()))
            current_id = int(m.group(1))
            current_lines = [m.group(2)]
        elif current_id is not None:
            current_lines.append(line)
    if current_id is not None:
        out.append((current_id, "\n".join(current_lines).strip()))
    return out


def _extract_cited_para_ids(*texts: str | None) -> set[int]:
    ids: set[int] = set()
    for text in texts:
        if not text:
            continue
        for m in RE_PARA.findall(text):
            try:
                ids.add(int(m))
            except ValueError:
                continue
    return ids


def _compact_segment_text(
    text: str,
    query: Counter,
    max_chars: int = 3500,
    max_paragraphs: int = 6,
    expand_neighbors: int = 0,
    priority_para_ids: set[int] | None = None,
) -> str:
    """Keep short focused segments; compress long/noisy ones to query-relevant paragraphs."""
    paras = _parse_numbered_paragraphs(text)
    if len(text) <= max_chars and len(paras) <= max_paragraphs:
        return text
    if not paras:
        return text[:max_chars] + "\n... [segment truncated]"

    scored = [(_dice_score(query, body), pid, body) for pid, body in paras]
    scored.sort(key=lambda x: (x[0], len(x[2])), reverse=True)
    para_ids = {pid for pid, _ in paras}
    priority_ids = set(priority_para_ids or set()) & para_ids
    ranked_keep = []
    for pid in sorted(priority_ids):
        ranked_keep.append(pid)
        if len(ranked_keep) >= max_paragraphs:
            break
    for score, pid, _ in scored:
        if len(ranked_keep) >= max_paragraphs:
            break
        if score > 0 and pid not in ranked_keep:
            ranked_keep.append(pid)
    keep_ids = set(ranked_keep)
    if not keep_ids:
        keep_ids = {pid for _, pid, _ in scored[:max(3, max_paragraphs // 3)]}
    if expand_neighbors > 0:
        expanded_ids = set(keep_ids)
        for pid in keep_ids:
            for delta in range(1, expand_neighbors + 1):
                if pid - delta in para_ids:
                    expanded_ids.add(pid - delta)
                if pid + delta in para_ids:
                    expanded_ids.add(pid + delta)
        keep_ids = expanded_ids

    lines = []
    last_pid = None
    used = 0
    for pid, body in paras:
        if pid not in keep_ids:
            continue
        para = f"[{pid}] {body}"
        if last_pid is not None and pid != last_pid + 1:
            lines.append("...")
        if used + len(para) > max_chars:
            lines.append("... [segment compacted for length]")
            break
        lines.append(para)
        used += len(para)
        last_pid = pid
    return "\n".join(lines)


def _content_token_count(text: str) -> int:
    return sum(_tokenize(text).values())


def _is_micro_retrieval(seg: Segment) -> bool:
    if not seg.kind.startswith("retrieval"):
        return False
    min_tokens = _env_int("FILM_CS_RETRIEVAL_MIN_CONTENT_TOKENS", 6)
    min_chars = _env_int("FILM_CS_RETRIEVAL_MIN_CHARS", 48)
    return _content_token_count(seg.text) < min_tokens and len(seg.text) < min_chars


def _adjusted_segment_score(seg: Segment, base_score: float) -> float:
    if _is_micro_retrieval(seg):
        penalty = os.environ.get("FILM_CS_MICRO_RETRIEVAL_PENALTY")
        try:
            scale = float(penalty) if penalty is not None else 0.45
        except ValueError:
            scale = 0.45
        return base_score * max(0.0, min(scale, 1.0))
    return base_score


def _contextual_rank_scores(
    segments: list[Segment],
    yes_ids: set[int],
    adjusted_scores: dict[int, float],
    bridge_enabled: bool,
) -> dict[int, float]:
    by_seg = {s.seg_id: s for s in segments}
    informative_yes_ids = [sid for sid in yes_ids if sid in by_seg and not _is_micro_retrieval(by_seg[sid])]
    lookback_gap = _env_int("FILM_CS_EVIDENCE_LOOKBACK_GAP", 80)
    anchor_neighbor_gap = _env_int("FILM_CS_EVIDENCE_ANCHOR_NEIGHBOR_GAP", 24)
    out = dict(adjusted_scores)

    for sid in yes_ids:
        seg = by_seg.get(sid)
        if not seg:
            continue
        bonus = 0.0
        if seg.kind.startswith("retrieval"):
            prev_windows = sorted(
                (
                    wid for wid in informative_yes_ids
                    if wid != sid
                    and by_seg[wid].kind == "window"
                    and 0 <= seg.start_para - by_seg[wid].end_para <= lookback_gap
                ),
                key=lambda wid: (by_seg[wid].end_para, adjusted_scores.get(wid, 0.0)),
            )
            bonus += 0.35 * sum(adjusted_scores.get(wid, 0.0) for wid in prev_windows[-2:])
            if bridge_enabled:
                next_windows = sorted(
                    (
                        wid for wid in informative_yes_ids
                        if wid != sid
                        and by_seg[wid].kind == "window"
                        and 0 <= by_seg[wid].start_para - seg.end_para <= anchor_neighbor_gap
                    ),
                    key=lambda wid: (by_seg[wid].start_para, -adjusted_scores.get(wid, 0.0)),
                )
                if next_windows:
                    bonus += 0.15 * adjusted_scores.get(next_windows[0], 0.0)
        else:
            nearby_retrieval = max(
                (
                    adjusted_scores.get(rid, 0.0)
                    for rid in informative_yes_ids
                    if rid != sid
                    and by_seg[rid].kind.startswith("retrieval")
                    and (
                        by_seg[rid].start_para <= seg.end_para + anchor_neighbor_gap
                        and by_seg[rid].end_para >= seg.start_para - anchor_neighbor_gap
                    )
                ),
                default=0.0,
            )
            bonus += 0.20 * nearby_retrieval
        out[sid] = adjusted_scores.get(sid, 0.0) + bonus
    return out


def _seed_chronological_selection(
    segments: list[Segment],
    yes_ids: set[int],
    adjusted_scores: dict[int, float],
    max_segments: int,
    bridge_enabled: bool,
    prefer_late: bool = False,
) -> list[int]:
    by_seg = {s.seg_id: s for s in segments}
    lookback_gap = _env_int("FILM_CS_EVIDENCE_LOOKBACK_GAP", 80)
    lookback_windows = _env_int("FILM_CS_EVIDENCE_LOOKBACK_WINDOWS", 2)
    anchor_neighbor_gap = _env_int("FILM_CS_EVIDENCE_ANCHOR_NEIGHBOR_GAP", 24)
    max_para = max((s.end_para for s in segments), default=1)

    selected_ids: list[int] = []
    selected_set: set[int] = set()
    suppress_micro_retrieval = True

    def add_selected(sid: int) -> None:
        if sid in by_seg and sid not in selected_set and len(selected_ids) < max_segments:
            if suppress_micro_retrieval and _is_micro_retrieval(by_seg[sid]):
                return
            selected_ids.append(sid)
            selected_set.add(sid)

    informative_yes_ids = [sid for sid in yes_ids if sid in by_seg and not _is_micro_retrieval(by_seg[sid])]

    def previous_windows(anchor: Segment, exclude: set[int] | None = None) -> list[int]:
        exclude = exclude or set()
        candidates = [
            sid for sid in informative_yes_ids
            if sid not in exclude
            and sid != anchor.seg_id
            and by_seg[sid].kind == "window"
            and 0 <= anchor.start_para - by_seg[sid].end_para <= lookback_gap
        ]
        candidates.sort(key=lambda sid: (by_seg[sid].end_para, adjusted_scores.get(sid, 0.0)))
        kept = candidates[-lookback_windows:] if lookback_windows > 0 else []
        return sorted(kept, key=lambda sid: by_seg[sid].start_para)

    def next_windows(anchor: Segment, exclude: set[int] | None = None) -> list[int]:
        exclude = exclude or set()
        candidates = [
            sid for sid in informative_yes_ids
            if sid not in exclude
            and sid != anchor.seg_id
            and by_seg[sid].kind == "window"
            and 0 <= by_seg[sid].start_para - anchor.end_para <= anchor_neighbor_gap
        ]
        if not candidates:
            return []
        candidates.sort(key=lambda sid: (by_seg[sid].start_para, -adjusted_scores.get(sid, 0.0)))
        return [candidates[0]]

    def nearby_retrievals(anchor: Segment, exclude: set[int] | None = None) -> list[int]:
        exclude = exclude or set()
        candidates = [
            sid for sid in informative_yes_ids
            if sid not in exclude
            and sid != anchor.seg_id
            and by_seg[sid].kind.startswith("retrieval")
            and (
                by_seg[sid].start_para <= anchor.end_para + anchor_neighbor_gap
                and by_seg[sid].end_para >= anchor.start_para - anchor_neighbor_gap
            )
        ]
        candidates.sort(
            key=lambda sid: (
                1 if anchor.start_para <= by_seg[sid].start_para <= anchor.end_para else 0,
                adjusted_scores.get(sid, 0.0),
            ),
            reverse=True,
        )
        return candidates[:1]

    retrieval_anchor_ids = [sid for sid in informative_yes_ids if by_seg[sid].kind.startswith("retrieval")]
    if retrieval_anchor_ids:
        anchor_ids = retrieval_anchor_ids
    else:
        anchor_ids = informative_yes_ids or [sid for sid in yes_ids if sid in by_seg]

    def anchor_support_score(sid: int) -> float:
        anchor = by_seg[sid]
        score = adjusted_scores.get(sid, 0.0)
        if anchor.kind.startswith("retrieval"):
            score += 0.35 * sum(adjusted_scores.get(wid, 0.0) for wid in previous_windows(anchor))
            if bridge_enabled:
                score += 0.15 * sum(adjusted_scores.get(wid, 0.0) for wid in next_windows(anchor))
        else:
            score += 0.35 * sum(adjusted_scores.get(rid, 0.0) for rid in nearby_retrievals(anchor))
        if prefer_late:
            score += 0.05 * (anchor.start_para / max(1, max_para))
        return score

    anchor_ids = sorted(
        anchor_ids,
        key=lambda sid: (anchor_support_score(sid), adjusted_scores.get(sid, 0.0)),
        reverse=True,
    )

    anchor_budget = min(
        len(anchor_ids),
        max(1, _env_int("FILM_CS_EVIDENCE_MAX_ANCHORS", max(1, max_segments // 2))),
    )
    diversify_gap = _env_int("FILM_CS_EVIDENCE_ANCHOR_DIVERSIFY_GAP", 18)
    chosen_anchor_ids: list[int] = []
    for sid in anchor_ids:
        seg = by_seg[sid]
        if any(
            abs(seg.start_para - by_seg[aid].start_para) <= diversify_gap
            or abs(seg.end_para - by_seg[aid].end_para) <= diversify_gap
            for aid in chosen_anchor_ids
        ):
            continue
        chosen_anchor_ids.append(sid)
        if len(chosen_anchor_ids) >= anchor_budget:
            break
    if not chosen_anchor_ids and anchor_ids:
        chosen_anchor_ids = [anchor_ids[0]]

    extra_candidates: list[tuple[float, int]] = []
    for sid in sorted(chosen_anchor_ids, key=lambda item: by_seg[item].start_para):
        if len(selected_ids) >= max_segments:
            break
        anchor = by_seg[sid]
        local_ids: list[int] = []
        if anchor.kind.startswith("retrieval"):
            prevs = previous_windows(anchor, selected_set)
            nexts = next_windows(anchor, selected_set)
            if prevs:
                local_ids.append(prevs[-1])
                for ctx_sid in reversed(prevs[:-1]):
                    extra_candidates.append((adjusted_scores.get(ctx_sid, 0.0), ctx_sid))
            local_ids.append(sid)
            if bridge_enabled and nexts and not prevs:
                local_ids.append(nexts[0])
            elif bridge_enabled:
                for ctx_sid in nexts:
                    extra_candidates.append((adjusted_scores.get(ctx_sid, 0.0), ctx_sid))
        else:
            local_ids.append(sid)
            near_retrievals = nearby_retrievals(anchor, selected_set)
            if near_retrievals:
                local_ids.append(near_retrievals[0])
        local_ids = sorted(dict.fromkeys(local_ids), key=lambda item: by_seg[item].start_para)
        for local_sid in local_ids:
            add_selected(local_sid)
            if len(selected_ids) >= max_segments:
                break

    if len(selected_ids) < max_segments and extra_candidates:
        for _, sid in sorted(extra_candidates, key=lambda item: item[0], reverse=True):
            add_selected(sid)
            if len(selected_ids) >= max_segments:
                break

    if len(selected_ids) < max_segments:
        for sid in anchor_ids:
            add_selected(sid)
            if len(selected_ids) >= max_segments:
                break

    return selected_ids


def _paragraph_map_from_segments(segments: list[Segment]) -> dict[int, str]:
    para_by_id: dict[int, str] = {}
    for seg in segments:
        for pid, body in _parse_numbered_paragraphs(seg.text):
            body = body.strip()
            if not body:
                continue
            old = para_by_id.get(pid)
            if old is None or len(body) > len(old):
                para_by_id[pid] = body
    return para_by_id


def _merge_para_ranges(ranges: list[tuple[int, int]], max_gap: int) -> list[tuple[int, int]]:
    if not ranges:
        return []
    ranges = sorted(ranges)
    merged = [ranges[0]]
    for start, end in ranges[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end + max_gap + 1:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def _local_paragraph_window_text(
    para_by_id: dict[int, str],
    center_pid: int,
    radius: int,
) -> str:
    if not para_by_id or center_pid not in para_by_id:
        return ""
    para_ids = sorted(para_by_id)
    start = max(para_ids[0], center_pid - radius)
    end = min(para_ids[-1], center_pid + radius)
    return "\n".join(
        f"[{pid}] {para_by_id[pid]}"
        for pid in range(start, end + 1)
        if pid in para_by_id
    )


def _query_text(question: str, options: dict[str, str]) -> str:
    if not options:
        return question
    return question + "\n" + "\n".join(f"{k}. {options.get(k, '')}" for k in "ABCD")


def _should_use_retrieval_ordered(question: str, options: dict[str, str]) -> bool:
    if not options:
        return False
    option_words = sum(len((options.get(k) or "").split()) for k in "ABCD")
    max_option_words = _env_int("FILM_CS_RETRIEVAL_ORDERED_MAX_OPTION_WORDS", 40)
    if option_words > max_option_words:
        return False
    return bool(RE_AUTO_RETRIEVAL_ORDERED.search(question)) and not bool(
        RE_AUTO_RETRIEVAL_ORDERED_AVOID.search(question)
    )


def _ordered_windows_from_anchor_pids(
    para_by_id: dict[int, str],
    anchor_pids: list[int],
    radius: int,
    max_total_chars: int,
    merge_gap: int = 1,
    source: str = "ordered-window",
) -> tuple[str, int]:
    if not para_by_id or not anchor_pids:
        return "", 0
    para_ids = sorted(para_by_id)
    min_pid, max_pid = para_ids[0], para_ids[-1]
    ranges = [
        (max(min_pid, pid - radius), min(max_pid, pid + radius))
        for pid in anchor_pids
        if pid in para_by_id
    ]
    ranges = _merge_para_ranges(ranges, merge_gap)
    blocks: list[str] = []
    used = 0
    for start, end in ranges:
        lines = [f"[{pid}] {para_by_id[pid]}" for pid in range(start, end + 1) if pid in para_by_id]
        if not lines:
            continue
        hdr = f"--- Segment (paragraphs {start}-{end}; {source}) ---"
        block = f"{hdr}\n" + "\n".join(lines)
        if used + len(block) > max_total_chars:
            if blocks:
                blocks.append(f"--- (... {len(ranges) - len(blocks)} more ordered windows truncated for length ...)")
                break
            block = block[:max_total_chars].rstrip() + "\n... [ordered window truncated]"
        blocks.append(block)
        used += len(block)
    return "\n\n".join(blocks), len(blocks)


def _top_paragraph_anchors(
    seg: Segment,
    query: Counter,
    cited_para_ids: set[int],
    limit: int,
) -> list[tuple[float, int]]:
    paras = _parse_numbered_paragraphs(seg.text)
    if not paras:
        return []
    para_ids = {pid for pid, _ in paras}
    body_by_pid = {pid: body for pid, body in paras}
    cited = sorted(pid for pid in cited_para_ids if pid in para_ids)
    if cited:
        return [
            (_dice_score(query, body_by_pid.get(pid, "")) + 0.20, pid)
            for pid in cited[:limit]
        ]

    scored = []
    for pid, body in paras:
        score = _dice_score(query, body)
        if RE_BRIDGE_TEXT.search(body):
            score += 0.03
        scored.append((score, pid))
    scored.sort(key=lambda item: (item[0], -abs(item[1] - seg.start_para)), reverse=True)
    positive = [item for item in scored if item[0] > 0]
    if positive:
        return positive[:limit]
    mid = (seg.start_para + seg.end_para) // 2
    return [(0.0, min(para_ids, key=lambda pid: abs(pid - mid)))]


def _build_ordered_window_evidence_block(
    segments: list[Segment],
    decisions: list[FinderDecision],
    question: str,
    options: dict[str, str],
    yes_ids: set[int],
    scores: dict[int, float],
    cited_para_by_seg: dict[int, set[int]],
    max_total_chars: int,
    max_segments: int,
    min_segments: int,
) -> tuple[str, int]:
    """Convert high-recall YES segments into ordered local narrative windows."""
    by_seg = {s.seg_id: s for s in segments}
    query = _query_counter(question, options)
    para_by_id = _paragraph_map_from_segments(segments)
    if not para_by_id:
        return "", 0

    radius = _env_int("FILM_CS_ORDERED_WINDOW_RADIUS", 2)
    anchor_limit = _env_int("FILM_CS_ORDERED_WINDOW_MAX_ANCHORS", max_segments)
    anchor_limit = max(1, anchor_limit)
    anchors_per_window = _env_int("FILM_CS_ORDERED_WINDOW_ANCHORS_PER_SEGMENT", 1)
    merge_gap = _env_int("FILM_CS_ORDERED_WINDOW_MERGE_GAP", 1)
    diversify_gap = _env_int("FILM_CS_ORDERED_WINDOW_DIVERSIFY_GAP", max(2, radius * 2))

    scored_anchors: list[tuple[float, int]] = []
    for sid in yes_ids:
        seg = by_seg.get(sid)
        if not seg or _is_micro_retrieval(seg):
            continue
        if seg.kind.startswith("retrieval"):
            pid = seg.start_para
            score = scores.get(sid, 0.0) + 0.60
            if cited_para_by_seg.get(sid):
                score += 0.15
            scored_anchors.append((score, pid))
            continue
        for local_score, pid in _top_paragraph_anchors(
            seg,
            query,
            cited_para_by_seg.get(sid, set()),
            max(1, anchors_per_window),
        ):
            scored_anchors.append((scores.get(sid, 0.0) + local_score, pid))

    if len(scored_anchors) < min_segments:
        for seg in sorted(segments, key=lambda s: scores.get(s.seg_id, 0.0), reverse=True):
            if seg.seg_id in yes_ids or _is_micro_retrieval(seg):
                continue
            anchors = _top_paragraph_anchors(seg, query, set(), 1)
            if not anchors:
                continue
            local_score, pid = anchors[0]
            if local_score <= 0 and scored_anchors:
                continue
            scored_anchors.append((scores.get(seg.seg_id, 0.0) + local_score, pid))
            if len(scored_anchors) >= min_segments:
                break

    scored_anchors.sort(key=lambda item: item[0], reverse=True)
    chosen_pids: list[int] = []
    for _, pid in scored_anchors:
        if pid not in para_by_id:
            continue
        if any(abs(pid - chosen) <= diversify_gap for chosen in chosen_pids):
            continue
        chosen_pids.append(pid)
        if len(chosen_pids) >= anchor_limit:
            break
    if not chosen_pids and scored_anchors:
        chosen_pids = [pid for _, pid in scored_anchors[:1] if pid in para_by_id]
    if not chosen_pids:
        return "", 0

    para_ids = sorted(para_by_id)
    min_pid, max_pid = para_ids[0], para_ids[-1]
    ranges = [
        (max(min_pid, pid - radius), min(max_pid, pid + radius))
        for pid in chosen_pids
    ]
    ranges = _merge_para_ranges(ranges, merge_gap)

    blocks: list[str] = []
    used = 0
    for start, end in ranges:
        lines = [f"[{pid}] {para_by_id[pid]}" for pid in range(start, end + 1) if pid in para_by_id]
        if not lines:
            continue
        hdr = f"--- Segment (paragraphs {start}-{end}; finder=YES ordered-window) ---"
        block = f"{hdr}\n" + "\n".join(lines)
        if used + len(block) > max_total_chars:
            if blocks:
                blocks.append(f"--- (... {len(ranges) - len(blocks)} more ordered windows truncated for length ...)")
                break
            block = block[:max_total_chars].rstrip() + "\n... [ordered window truncated]"
        blocks.append(block)
        used += len(block)
    return "\n\n".join(blocks), len(blocks)


def _build_retrieval_ordered_evidence_block(
    segments: list[Segment],
    question: str,
    options: dict[str, str],
    yes_ids: set[int],
    max_total_chars: int,
    max_segments: int,
) -> tuple[str, int]:
    """Use retrieval-ranked paragraph anchors, with Finder YES as a soft gate."""
    para_by_id = _paragraph_map_from_segments(segments)
    if not para_by_id:
        return "", 0

    para_ids = sorted(para_by_id)
    paragraphs = [para_by_id[pid] for pid in para_ids]
    top_k = max(max_segments * 4, _env_int("FILM_CS_RETRIEVAL_ORDERED_POOL", 24))
    bm25_local_ids = bm25_top_paragraphs(paragraphs, _query_text(question, options), top_k=top_k)
    ranked_pids = [para_ids[i] for i in bm25_local_ids if 0 <= i < len(para_ids)]
    if not ranked_pids:
        return "", 0

    by_seg = {s.seg_id: s for s in segments}
    yes_para_ids: set[int] = set()
    for sid in yes_ids:
        seg = by_seg.get(sid)
        if not seg:
            continue
        yes_para_ids.update(range(seg.start_para, seg.end_para + 1))

    anchor_limit = _env_int("FILM_CS_RETRIEVAL_ORDERED_MAX_ANCHORS", max_segments)
    anchor_limit = max(1, anchor_limit)
    diversify_gap = _env_int("FILM_CS_RETRIEVAL_ORDERED_DIVERSIFY_GAP", 4)
    chosen: list[int] = []

    def add_pid(pid: int) -> None:
        if pid not in para_by_id:
            return
        if any(abs(pid - old) <= diversify_gap for old in chosen):
            return
        chosen.append(pid)

    for pid in ranked_pids:
        if pid in yes_para_ids:
            add_pid(pid)
            if len(chosen) >= anchor_limit:
                break
    if len(chosen) < anchor_limit:
        for pid in ranked_pids:
            add_pid(pid)
            if len(chosen) >= anchor_limit:
                break

    radius = _env_int("FILM_CS_RETRIEVAL_ORDERED_RADIUS", 2)
    merge_gap = _env_int("FILM_CS_RETRIEVAL_ORDERED_MERGE_GAP", 1)
    return _ordered_windows_from_anchor_pids(
        para_by_id,
        chosen,
        radius=radius,
        max_total_chars=max_total_chars,
        merge_gap=merge_gap,
        source="retrieval-ranked ordered-window",
    )


def _build_hybrid_yes_ordered_evidence_block(
    segments: list[Segment],
    question: str,
    options: dict[str, str],
    yes_ids: set[int],
    max_total_chars: int,
    max_segments: int,
) -> tuple[str, int]:
    """Hybrid-rank paragraph anchors only inside Finder-YES evidence."""
    para_by_id = _paragraph_map_from_segments(segments)
    if not para_by_id or not yes_ids:
        return "", 0

    by_seg = {s.seg_id: s for s in segments}
    yes_para_ids: set[int] = set()
    for sid in yes_ids:
        seg = by_seg.get(sid)
        if not seg:
            continue
        parsed = _parse_numbered_paragraphs(seg.text)
        if parsed:
            yes_para_ids.update(pid for pid, _ in parsed)
        else:
            yes_para_ids.update(range(seg.start_para, seg.end_para + 1))

    candidate_pids = sorted(pid for pid in yes_para_ids if pid in para_by_id)
    if not candidate_pids:
        return "", 0
    candidate_texts = [para_by_id[pid] for pid in candidate_pids]
    query_text = _query_text(question, options)

    bm25_ids = bm25_top_paragraphs(candidate_texts, query_text, top_k=len(candidate_texts))
    dense_ids = dense_top_paragraphs(
        candidate_texts,
        query_text,
        candidate_ids=list(range(len(candidate_texts))),
        top_k=len(candidate_texts),
        batch_size=_env_int("FILM_CS_HYBRID_YES_DENSE_BATCH_SIZE", 16),
    )

    rrf_k = _env_int("FILM_CS_HYBRID_YES_RRF_K", 60)
    bm25_weight = float(os.environ.get("FILM_CS_HYBRID_YES_BM25_WEIGHT", "1.0") or 1.0)
    dense_weight = float(os.environ.get("FILM_CS_HYBRID_YES_DENSE_WEIGHT", "1.0") or 1.0)
    scores: dict[int, float] = {}
    for rank, idx in enumerate(bm25_ids):
        scores[idx] = scores.get(idx, 0.0) + bm25_weight / (rrf_k + rank + 1)
    for rank, idx in enumerate(dense_ids):
        scores[idx] = scores.get(idx, 0.0) + dense_weight / (rrf_k + rank + 1)
    if not scores:
        return "", 0

    ranked = sorted(scores, key=lambda idx: scores[idx], reverse=True)
    anchor_limit = max(1, _env_int("FILM_CS_HYBRID_YES_MAX_ANCHORS", max_segments))
    diversify_gap = _env_int("FILM_CS_HYBRID_YES_DIVERSIFY_GAP", 4)
    chosen: list[int] = []
    for idx in ranked:
        pid = candidate_pids[idx]
        if any(abs(pid - old) <= diversify_gap for old in chosen):
            continue
        chosen.append(pid)
        if len(chosen) >= anchor_limit:
            break

    radius = _env_int("FILM_CS_HYBRID_YES_RADIUS", 2)
    merge_gap = _env_int("FILM_CS_HYBRID_YES_MERGE_GAP", 1)
    return _ordered_windows_from_anchor_pids(
        para_by_id,
        chosen,
        radius=radius,
        max_total_chars=max_total_chars,
        merge_gap=merge_gap,
        source="hybrid-YES ordered-window",
    )


def _build_evidence_block(
    segments: list[Segment],
    decisions: list[FinderDecision],
    question: str,
    options: dict[str, str],
    max_total_chars: int = 13000,
    max_segments: int = 5,
    min_segments: int = 3,
) -> str:
    """Select and compact evidence for the small final interpreter.

    Finder selection is high-recall but noisy. This deterministic stage keeps the
    best YES segments, rescues lexical top hits when the Finder is too sparse,
    and compacts long segments to the most query-relevant paragraphs.
    """
    query = _query_counter(question, options)
    neighbor_expand = 1 if re.search(r"\b(why|reason|because)\b", question, re.I) else 0
    bridge_enabled = bool(re.search(r"\b(why|reason)\b", question, re.I)) or (
        bool(re.search(r"\bhow\b", question, re.I))
        and not re.search(r"\b(deduce|deduction|true story|real cause)\b", question, re.I)
    )
    tail_bias = bool(options and re.search(
        r"\b(false|not true|except)\b|most likely .*responsible|object.*switch|switch.*object",
        question,
        re.I,
    ))
    tail_focus = tail_bias or (
        bool(options and re.search(r"\bwho (killed|pushed)\b", question, re.I))
        and not re.search(r"\b(perspective|suspect|suspected|police|inspector)\b", question, re.I)
    )
    prefer_late = tail_focus or bool(re.search(r"\b(why|reason|cause|purpose)\b", question, re.I))
    option_bias = tail_bias or os.environ.get("FILM_CS_OPTION_COVERAGE", "0").lower() in {
        "1", "true", "yes"
    }
    by_seg = {s.seg_id: s for s in segments}
    decision_by_seg = {d.seg_id: d for d in decisions}
    selection_mode = os.environ.get("FILM_CS_EVIDENCE_SELECTOR_MODE", "ranked").strip().lower()
    citation_priority_enabled = (
        selection_mode == "ordered_windows"
        or os.environ.get("FILM_CS_USE_FILTER_CITATIONS", "0").lower() not in {
        "0", "false", "no"
        }
    )
    cited_para_by_seg: dict[int, set[int]] = {}
    if citation_priority_enabled:
        for d in decisions:
            seg = by_seg.get(d.seg_id)
            if not seg or d.verdict != "YES":
                continue
            cited = _extract_cited_para_ids(d.reason, d.raw)
            cited_para_by_seg[d.seg_id] = {
                pid for pid in cited if seg.start_para <= pid <= seg.end_para
            }
    scores = {
        s.seg_id: (
            _dice_score(query, s.text)
            + (0.05 if s.kind.startswith("retrieval") else 0.0)
            + (0.04 if "tail" in s.kind else 0.0)
            + (
                min(0.12, 0.06 + 0.02 * len(cited_para_by_seg.get(s.seg_id, set())))
                if cited_para_by_seg.get(s.seg_id)
                else 0.0
            )
        )
        for s in segments
    }
    yes_ids = {d.seg_id for d in decisions if d.verdict == "YES" and d.seg_id in by_seg}
    adjusted_scores = {s.seg_id: _adjusted_segment_score(s, scores[s.seg_id]) for s in segments}
    rank_scores = (
        _contextual_rank_scores(segments, yes_ids, adjusted_scores, bridge_enabled)
        if selection_mode == "support_ranked"
        else adjusted_scores
    )

    if selection_mode == "ordered_windows":
        evidence, n_blocks = _build_ordered_window_evidence_block(
            segments,
            decisions,
            question,
            options,
            yes_ids,
            adjusted_scores,
            cited_para_by_seg,
            max_total_chars,
            max_segments,
            min_segments,
        )
        if evidence:
            return evidence, n_blocks
    use_retrieval_ordered = selection_mode == "retrieval_ordered_windows" or (
        selection_mode == "auto_retrieval_ordered"
        and _should_use_retrieval_ordered(question, options)
    )
    if use_retrieval_ordered:
        evidence, n_blocks = _build_retrieval_ordered_evidence_block(
            segments,
            question,
            options,
            yes_ids,
            max_total_chars,
            max_segments,
        )
        if evidence:
            return evidence, n_blocks
    if selection_mode == "hybrid_yes_ordered_windows":
        evidence, n_blocks = _build_hybrid_yes_ordered_evidence_block(
            segments,
            question,
            options,
            yes_ids,
            max_total_chars,
            max_segments,
        )
        if evidence:
            return evidence, n_blocks

    # --- Experiment escape hatch (FILM_CS_PASS_ALL_YES=1) -------------------
    # Dump ALL stage-1 YES segments WHOLE to the interpreter (no per-segment
    # compaction, no max_segments cap), ordered by reading position, capped
    # only by max_total_chars. Tests whether the 2/5-paragraph compaction is
    # dropping answer-bearing paragraphs.
    if os.environ.get("FILM_CS_PASS_ALL_YES") == "1":
        yes_segs = sorted((by_seg[sid] for sid in yes_ids), key=lambda s: s.start_para)
        blocks, used = [], 0
        for seg in yes_segs:
            src = "finder=YES"
            hdr = f"--- Segment (paragraphs {seg.start_para}-{seg.end_para}; {src}) ---"
            block = f"{hdr}\n{seg.text}"
            if used + len(block) > max_total_chars:
                blocks.append(f"--- (... {len(yes_segs) - len(blocks)} more YES segments truncated for length ...)")
                break
            blocks.append(block)
            used += len(block)
        if blocks:
            return "\n\n".join(blocks), len(blocks)
        # fall through to normal logic if no YES at all
    # -----------------------------------------------------------------------

    selected_ids: list[int] = []
    selected_set: set[int] = set()
    suppress_micro_retrieval = selection_mode in {"support_ranked", "chronological"}

    def add_selected(sid: int) -> None:
        if sid in by_seg and sid not in selected_set and len(selected_ids) < max_segments:
            if suppress_micro_retrieval and _is_micro_retrieval(by_seg[sid]):
                return
            selected_ids.append(sid)
            selected_set.add(sid)

    ranked_yes = sorted(yes_ids, key=lambda sid: rank_scores.get(sid, 0.0), reverse=True)
    if selection_mode == "chronological":
        selected_ids = _seed_chronological_selection(
            segments,
            yes_ids,
            adjusted_scores,
            max_segments,
            bridge_enabled=bridge_enabled,
            prefer_late=prefer_late,
        )
        selected_set = set(selected_ids)
    else:
        for sid in ranked_yes[:2]:
            add_selected(sid)

        if bridge_enabled:
            for sid in list(selected_ids):
                base = by_seg[sid]
                bridge_candidates = sorted(
                    (
                        s for s in segments
                        if s.seg_id in yes_ids
                        and s.seg_id not in selected_set
                        and (
                            0 <= s.start_para - base.end_para <= 2
                            or 0 <= base.start_para - s.end_para <= 2
                        )
                        and RE_BRIDGE_TEXT.search(s.text)
                    ),
                    key=lambda s: rank_scores.get(s.seg_id, 0.0),
                    reverse=True,
                )
                for seg in bridge_candidates[:1]:
                    add_selected(seg.seg_id)
        if tail_focus:
            for sid in (sid for sid in ranked_yes if "tail" in by_seg[sid].kind):
                add_selected(sid)
                break

        if option_bias:
            for letter in "ABCD":
                opt = options.get(letter, "").strip()
                if not opt or len(selected_ids) >= max_segments:
                    continue
                opt_query = _query_counter(question, {letter: opt})
                opt_literal_query = _tokenize(opt)
                opt_literal = Counter(opt_literal_query)
                ranked_for_option = sorted(
                    segments,
                    key=lambda s: (
                        1 if tail_bias and "tail" in s.kind else 0,
                        1 if opt_literal and _dice_score(opt_literal, s.text) > 0 else 0,
                        s.start_para if tail_bias else 0,
                        _dice_score(opt_literal, s.text) if opt_literal else 0,
                        _dice_score(opt_query, s.text),
                        1 if s.kind.startswith("retrieval") else 0,
                        1 if s.seg_id in yes_ids else 0,
                    ),
                    reverse=True,
                )
                for seg in ranked_for_option:
                    opt_score = _dice_score(opt_query, seg.text)
                    if opt_score <= 0:
                        break
                    if seg.kind.startswith("retrieval") or seg.seg_id in yes_ids:
                        add_selected(seg.seg_id)
                        break

        for sid in ranked_yes:
            add_selected(sid)
            if len(selected_ids) >= max_segments:
                break

    if len(selected_ids) < min(min_segments, len(segments)):
        ranked_all = sorted(segments, key=lambda s: scores.get(s.seg_id, 0.0), reverse=True)
        for seg in ranked_all:
            if scores.get(seg.seg_id, 0.0) <= 0 and selected_ids:
                continue
            add_selected(seg.seg_id)
            if len(selected_ids) >= min(min_segments, len(segments)):
                break

    if not selected_ids and segments:
        add_selected(max(segments, key=lambda s: scores.get(s.seg_id, 0.0)).seg_id)
    if not selected_ids and suppress_micro_retrieval:
        suppress_micro_retrieval = False
        for sid in ranked_yes:
            add_selected(sid)
            if len(selected_ids) >= min(min_segments, len(segments)):
                break

    kept = [(by_seg[sid], decision_by_seg.get(sid)) for sid in selected_ids if sid in by_seg]
    kept.sort(key=lambda sd: sd[0].start_para)

    retrieval_max_chars = _env_int("FILM_CS_RETRIEVAL_SEGMENT_MAX_CHARS", 4500)
    retrieval_max_paragraphs = _env_int("FILM_CS_RETRIEVAL_SEGMENT_MAX_PARAGRAPHS", 2)
    window_max_chars = _env_int("FILM_CS_WINDOW_SEGMENT_MAX_CHARS", 3000)
    window_max_paragraphs = _env_int("FILM_CS_WINDOW_SEGMENT_MAX_PARAGRAPHS", 5)

    blocks = []
    used = 0
    n_blocks = 0
    for seg, d in kept:
        source = "finder=YES" if d and d.verdict == "YES" else "finder=LEXICAL-RESCUE"
        hdr = f"--- Segment (paragraphs {seg.start_para}-{seg.end_para}; {source}) ---"
        if seg.kind.startswith("retrieval"):
            body = _compact_segment_text(
                seg.text,
                query,
                max_chars=retrieval_max_chars,
                max_paragraphs=retrieval_max_paragraphs,
                expand_neighbors=neighbor_expand,
                priority_para_ids=cited_para_by_seg.get(seg.seg_id, set()),
            )
        else:
            body = _compact_segment_text(
                seg.text,
                query,
                max_chars=window_max_chars,
                max_paragraphs=window_max_paragraphs,
                expand_neighbors=neighbor_expand,
                priority_para_ids=cited_para_by_seg.get(seg.seg_id, set()),
            )
        block = f"{hdr}\n{body}"
        if used + len(block) > max_total_chars:
            blocks.append(f"--- (... {len(kept) - len(blocks)} more kept segments truncated for length ...)")
            break
        blocks.append(block)
        used += len(block)
        n_blocks += 1
    return "\n\n".join(blocks), n_blocks


async def interpret(
    client: LLMClient,
    question: str,
    options: dict[str, str],
    segments: list[Segment],
    decisions: list[FinderDecision],
    max_tokens: int = 768,
    binary_mode: bool = False,
    verify: bool = True,
) -> InterpreterOutput:
    max_tokens = _env_int("FILM_CS_INTERPRETER_MAX_TOKENS", max_tokens)
    verifier_max_tokens = _env_int("FILM_CS_VERIFIER_MAX_TOKENS", 512)
    if binary_mode:
        evidence_max_chars = _env_int("FILM_CS_BINARY_EVIDENCE_MAX_CHARS", 13000)
        evidence_max_segments = _env_int("FILM_CS_BINARY_EVIDENCE_MAX_SEGMENTS", 5)
        evidence_min_segments = _env_int("FILM_CS_BINARY_EVIDENCE_MIN_SEGMENTS", 3)
    else:
        evidence_max_chars = _env_int("FILM_CS_EVIDENCE_MAX_CHARS", 13000)
        evidence_max_segments = _env_int("FILM_CS_EVIDENCE_MAX_SEGMENTS", 5)
        evidence_min_segments = _env_int("FILM_CS_EVIDENCE_MIN_SEGMENTS", 3)
    evidence, n_evidence_segments = _build_evidence_block(
        segments,
        decisions,
        question,
        options,
        max_total_chars=evidence_max_chars,
        max_segments=evidence_max_segments,
        min_segments=evidence_min_segments,
    )
    if binary_mode:
        prompt = INTERPRETER_BINARY_USER_TEMPLATE.format(
            question=question, evidence_block=evidence,
        )
        sys_prompt = INTERPRETER_BINARY_SYSTEM
        parse_re = RE_INTERPRETER_BINARY_PARSE
        valid_letters = {"TRUE", "FALSE"}
    else:
        prompt = INTERPRETER_USER_TEMPLATE.format(
            question=question,
            opt_a=options.get("A", ""),
            opt_b=options.get("B", ""),
            opt_c=options.get("C", ""),
            opt_d=options.get("D", ""),
            evidence_block=evidence,
        )
        sys_prompt = INTERPRETER_SYSTEM
        parse_re = RE_INTERPRETER_PARSE
        valid_letters = {"A", "B", "C", "D"}
    raw = await client.ask(prompt, system=sys_prompt, max_tokens=max_tokens)
    m = parse_re.search(raw)
    if m:
        answer = m.group(2).upper()
        reason_text = m.group(1).strip()
        if verify and binary_mode:
            verify_prompt = INTERPRETER_BINARY_VERIFIER_USER_TEMPLATE.format(
                question=question,
                evidence_block=evidence,
                previous_answer=answer,
                previous_reason=reason_text,
            )
            verify_raw = await client.ask(verify_prompt, system=INTERPRETER_BINARY_VERIFIER_SYSTEM, max_tokens=verifier_max_tokens)
            vm = RE_INTERPRETER_BINARY_PARSE.search(verify_raw)
            if vm:
                return InterpreterOutput(
                    answer=vm.group(2).upper(),
                    reason=vm.group(1).strip(),
                    raw=raw + "\n\n[verifier]\n" + verify_raw,
                    n_evidence_segments=n_evidence_segments,
                    evidence_chars=len(evidence),
                    evidence=evidence,
                )
        if verify and not binary_mode and RE_VERIFY_TRIGGER.search(question):
            verify_prompt = INTERPRETER_VERIFIER_USER_TEMPLATE.format(
                question=question,
                opt_a=options.get("A", ""),
                opt_b=options.get("B", ""),
                opt_c=options.get("C", ""),
                opt_d=options.get("D", ""),
                evidence_block=evidence,
                previous_answer=answer,
                previous_reason=reason_text,
            )
            verify_raw = await client.ask(verify_prompt, system=INTERPRETER_VERIFIER_SYSTEM, max_tokens=verifier_max_tokens)
            vm = RE_INTERPRETER_PARSE.search(verify_raw)
            if vm:
                return InterpreterOutput(
                    answer=vm.group(2).upper(),
                    reason=vm.group(1).strip(),
                    raw=raw + "\n\n[verifier]\n" + verify_raw,
                    n_evidence_segments=n_evidence_segments,
                    evidence_chars=len(evidence),
                    evidence=evidence,
                )
        return InterpreterOutput(
            answer=answer,
            reason=reason_text,
            raw=raw,
            n_evidence_segments=n_evidence_segments,
            evidence_chars=len(evidence),
            evidence=evidence,
        )
    # loose fallback
    letter = ""
    if binary_mode:
        matches = re.findall(r"\b(TRUE|FALSE)\b", raw.upper())
        if matches:
            letter = matches[-1]
    else:
        answer_like = re.findall(r"(?:answer|ANSWER|答案)\s*[:：]?\s*[\(\[]?\s*([ABCD])\b", raw)
        if answer_like:
            letter = answer_like[-1].upper()
        else:
            tail_letters = re.findall(r"\b([ABCD])\b", raw[-200:])
            if tail_letters:
                letter = tail_letters[-1].upper()
    return InterpreterOutput(
        answer=letter,
        reason="(parse-fail)",
        raw=raw,
        n_evidence_segments=n_evidence_segments,
        evidence_chars=len(evidence),
        evidence=evidence,
    )
