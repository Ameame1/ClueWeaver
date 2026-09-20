"""Finder agent — judges whether a single segment is relevant to a question.

Output schema (strict):
    <reason>brief reasoning, cite paragraph numbers like [N]</reason>
    <answer>YES</answer>     or     <answer>NO</answer>
"""
from __future__ import annotations
import asyncio
import os
import re
from dataclasses import dataclass

from src.agent.segmenter import Segment
from src.llm_client import LLMClient


FINDER_SYSTEM = (
    "You are the evidence Finder agent in a long-narrative QA and claim-verification pipeline.\n"
    "You read ONE segment from a long story (paragraphs are numbered like [N]) and decide whether it "
    "contains concrete evidence that should be shown to a downstream Interpreter.\n"
    "\n"
    "Say YES only when the segment contains a concrete fact that helps answer the question, choose/rule out "
    "one option, or confirm/refute the claim. Strong evidence includes a relevant action, dialogue line, motive, "
    "relationship, causal explanation, time/place clue, object, identity, or explicit contradiction.\n"
    "\n"
    "Say NO when:\n"
    "  - The segment is scene-setting, scenery, weather, or transition narrative.\n"
    "  - The segment mentions characters but says nothing about what the question is asking.\n"
    "  - The overlap is only a common word, option word, or passing name with no relevant fact.\n"
    "  - The segment merely raises suspicion but gives no fact that distinguishes options.\n"
    "  - You cannot name a concrete clue from the segment.\n"
    "\n"
    "Bias: preserve answer-critical evidence. Prefer NO for pure background, but choose YES for any concrete "
    "fact that could help answer, rule out an option, confirm, or refute the claim.\n"
    "Judge this segment independently; do not follow a fixed YES rate. Partial evidence is still evidence.\n"
    "\n"
    "OUTPUT FORMAT — exactly two XML blocks and NOTHING else (no preamble, no markdown, no extra text):\n"
    "  <reason>one sentence naming the concrete clue, or saying no concrete clue is present; cite [N] when applicable</reason>\n"
    "  <answer>YES</answer>     or     <answer>NO</answer>\n"
    "\n"
    "EXAMPLE OUTPUT (relevant):\n"
    "  <reason>Paragraph [478] shows Mrs. Marshall asking Poirot for secrecy, directly addressing the question about her motive.</reason>\n"
    "  <answer>YES</answer>\n"
    "\n"
    "EXAMPLE OUTPUT (irrelevant):\n"
    "  <reason>Segment describes background scenery of the beach with no mention of any character or action relevant to the question.</reason>\n"
    "  <answer>NO</answer>"
)

FINDER_USER_TEMPLATE = """Question: {question}

Options:
  (A) {opt_a}
  (B) {opt_b}
  (C) {opt_c}
  (D) {opt_d}

Segment (paragraphs {start}-{end}):
{segment_text}

Decide whether this segment is relevant. Output only the two XML tags."""


FINDER_BINARY_USER_TEMPLATE = """Claim to verify: {question}

Segment (paragraphs {start}-{end}):
{segment_text}

Decide whether this segment contains evidence that helps confirm or refute the claim.
Output only the two XML tags."""


RE_FINDER_PARSE = re.compile(
    r"<reason>(.+?)</reason>\s*<answer>\s*(YES|NO)\s*</answer>",
    re.S | re.I,
)
RE_FINDER_ANSWER = re.compile(r"<answer>\s*(YES|NO)\s*</answer>", re.S | re.I)
RE_FINDER_REASON = re.compile(r"<reason>(.+?)</reason>", re.S | re.I)
RE_FINDER_RELEVANT = re.compile(r"<relevant>\s*(TRUE|FALSE|YES|NO)\s*</relevant>", re.S | re.I)


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


@dataclass
class FinderDecision:
    seg_id: int
    start_para: int
    end_para: int
    verdict: str  # "YES" / "NO" / "" (parse fail)
    reason: str
    raw: str


def _format_prompt(seg: Segment, question: str, options: dict[str, str]) -> str:
    if not options:  # binary T/F mode
        return FINDER_BINARY_USER_TEMPLATE.format(
            question=question, start=seg.start_para, end=seg.end_para, segment_text=seg.text,
        )
    return FINDER_USER_TEMPLATE.format(
        question=question,
        opt_a=options.get("A", ""),
        opt_b=options.get("B", ""),
        opt_c=options.get("C", ""),
        opt_d=options.get("D", ""),
        start=seg.start_para,
        end=seg.end_para,
        segment_text=seg.text,
    )


async def find_segment(
    client: LLMClient,
    seg: Segment,
    question: str,
    options: dict[str, str],
    max_tokens: int = 192,
) -> FinderDecision:
    max_tokens = _env_int("FILM_CS_FINDER_MAX_TOKENS", max_tokens)
    prompt = _format_prompt(seg, question, options)
    raw = await client.ask(prompt, system=FINDER_SYSTEM, max_tokens=max_tokens)
    m = RE_FINDER_PARSE.search(raw)
    if m:
        return FinderDecision(
            seg_id=seg.seg_id,
            start_para=seg.start_para,
            end_para=seg.end_para,
            verdict=m.group(2).upper(),
            reason=m.group(1).strip(),
            raw=raw,
        )
    answer_m = RE_FINDER_ANSWER.search(raw)
    relevant_m = RE_FINDER_RELEVANT.search(raw)
    reason_m = RE_FINDER_REASON.search(raw)
    if answer_m or relevant_m:
        if answer_m:
            verdict = answer_m.group(1).upper()
        else:
            verdict_raw = relevant_m.group(1).upper()
            verdict = "YES" if verdict_raw in {"TRUE", "YES"} else "NO"
        return FinderDecision(
            seg_id=seg.seg_id,
            start_para=seg.start_para,
            end_para=seg.end_para,
            verdict=verdict,
            reason=reason_m.group(1).strip() if reason_m else "(parse-fail)",
            raw=raw,
        )
    # parse fail — try loose fallback: look for any YES/NO token in output
    fallback = ""
    txt = raw.upper()
    if "<ANSWER>YES" in txt or "ANSWER: YES" in txt or "\nYES" in txt[-50:]:
        fallback = "YES"
    elif "<ANSWER>NO" in txt or "ANSWER: NO" in txt or "\nNO" in txt[-50:]:
        fallback = "NO"
    return FinderDecision(
        seg_id=seg.seg_id,
        start_para=seg.start_para,
        end_para=seg.end_para,
        verdict=fallback,
        reason="(parse-fail)",
        raw=raw,
    )


async def find_all(
    client: LLMClient,
    segments: list[Segment],
    question: str,
    options: dict[str, str],
) -> list[FinderDecision]:
    tasks = [find_segment(client, s, question, options) for s in segments]
    return await asyncio.gather(*tasks)
