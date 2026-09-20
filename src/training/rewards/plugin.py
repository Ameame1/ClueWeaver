"""Final ClueWeaver GRPO rewards.

This submission package keeps only the two reward functions used by the final
paper checkpoints:

  - film_cs_finder_reward_v3: Finder reward for clue selection.
  - film_cs_interpreter_reward_v5: Interpreter reward for grounded answering.

The functions are registered in the optional trainer registry when available,
while still allowing local import tests without training dependencies installed.
"""
from __future__ import annotations

import os
import re
from numbers import Integral
from typing import Any, List

try:
    from swift.rewards.orm import ORM, orms
except ModuleNotFoundError:
    class ORM:
        def __init__(self, *args, **kwargs):
            pass

    orms = {}


RE_FINDER = re.compile(
    r"^\s*<reason>(.+?)</reason>\s*<answer>\s*(YES|NO)\s*</answer>\s*$",
    re.S | re.I,
)
RE_INTERPRETER = re.compile(
    r"<reason>(.+?)</reason>\s*<answer>\s*([ABCD])\s*</answer>",
    re.S | re.I,
)
RE_INTERPRETER_BINARY = re.compile(
    r"<reason>(.+?)</reason>\s*<answer>\s*(TRUE|FALSE)\s*</answer>",
    re.S | re.I,
)
RE_THINK_PREFIX = re.compile(r"^\s*<think>.*?</think>\s*", re.S | re.I)
RE_PARA = re.compile(r"\[(\d+)\]")
RE_WORD = re.compile(r"[A-Za-z][A-Za-z'-]{1,}")
RE_NEGATIVE_RATIONALE = re.compile(
    r"\b(no|not|irrelevant|unrelated|background|scene-setting|transition|"
    r"passing mention|no concrete clue|without concrete evidence|does not address|"
    r"cannot answer|cannot confirm|cannot refute)\b",
    re.I,
)
RE_QUOTED = re.compile(r'"[^"]{2,80}"|\'[^\']{2,80}\'')
RE_NUMBER = re.compile(r"\b(?:\d{2,5}|\d+(?:st|nd|rd|th))\b")
RE_PROPER_NOUN_PAIR = re.compile(r"\b([A-Z][a-z]+ [A-Z][a-z]+)\b")
RE_HEDGE = re.compile(
    r"\b(uncertain|cannot determine|insufficient evidence|the evidence is unclear|"
    r"I am unsure|hard to say|impossible to tell|no clear answer|inconclusive)\b",
    re.I,
)
STOP_TOKENS = ("<|im_end|>", "<|endoftext|>")


def _clean_completion(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.strip()
    for token in STOP_TOKENS:
        text = text.replace(token, "")
    return RE_THINK_PREFIX.sub("", text, count=1).strip()


def _coerce_token_id(value: Any):
    if isinstance(value, Integral):
        return int(value)
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu()
    if hasattr(value, "item"):
        try:
            item = value.item()
        except Exception:
            return None
        if isinstance(item, Integral):
            return int(item)
    return None


def _completion_to_text(completion: Any, tokenizer=None) -> str:
    if isinstance(completion, str):
        return completion
    token_id = _coerce_token_id(completion)
    if token_id is not None:
        return tokenizer.decode([token_id], skip_special_tokens=False) if tokenizer else ""
    if hasattr(completion, "detach") and hasattr(completion, "cpu"):
        completion = completion.detach().cpu().tolist()
    elif hasattr(completion, "tolist") and not isinstance(completion, (dict, list, tuple)):
        completion = completion.tolist()
    if isinstance(completion, dict):
        for key in ("content", "text"):
            value = completion.get(key)
            if isinstance(value, str):
                return value
        completion = completion.get("token_ids", completion.get("input_ids", completion))
    if isinstance(completion, tuple):
        completion = list(completion)
    if isinstance(completion, list):
        token_ids = [_coerce_token_id(x) for x in completion]
        if token_ids and all(x is not None for x in token_ids):
            return tokenizer.decode(token_ids, skip_special_tokens=False) if tokenizer else ""
        return "\n".join(x for x in (_completion_to_text(x, tokenizer) for x in completion) if x)
    return ""


def _load_tokenizer(args):
    model = getattr(args, "model", None) if args is not None else None
    if not model:
        return None
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(model, trust_remote_code=True, local_files_only=True)
    except Exception:
        return None


def _debug_reward_call(name: str, completions, texts, scores) -> None:
    path = os.environ.get("FILM_CS_REWARD_DEBUG")
    if not path:
        return
    with open(path, "a", encoding="utf-8") as f:
        for text, score in zip(texts[:4], scores[:4]):
            f.write(f"{name}\t{score}\t{text[:240].replace(chr(10), ' ')}\n")


def _as_bool_label(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _as_int_set(values) -> set[int]:
    if values is None:
        return set()
    if isinstance(values, str):
        return {int(x) for x in RE_PARA.findall(values)}
    return {int(x) for x in values}


def _word_count(text: str) -> int:
    return len(RE_WORD.findall(text or ""))


def _citation_f1(predicted: set[int], gold_set: set[int]) -> float:
    if not gold_set and not predicted:
        return 1.0
    if not gold_set or not predicted:
        return 0.0
    tp = len(predicted & gold_set)
    if tp == 0:
        return 0.0
    precision = tp / len(predicted)
    recall = tp / len(gold_set)
    return 2 * precision * recall / (precision + recall)


def _gold_answer_from_fields(gold_letter=None, gold_bool=None, gold_answer=None) -> str:
    if gold_answer is not None and str(gold_answer).strip():
        return str(gold_answer).strip().upper()
    if gold_bool is not None:
        return "TRUE" if _as_bool_label(gold_bool) else "FALSE"
    if gold_letter is not None:
        return str(gold_letter).strip().upper()
    return ""


def _has_specificity(reason: str) -> bool:
    return bool(
        RE_QUOTED.search(reason)
        or RE_NUMBER.search(reason)
        or RE_PROPER_NOUN_PAIR.search(reason)
    )


def _finder_reward_v3(completion: str, gold_label: bool, gold_paragraphs_in_segment) -> float:
    """Finder reward: valid XML, calibrated YES/NO, and faithful paragraph IDs."""
    completion = _clean_completion(completion)
    match = RE_FINDER.search(completion)
    if not match:
        return 0.0

    score = 1.0
    reason = match.group(1).strip()
    verdict = match.group(2).upper()
    is_positive = _as_bool_label(gold_label)
    if (verdict == "YES") != is_positive:
        return score

    predicted = {int(x) for x in RE_PARA.findall(reason)}
    gold_set = _as_int_set(gold_paragraphs_in_segment)

    if is_positive:
        score += 1.5
        f1 = _citation_f1(predicted, gold_set) if gold_set else 0.0
        score += 1.5 * f1
        if f1 > 0 and len(predicted) <= max(3, len(gold_set) + 1):
            score += 0.25
    else:
        score += 1.25
        wc = _word_count(reason)
        no_citation = not predicted
        if no_citation and wc >= 6:
            score += 0.50
        if no_citation and RE_NEGATIVE_RATIONALE.search(reason):
            score += 0.25
        if 6 <= wc <= 45:
            score += 0.25
        if predicted:
            score += max(0.0, 0.20 - 0.10 * len(predicted))
    return score


def _interpreter_reward_v5(
    completion: str,
    gold_letter=None,
    gold_bool=None,
    gold_answer=None,
    case_type=None,
    evidence_paragraphs=None,
) -> float:
    """Interpreter reward: correctness first, then grounded concise support."""
    completion = _clean_completion(completion)
    gold = _gold_answer_from_fields(gold_letter=gold_letter, gold_bool=gold_bool, gold_answer=gold_answer)
    is_binary = gold in {"TRUE", "FALSE"}
    match = RE_INTERPRETER_BINARY.search(completion) if is_binary else RE_INTERPRETER.search(completion)
    if not match:
        return 0.0

    reason = match.group(1).strip()
    pred = match.group(2).upper()
    score = 1.0
    correct = bool(gold) and pred == gold

    if correct:
        score += 1.5 if is_binary else 2.5
        normalized_case = str(case_type or "").replace("reason" + "er", "interpreter")
        if normalized_case == "hard_interpreter_error":
            score += 0.5

        cites = {int(x) for x in RE_PARA.findall(reason)}
        if cites:
            score += 0.3
            if evidence_paragraphs:
                try:
                    allowed = {int(x) for x in evidence_paragraphs}
                    if cites - allowed:
                        score -= 0.3
                except (TypeError, ValueError):
                    pass

        wc = _word_count(reason)
        if 30 <= wc <= 120:
            score += 0.2
        if _has_specificity(reason):
            score += 0.3

    if RE_HEDGE.search(reason):
        score -= 0.4
    return max(0.0, score)


class FilmCSFinderRewardV3(ORM):
    """Finder reward used by the final ClueWeaver Finder checkpoint."""

    def __init__(self, args=None, **kwargs):
        super().__init__(args, **kwargs)
        self.tokenizer = _load_tokenizer(args)

    def __call__(self, completions: List[str], gold_label=None, gold_paragraphs_in_segment=None, **kwargs) -> List[float]:
        scores = []
        texts = []
        for i, completion in enumerate(completions):
            text = _completion_to_text(completion, self.tokenizer)
            texts.append(text)
            gl = gold_label[i] if gold_label is not None else None
            gp = gold_paragraphs_in_segment[i] if gold_paragraphs_in_segment is not None else []
            scores.append(_finder_reward_v3(text, gl, gp))
        _debug_reward_call("film_cs_finder_reward_v3", completions, texts, scores)
        return scores


class FilmCSInterpreterRewardV5(ORM):
    """Interpreter reward used by the final ClueWeaver Interpreter checkpoint."""

    def __init__(self, args=None, **kwargs):
        super().__init__(args, **kwargs)
        self.tokenizer = _load_tokenizer(args)

    def __call__(
        self,
        completions: List[str],
        gold_letter=None,
        gold_bool=None,
        gold_answer=None,
        case_type=None,
        evidence_paragraphs=None,
        **kwargs,
    ) -> List[float]:
        scores = []
        texts = []
        for i, completion in enumerate(completions):
            text = _completion_to_text(completion, self.tokenizer)
            texts.append(text)
            scores.append(
                _interpreter_reward_v5(
                    text,
                    gold_letter=gold_letter[i] if gold_letter is not None else None,
                    gold_bool=gold_bool[i] if gold_bool is not None else None,
                    gold_answer=gold_answer[i] if gold_answer is not None else None,
                    case_type=case_type[i] if case_type is not None else None,
                    evidence_paragraphs=evidence_paragraphs[i] if evidence_paragraphs is not None else None,
                )
            )
        _debug_reward_call("film_cs_interpreter_reward_v5", completions, texts, scores)
        return scores


orms["film_cs_finder_reward_v3"] = FilmCSFinderRewardV3
orms["film_cs_interpreter_reward_v5"] = FilmCSInterpreterRewardV5
