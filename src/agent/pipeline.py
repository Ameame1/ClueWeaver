"""ClueWeaver two-agent orchestration.

For each question:
  1. Build retrieval-aware narrative segments.
  2. The Finder judges whether each segment contains useful clues.
  3. The Interpreter reads selected evidence and outputs a grounded answer.
"""
from __future__ import annotations
import time
from dataclasses import dataclass

from src.agent.segmenter import paragraphs_to_segments, retrieval_aware_segments
from src.agent.finder import find_all
from src.agent.interpreter import interpret
from src.llm_client import LLMClient


@dataclass
class PipelineResult:
    novel_id: int
    qid: int
    predicted: str            # "A"/"B"/"C"/"D" or ""
    gold: str
    correct: bool
    n_segments: int
    n_yes: int
    n_no: int
    n_parse_fail: int
    n_retrieval_segments: int
    n_evidence_segments: int
    evidence_chars: int
    evidence: str
    rationale: str
    raw_interpreter_output: str
    latency_sec: float
    seg_sec: float = 0.0
    finder_sec: float = 0.0
    interpreter_sec: float = 0.0


async def run_question(
    client: LLMClient,
    paragraphs: list[str],
    question: str,
    options: dict[str, str],
    gold: str,
    novel_id: int = 0,
    qid: int = 0,
    paras_per_segment: int = 60,
    overlap_paras: int = 5,
    binary_mode: bool = False,
    retrieval_enabled: bool = True,
    max_segment_words: int = 1600,
    bm25_top_k: int = 4,
    dense_top_k: int = 4,
    dense_candidates: int = 48,
    option_bm25_top_k: int = 2,
    option_dense_top_k: int = 1,
    option_dense_candidates: int = 32,
    tail_bm25_top_k: int = 4,
    option_tail_bm25_top_k: int = 8,
    finder_client: LLMClient | None = None,
    binary_finder_client: LLMClient | None = None,
    self_calibration: bool = True,
) -> PipelineResult:
    t0 = time.time()
    _t_seg = time.time()
    if retrieval_enabled:
        segments = retrieval_aware_segments(
            paragraphs,
            question,
            options,
            paras_per_segment=paras_per_segment,
            overlap_paras=overlap_paras,
            max_segment_words=max_segment_words,
            bm25_top_k=bm25_top_k,
            dense_top_k=dense_top_k,
            dense_candidates=dense_candidates,
            option_bm25_top_k=option_bm25_top_k,
            option_dense_top_k=option_dense_top_k,
            option_dense_candidates=option_dense_candidates,
            tail_bm25_top_k=tail_bm25_top_k,
            option_tail_bm25_top_k=option_tail_bm25_top_k,
            use_dense=True,
        )
    else:
        segments = paragraphs_to_segments(
            paragraphs,
            paras_per_segment=paras_per_segment,
            overlap_paras=overlap_paras,
            max_segment_words=max_segment_words,
        )
    seg_sec = time.time() - _t_seg
    active_finder_client = (
        binary_finder_client
        if binary_mode and binary_finder_client is not None
        else (finder_client or client)
    )
    _t_finder = time.time()
    decisions = await find_all(active_finder_client, segments, question, options)
    finder_sec = time.time() - _t_finder
    _t_interpreter = time.time()
    iout = await interpret(
        client,
        question,
        options,
        segments,
        decisions,
        binary_mode=binary_mode,
        verify=self_calibration,
    )
    interpreter_sec = time.time() - _t_interpreter

    n_yes = sum(1 for d in decisions if d.verdict == "YES")
    n_no = sum(1 for d in decisions if d.verdict == "NO")
    n_fail = sum(1 for d in decisions if d.verdict == "")

    return PipelineResult(
        novel_id=novel_id,
        qid=qid,
        predicted=iout.answer,
        gold=gold,
        correct=(iout.answer == gold),
        n_segments=len(segments),
        n_yes=n_yes,
        n_no=n_no,
        n_parse_fail=n_fail,
        n_retrieval_segments=sum(1 for s in segments if s.kind.startswith("retrieval")),
        n_evidence_segments=iout.n_evidence_segments,
        evidence_chars=iout.evidence_chars,
        evidence=iout.evidence,
        rationale=iout.reason,
        raw_interpreter_output=iout.raw,
        latency_sec=time.time() - t0,
        seg_sec=seg_sec,
        finder_sec=finder_sec,
        interpreter_sec=interpreter_sec,
    )
