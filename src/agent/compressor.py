"""Question-conditioned compressor.

Given a segment of the novel + a question, produce a focused summary that
preserves information relevant to the question (and explicitly says
'NOT_RELEVANT' if the segment is irrelevant, which lets the Interpreter skip it).
"""
from __future__ import annotations
import asyncio
from dataclasses import dataclass

from src.agent.segmenter import Segment
from src.llm_client import LLMClient


COMPRESS_SYSTEM = (
    "You are a focused reader for a detective-novel QA system. "
    "Given a segment of a novel and a multiple-choice question, "
    "extract only the information from the segment that is relevant to answering the question. "
    "Keep character names, actions, and crucial details. "
    "If the segment contains nothing relevant, respond with exactly: NOT_RELEVANT"
)

COMPRESS_TEMPLATE = """## Question
{question}

## Options
A. {opt_a}
B. {opt_b}
C. {opt_c}
D. {opt_d}

## Novel Segment (paragraphs {start}-{end})
{segment_text}

## Task
Write a concise summary (<=120 words) of ONLY the information in this segment that is relevant to answering the question. Use paragraph numbers like [N] when citing evidence. If nothing in this segment helps answer the question, output exactly: NOT_RELEVANT
"""


@dataclass
class CompressedSegment:
    seg_id: int
    start_para: int
    end_para: int
    summary: str  # 'NOT_RELEVANT' if irrelevant
    is_relevant: bool


def _format_prompt(seg: Segment, question: str, options: dict[str, str]) -> str:
    return COMPRESS_TEMPLATE.format(
        question=question,
        opt_a=options.get("A", ""),
        opt_b=options.get("B", ""),
        opt_c=options.get("C", ""),
        opt_d=options.get("D", ""),
        start=seg.start_para,
        end=seg.end_para,
        segment_text=seg.text,
    )


async def compress_segment(
    client: LLMClient,
    seg: Segment,
    question: str,
    options: dict[str, str],
    max_tokens: int = 256,
) -> CompressedSegment:
    prompt = _format_prompt(seg, question, options)
    out = await client.ask(prompt, system=COMPRESS_SYSTEM, max_tokens=max_tokens)
    out_clean = out.strip()
    is_relevant = "NOT_RELEVANT" not in out_clean.upper().split("\n")[0]
    return CompressedSegment(
        seg_id=seg.seg_id,
        start_para=seg.start_para,
        end_para=seg.end_para,
        summary=out_clean,
        is_relevant=is_relevant,
    )


async def compress_all(
    client: LLMClient,
    segments: list[Segment],
    question: str,
    options: dict[str, str],
) -> list[CompressedSegment]:
    tasks = [compress_segment(client, s, question, options) for s in segments]
    return await asyncio.gather(*tasks)
