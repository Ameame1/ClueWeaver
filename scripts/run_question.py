"""Run one JSON question through the released Finder and Interpreter."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
import json
from pathlib import Path

from src.agent.pipeline import run_question
from src.llm_client import get_client


def load_question(path: Path) -> dict:
    record = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(record, dict):
        raise ValueError("Input must be a JSON object")
    paragraphs = record.get("paragraphs")
    if not isinstance(paragraphs, list) or not paragraphs or not all(
        isinstance(p, str) and p.strip() for p in paragraphs
    ):
        raise ValueError("paragraphs must be a nonempty list of nonempty strings")
    if not isinstance(record.get("question"), str) or not record["question"].strip():
        raise ValueError("question must be a nonempty string")
    binary = record.get("binary_mode", False)
    if not isinstance(binary, bool):
        raise ValueError("binary_mode must be a JSON boolean")
    options = record.get("options", {})
    if not isinstance(options, dict):
        raise ValueError("options must be a JSON object")
    if binary:
        if options:
            raise ValueError("Binary claims require options={} (labels are TRUE/FALSE)")
        valid = {"TRUE", "FALSE"}
    else:
        if set(options) != {"A", "B", "C", "D"} or not all(
            isinstance(v, str) and v.strip() for v in options.values()
        ):
            raise ValueError("Multiple-choice questions require nonempty A, B, C, D options")
        valid = set(options)
    gold = record.get("gold", "")
    if not isinstance(gold, str) or (gold and gold not in valid):
        raise ValueError(f"gold must be empty or one of {sorted(valid)}")
    return {
        "paragraphs": paragraphs, "question": record["question"],
        "options": options, "binary_mode": binary, "gold": gold,
    }


async def run(record: dict, retrieval_enabled: bool) -> dict:
    finder = get_client("qwen3-4b-finder", max_concurrency=1)
    interpreter = get_client("qwen3-4b", max_concurrency=1)
    try:
        result = await run_question(
            client=interpreter, finder_client=finder,
            retrieval_enabled=retrieval_enabled, **record,
        )
        output = asdict(result)
        if not record["gold"]:
            output["correct"] = None
        return output
    finally:
        await asyncio.gather(finder.aclose(), interpreter.aclose())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--no-retrieval", action="store_true", help="Toy smoke checks only; not the paper pipeline")
    args = parser.parse_args()
    try:
        record = load_question(args.input)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    if args.validate_only:
        print("Input valid; no model requests sent.")
        return
    output = asyncio.run(run(record, retrieval_enabled=not args.no_retrieval))
    text = json.dumps(output, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
