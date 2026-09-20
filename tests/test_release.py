"""Offline contract checks; these do not measure model accuracy."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from scripts.run_question import load_question, run
from src.agent.pipeline import run_question
from src.llm_client import LLMClient, ModelSpec


ROOT = Path(__file__).resolve().parents[1]


class InputTests(unittest.TestCase):
    def test_examples(self):
        for name in ("multiple_choice", "binary_claim"):
            record = load_question(ROOT / "examples" / f"{name}.json")
            self.assertTrue(record["paragraphs"])

    def test_binary_rejects_choice_options(self):
        record = json.loads((ROOT / "examples/binary_claim.json").read_text())
        record["options"] = {"A": "TRUE", "B": "FALSE"}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "input.json"
            path.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, "options"):
                load_question(path)


class PipelineTests(unittest.IsolatedAsyncioTestCase):
    async def test_cli_unlabeled_result_and_client_cleanup(self):
        record = load_question(ROOT / "examples/multiple_choice.json")
        record["gold"] = ""
        finder = SimpleNamespace(
            ask=AsyncMock(return_value='<reason>Clue in [0].</reason><answer>YES</answer>'),
            aclose=AsyncMock(),
        )
        reader = SimpleNamespace(
            ask=AsyncMock(return_value='<reason>Clue in [0].</reason><answer>B</answer>'),
            aclose=AsyncMock(),
        )
        with patch("scripts.run_question.get_client", side_effect=[finder, reader]):
            result = await run(record, retrieval_enabled=False)
        self.assertEqual(result["predicted"], "B")
        self.assertIsNone(result["correct"])
        finder.aclose.assert_awaited_once()
        reader.aclose.assert_awaited_once()

    async def test_mcq_gold_not_in_prompts(self):
        finder = SimpleNamespace(ask=AsyncMock(return_value='<reason>Clue in [0].</reason><answer>YES</answer>'))
        reader = SimpleNamespace(ask=AsyncMock(return_value='<reason>Clue in [0].</reason><answer>B</answer>'))
        result = await run_question(
            client=reader, finder_client=finder,
            paragraphs=["Leon carried the key."], question="Who carried the key?",
            options={"A": "Mara", "B": "Leon", "C": "Iris", "D": "Tomas"},
            gold="GOLD_SENTINEL", retrieval_enabled=False,
        )
        self.assertEqual(result.predicted, "B")
        self.assertNotIn("GOLD_SENTINEL", str(finder.ask.call_args_list + reader.ask.call_args_list))

    async def test_binary_uses_claim_prompt_and_same_evidence(self):
        finder = SimpleNamespace(ask=AsyncMock(return_value='<reason>Clue in [0].</reason><answer>YES</answer>'))
        reader = SimpleNamespace(ask=AsyncMock(side_effect=[
            '<reason>Clue in [0].</reason><answer>TRUE</answer>',
            '<reason>Confirmed in [0].</reason><answer>TRUE</answer>',
        ]))
        result = await run_question(
            client=reader, finder_client=finder,
            paragraphs=["Leon carried the key."], question="Leon carried the key.",
            options={"A": "TRUE", "B": "FALSE"}, gold="TRUE",
            binary_mode=True, retrieval_enabled=False,
        )
        self.assertTrue(result.correct)
        self.assertIn("Claim to verify:", finder.ask.call_args.args[0])
        self.assertNotIn("Options:", finder.ask.call_args.args[0])
        self.assertEqual(reader.ask.await_count, 2)
        self.assertTrue(result.evidence)
        for call in reader.ask.call_args_list:
            self.assertIn(result.evidence, call.args[0])

    async def test_thinking_parameter_reaches_chat_template(self):
        response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))])
        sdk = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=AsyncMock(return_value=response))),
            close=AsyncMock(),
        )
        with patch("src.llm_client.AsyncOpenAI", return_value=sdk):
            client = LLMClient(ModelSpec("test", "test", "http://localhost:8000/v1", enable_thinking=False))
            self.assertEqual(await client.ask("question"), "ok")
            self.assertEqual(sdk.chat.completions.create.call_args.kwargs["extra_body"],
                             {"chat_template_kwargs": {"enable_thinking": False}})
            await client.aclose()
            sdk.close.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
