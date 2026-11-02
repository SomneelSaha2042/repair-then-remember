import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from alphagen_llm.repair_memory import RepairEngine, build_parser, parse_candidate, error_class
from alphagen_llm.client.base import ChatClient, ChatConfig
from alphagen_llm.client.openai_client import OpenAIClient
from alphagen_llm.prompts.interaction import DefaultInteraction
from alphagen_llm.prompts.feedback_interaction import FeedbackInteraction


class FakeClient(ChatClient):
    def __init__(self, replies):
        super().__init__()
        self.replies = iter(replies)
        self.prompts = []

    def chat_complete(self, content):
        self.prompts.append(content)
        return next(self.replies)


class RepairMemoryTests(unittest.TestCase):
    def test_causal_retrieval_persistence_and_failed_repair(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "memory.jsonl"
            engine = RepairEngine(build_parser(), "nearest", path)
            client = FakeClient(["Corr(close,volume,20)", "Corr(open,volume,20)", "Mean(close)"])
            engine.process("Corr(close,volume)", client, repair=True)
            self.assertNotIn("Prior repair cases", client.prompts[0])
            engine.process("Corr(open,volume)", client, repair=True)
            self.assertIn("Prior repair cases", client.prompts[1])
            self.assertEqual(len(engine.records), 2)
            engine.process("Mean(close)", client, repair=True)
            self.assertEqual(len(engine.records), 2)
            loaded = RepairEngine(build_parser(), "nearest", path)
            self.assertEqual(len(loaded.records), 2)
            self.assertEqual(engine.summary()["repair_calls"], 3)
            self.assertEqual(engine.summary()["repair_successes"], 2)
            self.assertEqual(engine.events[0]["retrieved_ids"], [])
            self.assertEqual(engine.summary()["initial_valid"], 0)

    def test_reject_constant_and_invalid_memory(self):
        parser = build_parser()
        for text in ("1", "20d", "Mean()", "Mean(close,-1)", "Log(volume,5)"):
            self.assertIsNone(parse_candidate(parser, text)[0])
        self.assertIn("ExpressionParsingError", parse_candidate(parser, "Mean()")[1])
        with TemporaryDirectory() as directory:
            path = Path(directory) / "memory.jsonl"
            engine = RepairEngine(parser, "nearest", path)
            engine.process("Mean(close)", FakeClient(["1"]), repair=True)
            self.assertFalse(path.exists())
            engine.process("Mean(close)", FakeClient(["Mean(close,20)"]), repair=True)
            record = json.loads(path.read_text())
            record["repaired_fragment"] = "Mean(close)"
            path.write_text(json.dumps(record))
            with self.assertRaises(ValueError):
                RepairEngine(parser, "nearest", path)

    def test_none_never_retrieves_and_errors_are_distinct(self):
        engine = RepairEngine(build_parser())
        client = FakeClient(["Mean(close,20)", "Mean(open,20)"])
        engine.process("Mean(close)", client, repair=True)
        engine.process("Mean(open)", client, repair=True)
        self.assertEqual(engine.records, [])
        self.assertTrue(all("Prior repair cases" not in p for p in client.prompts))
        self.assertEqual(error_class("Mean expects a DeltaTime as its last operand"), "window_type")
        self.assertEqual(error_class("Mean expects 2 operand(s), but received 1"), "operator_arity")

    def test_controls_and_incompatible_parser(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "memory.jsonl"
            engine = RepairEngine(build_parser(), "nearest", path)
            engine.process("Mean(close)\nCorr(close,volume)",
                           FakeClient(["Mean(close,20)", "Corr(close,volume,20)"]), repair=True)
            message = parse_candidate(build_parser(), "Corr(open,volume)")[1]
            selected = engine.retrieve("Corr(open,volume)", message)
            self.assertEqual(selected[0]["operator"], "corr")
            for mode in ("random", "shuffled"):
                a = RepairEngine(build_parser(), mode, path, seed=5)
                b = RepairEngine(build_parser(), mode, path, seed=5)
                self.assertEqual(a.retrieve("Corr(open,volume)", message),
                                 b.retrieve("Corr(open,volume)", message))
            shuffled = RepairEngine(build_parser(), "shuffled", path, k=2)
            self.assertTrue(all(a["repaired_fragment"] != b["repaired_fragment"]
                                for a, b in zip(shuffled.retrieve("Corr(open,volume)", message),
                                                RepairEngine(build_parser(), "random", path, k=2).retrieve("Corr(open,volume)", message))))
            parser = build_parser()
            parser._allow_np_dt = True
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                RepairEngine(parser, "nearest", path)

    def test_same_generation_and_initialization_use_same_pipeline(self):
        for cls, repairs in ((DefaultInteraction, False), (FeedbackInteraction, True)):
            engine = RepairEngine(build_parser(), "nearest" if repairs else "none")
            client = FakeClient(["Mean(close)\nMean(volume,10)", "Mean(close,20)"])
            session = cls(build_parser(), client, lambda exprs: None, calculator_train=None,
                          repair_engine=engine)
            exprs = session._chat_and_parse("generate")
            self.assertEqual(len(exprs), 2 if repairs else 1)
            self.assertEqual(engine.summary()["initial_candidates"], 2)
            self.assertEqual(engine.summary()["initial_valid"], 1)
            client.replies = iter(["Mean(close)", "Mean(close,20)"])
            pool = Mock(capacity=1)
            session._evaluate_pool = Mock(return_value="report")
            session._initialize(pool, [])
            self.assertEqual(len(pool.force_load_exprs.call_args.args[0]), 1 if repairs else 0)

    def test_api_settings_usage_and_call_ceiling(self):
        requests = []
        def create(**kwargs):
            requests.append(kwargs)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="close"))],
                                   usage=SimpleNamespace(prompt_tokens=10, completion_tokens=2,
                                                         prompt_tokens_details=None))
        api = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        client = OpenAIClient(api, ChatConfig(system_prompt="DSL"), trim_to_token_limit=False, max_calls=1)
        self.assertEqual(client.chat_complete("generate"), "close")
        self.assertFalse(requests[0]["store"])
        self.assertEqual(requests[0]["model"], "gpt-4o-2024-11-20")
        self.assertEqual(requests[0]["messages"][0]["content"], "DSL")
        self.assertEqual(client.usage["prompt_tokens"], 10)
        with self.assertRaises(RuntimeError):
            client.chat_complete("generate")
        self.assertEqual(len(requests), 1)


if __name__ == "__main__":
    unittest.main()
