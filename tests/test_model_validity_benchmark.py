import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from openai import OpenAIError

from scripts.model_validity_benchmark import load_cases, run_benchmark, summarize


class ModelValidityTests(unittest.TestCase):
    def test_extraction_uses_recorded_system_and_user_prompts_only(self):
        lines = ["2026-11-02 12:00:00,000-DEBUG-[system] DSL rules"]
        for i in range(41):
            number = "20" if i == 0 else "five"
            lines.extend([f"2026-11-02 12:00:00,000-DEBUG-[user] Please generate {number} alphas. Context {i}",
                          "2026-11-02 12:00:00,000-DEBUG-[assistant] close",
                          "2026-11-02 12:00:00,000-DEBUG-[script] Ensemble on test: private evaluation"])
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "source.log"
            path.write_text("\n".join(lines))
            cases = load_cases(path)
        self.assertEqual(len(cases), 9)
        self.assertEqual(sum(c["requested"] for c in cases), 60)
        self.assertTrue(all(c["system"] == "DSL rules" and "private evaluation" not in c["prompt"] for c in cases))
        self.assertEqual(len({c["sha256"] for c in cases}), 9)

    def test_models_receive_paired_prompts_seeds_and_temperatures(self):
        calls = []

        class Client:
            def __init__(self, model, seed, temp, system):
                self.model, self.seed, self.temp, self.system = model, seed, temp, system
                self.usage, self.request_usage = {}, []

            def chat_complete(self, prompt):
                calls.append((self.model, self.seed, self.temp, self.system, prompt))
                self.usage = dict(calls=1, prompt_tokens=10, completion_tokens=5)
                self.request_usage = [dict(model=self.model, finish_reason="stop", system_fingerprint="test")]
                return "Mean(close)" if self.model == "small" else "Mean(close,20)"

        def factory(directory, model, seed, temp, system):
            return Client(model, seed, temp, system)

        cases = [dict(prompt_id=0, system="same rules", prompt="same request", requested=1, sha256="same")]
        with TemporaryDirectory() as temporary:
            root = Path(temporary) / "benchmark"
            result = run_benchmark(factory, root, cases, models=("small", "reference"), seeds=(0, 1), workers=2)
            protocol = json.loads((root / "protocol.json").read_text())
            self.assertEqual(protocol["calls"], 8)
            self.assertFalse(protocol["store"])
            self.assertEqual(len((root / "generations.jsonl").read_text().splitlines()), 8)
        self.assertEqual({c[1:] for c in calls if c[0] == "small"}, {c[1:] for c in calls if c[0] == "reference"})
        self.assertTrue(all(cell["invalid"] == 2 for cell in result["cells"] if cell["model"] == "small"))
        self.assertTrue(all(cell["valid"] == 2 for cell in result["cells"] if cell["model"] == "reference"))

    def test_api_errors_and_truncation_are_not_silently_lost(self):
        rows = [dict(batch=0, model="one", temperature=0.0, seed=0, requested=1,
                     ok=False, error_type=OpenAIError.__name__, usage=dict(calls=1)),
                dict(batch=1, model="one", temperature=0.0, seed=1, requested=1,
                     ok=True, response="Mean(close", usage=dict(calls=1),
                     request_usage=[dict(finish_reason="length")])]
        result = summarize(rows, ("one",), (0.0,))
        cell = result["cells"][0]
        self.assertEqual((cell["api_errors"], cell["completed_calls"], cell["invalid"], cell["truncated_calls"]), (1, 1, 1, 1))


if __name__ == "__main__":
    unittest.main()
