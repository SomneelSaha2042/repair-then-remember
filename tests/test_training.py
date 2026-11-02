"""Small real PPO smoke test with synthetic prices and a scripted LLM (no API)."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from sb3_contrib import MaskablePPO

from alphagen_llm.client.base import ChatClient
from scripts.rl import run_single_experiment


class ScriptedClient(ChatClient):
    def __init__(self):
        super().__init__()
        self.usage = {"calls": 0}
        self.request_usage = []

    def chat_complete(self, content):
        self.usage["calls"] += 1
        self.request_usage.append({"call": self.usage["calls"]})
        if "Fix this alpha expression" in content:
            return "Mean(close,5)"
        return "Mean(close)\nclose\nvolume\nSub(close,open)\nMean(high,5)"


class TrainingTests(unittest.TestCase):
    def test_three_arms_train_and_write_results_without_api(self):
        torch.set_num_threads(1)
        def short_ppo(*args, **kwargs):
            return MaskablePPO(*args, **kwargs, n_steps=32, n_epochs=1)
        def synthetic_data(**kwargs):
            data = torch.rand(162, 6, 8) + 1.0
            return SimpleNamespace(data=data, n_days=32, n_stocks=8,
                                   max_backtrack_days=100, max_future_days=30)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "calendars").mkdir()
            (root / "calendars/day.txt").write_text("\n".join(f"2021-12-{i:02d}" for i in range(1, 32)))
            for feedback, memory in ((False, "none"), (True, "none"), (True, "nearest")):
                client = ScriptedClient()
                with patch("scripts.rl.initialize_qlib"), patch("scripts.rl.StockData", side_effect=synthetic_data), \
                     patch("scripts.rl.build_chat_client", return_value=client), \
                     patch("scripts.rl.MaskablePPO", side_effect=short_ppo):
                    result = run_single_experiment(
                        pool_capacity=3, steps=64, use_llm=True, feedback_loop=feedback,
                        feedback_mode="fix-only", repair_memory=memory, llm_updates=1,
                        llm_every_n_steps=32, drop_rl_n=1, llm_replace_n=1,
                        qlib_data_path=str(root), device="cpu", output_dir=str(root / "runs"))
                self.assertEqual(result["final_metrics"]["timestep"], 64)
                saved = Path(result["save_path"]) / "final.json"
                self.assertEqual(json.loads(saved.read_text())["structural_metrics"]["mode"], memory)
                self.assertGreater(result["structural_metrics"]["generation_calls"], 1)
                self.assertEqual(result["llm_refreshes"], 2)
                events = [json.loads(row) for row in (saved.parent / "candidates.jsonl").read_text().splitlines()]
                self.assertTrue(any(row['phase'] == 'initial_pool' for row in events))
                self.assertTrue(any(row['phase'] == 'pool_refresh' and row['ppo_timestep'] == 64 for row in events))
                self.assertIn('operators', result['structural_metrics']['expression_shapes'])
                if feedback:
                    self.assertGreater(result["structural_metrics"]["repair_successes"], 0)


if __name__ == "__main__":
    unittest.main()
