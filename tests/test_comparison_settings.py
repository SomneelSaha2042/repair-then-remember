import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from scripts.compare_repairs import main as compare
from scripts.rl import build_chat_client, run_single_experiment


class ComparisonSettingsTests(unittest.TestCase):
    def test_cli_forwards_and_records_model_and_temperature_for_every_arm(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            argv = ["compare_repairs", "--data", temporary, "--seeds", "0", "--steps", "51200",
                    "--model", "gpt-4o-mini-2024-07-18", "--temperature", "1",
                    "--output", str(root / "runs"), "--dry-run"]
            with patch("sys.argv", argv), patch("scripts.compare_repairs.inspect_data", return_value={"path": temporary}), \
                 patch("scripts.compare_repairs.subprocess.check_output", side_effect=["revision\n", ""]), \
                 patch("scripts.compare_repairs.print") as output:
                compare()
            commands = [call.args[0] for call in output.call_args_list if "scripts.rl" in call.args[0]]
            self.assertEqual(len(commands), 3)
            self.assertTrue(all("--llm_temperature=1.0" in command and "--llm_model=gpt-4o-mini-2024-07-18" in command for command in commands))
            config = json.loads(next((root / "runs").glob("*/config.json")).read_text())
            self.assertEqual(config["temperature"], 1.0)

    def test_training_client_sends_the_requested_temperature(self):
        response = SimpleNamespace(model="gpt-4o-mini-2024-07-18", system_fingerprint="test",
            choices=[SimpleNamespace(message=SimpleNamespace(content="Mean(close,20)"), finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, prompt_tokens_details=None))
        request = Mock(return_value=response)
        api = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=request)))
        with patch("scripts.rl.OpenAI", return_value=api), patch("scripts.rl.load_dotenv"), patch("scripts.rl.get_logger"):
            client = build_chat_client("unused", model="gpt-4o-mini-2024-07-18", temperature=1.0)
            client.chat_complete("Generate one alpha")
        self.assertEqual(request.call_args.kwargs["model"], "gpt-4o-mini-2024-07-18")
        self.assertEqual(request.call_args.kwargs["temperature"], 1.0)
        self.assertFalse(request.call_args.kwargs["store"])
        self.assertEqual(client.request_usage[0]["temperature"], 1.0)

    def test_invalid_temperature_fails_before_data_loading(self):
        for temperature in (-0.1, 2.1, float("nan")):
            with self.subTest(temperature=temperature), patch("scripts.rl.initialize_qlib") as initialize:
                with self.assertRaises(ValueError):
                    run_single_experiment(llm_temperature=temperature)
                initialize.assert_not_called()


if __name__ == "__main__":
    unittest.main()
