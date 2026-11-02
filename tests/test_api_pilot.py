from tempfile import TemporaryDirectory
from pathlib import Path
import unittest

from scripts.api_pilot import run_pilot
from scripts.check_data import PROFILES
from alphagen_llm.client.base import ChatClient


class PilotTests(unittest.TestCase):
    def test_pilot_wires_both_arms_and_has_no_market_data_input(self):
        clients = []
        class Client(ChatClient):
            def __init__(self):
                super().__init__()
                self.usage = dict(calls=0)
                self.request_usage = []
                self.prompts = []

            def chat_complete(self, content):
                self.prompts.append(content)
                self.usage['calls'] += 1
                self.request_usage.append(dict(call=self.usage['calls']))
                if 'Fix this alpha' in content:
                    return 'Mean(close,20)'
                return 'close\nvolume'

        def factory(directory, max_calls):
            client = Client()
            clients.append(client)
            return client

        with TemporaryDirectory() as directory:
            output = Path(directory) / 'pilot'
            result = run_pilot(factory, output)
            self.assertEqual(result['model'], 'gpt-4o-2024-11-20')
            self.assertEqual(result['generation']['metrics']['initial_valid'], 2)
            self.assertEqual(result['repairs']['none']['metrics']['repair_calls'], 8)
            self.assertEqual(result['repairs']['nearest']['metrics']['repair_calls'], 8)
            self.assertGreater(result['repairs']['nearest']['metrics']['retrieved_cases'], 0)
            self.assertTrue((output / 'results.json').exists())
            self.assertNotIn('Prior repair cases', clients[1].prompts[-1])
            self.assertIn('Prior repair cases', clients[2].prompts[-1])

    def test_forward_evaluation_is_after_pinned_snapshot_release(self):
        for start, end in PROFILES['postrelease'][1:]:
            self.assertGreater(start, '2024-11-20')
            self.assertGreaterEqual(end, start)


if __name__ == '__main__':
    unittest.main()
