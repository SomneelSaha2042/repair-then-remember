from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from alphagen_llm.client.base import ChatClient
from alphagen_llm.client.openai_client import DEFAULT_MODEL
from scripts.natural_repair_benchmark import inspect_generations, run_benchmark


class NaturalBenchmarkTests(unittest.TestCase):
    def test_deduplicates_without_manufacturing_failures(self):
        summary, cases = inspect_generations([
            dict(batch=0, temperature=0.0, requested=3,
                 response="close\nMean(close)\nMean(close)"),
            dict(batch=1, temperature=0.7, requested=1, response="volume")])
        self.assertEqual(summary['by_temperature']['0.0']['invalid'], 2)
        self.assertEqual(summary['unique_invalid'], 1)
        self.assertEqual([row['response'] for row in cases], ['Mean(close)'])

    def test_all_conditions_receive_same_cases_and_empty_memory(self):
        class Client(ChatClient):
            model = DEFAULT_MODEL

            def __init__(self):
                super().__init__()
                self.usage, self.request_usage = dict(calls=0), []

            def chat_complete(self, content):
                self.usage['calls'] += 1
                self.request_usage.append(dict(call=self.usage['calls']))
                return "Mean(close,20)" if "Fix this alpha" in content else "close\nMean(close)\nMean(open)"

        def factory(directory, calls, seed, temperature):
            directory.mkdir(parents=True, exist_ok=True)
            return Client()

        with TemporaryDirectory() as directory:
            root = Path(directory) / 'benchmark'
            result = run_benchmark(factory, root, count=3, repeats=1,
                                   order_seeds=(0,), workers=2)
            self.assertEqual(result['cases_replayed'], 2)
            self.assertEqual(len(result['repairs']), 4)
            for mode in ('none', 'nearest', 'random', 'shuffled'):
                import json
                row = json.loads((root / f'{mode}_order0/results.json').read_text())
                self.assertEqual(set(row['input_order']), {'Mean(close)', 'Mean(open)'})
                self.assertEqual(row['events'][0]['retrieved_ids'], [])
            self.assertTrue(all(row['wins'] == 0 and row['losses'] == 0 for row in result['paired']))


if __name__ == '__main__':
    unittest.main()
