import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from alphagen_llm.client.base import ChatClient
from scripts.replay_repairs import load_cases, run_replay, MODEL


class ReplayTests(unittest.TestCase):
    def test_paired_streams_strip_historical_answers_and_record_subgroups(self):
        clients = []

        class Client(ChatClient):
            model = MODEL

            def __init__(self):
                super().__init__()
                self.prompts, self.request_usage = [], []
                self.usage = dict(calls=0, prompt_tokens=0, completion_tokens=0)

            def chat_complete(self, content):
                self.prompts.append(content)
                self.usage['calls'] += 1
                self.usage['prompt_tokens'] += 20
                self.usage['completion_tokens'] += 5
                self.request_usage.append(dict(call=self.usage['calls'], prompt_tokens=20, completion_tokens=5))
                return 'Mean(close,20)'

        def factory(directory, calls, seed, temperature):
            self.assertEqual(calls, 4)
            self.assertEqual(temperature, 1.0)
            client = Client()
            clients.append(client)
            return client

        with TemporaryDirectory() as temporary:
            source = Path(temporary) / 'cases.json'
            source.write_text(json.dumps([dict(case_id=str(i), expression=text,
                repair={'repaired_fragment': 'DO_NOT_LEAK_HISTORICAL_ANSWER'})
                for i, text in enumerate(('Mean(close)', 'Corr(close,volume)', '- volume', 'Log(10)'))]))
            cases = load_cases(source)
            self.assertEqual(sum(c['formatting_only'] for c in cases), 1)
            self.assertNotIn('DO_NOT_LEAK', json.dumps(cases))
            root = Path(temporary) / 'replay'
            result = run_replay(factory, root, cases)
            self.assertEqual(result['usage']['calls'], 48)
            self.assertEqual(result['normalization_baseline']['parser_passes'], 1)
            self.assertEqual(len(clients), 12)
            self.assertTrue(all('DO_NOT_LEAK' not in prompt for c in clients for prompt in c.prompts))
            for seed in (0, 1, 2):
                orders = []
                for mode in ('none', 'nearest', 'random', 'shuffled'):
                    row = json.loads((root / f'{mode}_order{seed}/results.json').read_text())
                    orders.append(row['input_order'])
                    self.assertEqual(row['events'][0]['retrieved_ids'], [])
                self.assertTrue(all(order == orders[0] for order in orders))
            for condition in result['conditions']:
                expected = {'all': 12, 'nonformat': 9, 'formatting_only': 3}[condition['group']]
                self.assertEqual(condition['attempts'], expected)
            self.assertTrue(all(row['wins'] == row['losses'] == 0 for row in result['paired']))
            with self.assertRaises(FileExistsError):
                run_replay(factory, root, cases)
            self.assertEqual(len(clients), 12)

    def test_valid_or_duplicate_inputs_are_rejected_before_api_use(self):
        with TemporaryDirectory() as temporary:
            source = Path(temporary) / 'cases.json'
            for inputs in ([dict(case_id='valid', expression='close')],
                           [dict(case_id=str(i), expression='Mean(close)') for i in range(2)]):
                source.write_text(json.dumps(inputs))
                with self.assertRaises(ValueError):
                    load_cases(source)


if __name__ == '__main__':
    unittest.main()
