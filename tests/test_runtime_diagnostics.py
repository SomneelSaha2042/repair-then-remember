import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from alphagen.data.expression import OutOfDataRangeError
from alphagen_llm.repair_memory import build_parser, expression_shape
from alphagen_qlib.calculator import QLibStockDataCalculator


class DiagnosticsTests(unittest.TestCase):
    def test_shape_includes_nested_operators_and_invalid_fragments(self):
        shape = expression_shape('Mean(Sub(close,open),20d)')
        self.assertEqual(shape['operators'], 2)
        self.assertEqual(shape['nesting_depth'], 2)
        self.assertEqual(expression_shape('Mean(Sub(close,open')['nesting_depth'], 2)

    def test_runtime_failure_is_logged_and_still_raised(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'runtime.jsonl'
            data = SimpleNamespace(max_backtrack_days=100, max_future_days=30)
            calc = QLibStockDataCalculator(data, diagnostics_path=path)
            expr = build_parser().parse('Ref(close,200d)')
            with self.assertRaises(OutOfDataRangeError):
                calc.evaluate_alpha(expr)
            self.assertEqual(calc.runtime_errors['OutOfDataRangeError'], 1)
            self.assertEqual(json.loads(path.read_text())['error_class'], 'OutOfDataRangeError')


if __name__ == '__main__':
    unittest.main()
