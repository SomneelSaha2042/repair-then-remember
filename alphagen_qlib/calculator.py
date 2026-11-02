from typing import Optional
from collections import Counter
import json
from pathlib import Path
from torch import Tensor
from alphagen.data.calculator import TensorAlphaCalculator
from alphagen.data.expression import Expression
from alphagen.utils.pytorch_utils import normalize_by_day
from alphagen_qlib.stock_data import StockData


class QLibStockDataCalculator(TensorAlphaCalculator):
    def __init__(self, data: StockData, target: Optional[Expression] = None, diagnostics_path=None):
        super().__init__(normalize_by_day(target.evaluate(data)) if target is not None else None)
        self.data = data
        self.diagnostics_path = Path(diagnostics_path) if diagnostics_path else None
        self.runtime_errors = Counter()

    def evaluate_alpha(self, expr: Expression) -> Tensor:
        try:
            return normalize_by_day(expr.evaluate(self.data))
        except Exception as error:
            self.runtime_errors[type(error).__name__] += 1
            if self.diagnostics_path:
                self.diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
                with self.diagnostics_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(dict(expression=str(expr), error_class=type(error).__name__,
                                                 message=str(error))) + "\n")
            raise
    
    @property
    def n_days(self) -> int:
        return self.data.n_days
