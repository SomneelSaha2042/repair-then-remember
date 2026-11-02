from datetime import date, timedelta
from pathlib import Path
import struct
from tempfile import TemporaryDirectory
import unittest

from scripts.check_data import inspect_data, FEATURES


class DataTests(unittest.TestCase):
    def test_missing_features_and_future_buffer(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "calendars").mkdir()
            days = [(date(2000, 1, 1) + timedelta(days=i)).isoformat() for i in range(10000)]
            (root / "calendars/day.txt").write_text("\n".join(days))
            (root / "instruments").mkdir()
            (root / "instruments/csi300.txt").write_text("SH600000\t2000-01-01\t2026-01-01\n")
            with self.assertRaisesRegex(ValueError, "Missing features"):
                inspect_data(root)
            features = root / "features/sh600000"
            features.mkdir(parents=True)
            for feature in FEATURES:
                (features / f"{feature}.day.bin").write_bytes(struct.pack("<ff", 0, 1.0))
            result = inspect_data(root)
            self.assertEqual(result["checked_features"], 6)
            (root / "calendars/day.txt").write_text("\n".join(day for day in days if day <= "2023-06-30"))
            with self.assertRaisesRegex(ValueError, "coverage/buffers"):
                inspect_data(root)


if __name__ == "__main__":
    unittest.main()
