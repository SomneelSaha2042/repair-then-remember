"""Inspect a Qlib directory or archive without importing Qlib or using a GPU."""
import argparse
from bisect import bisect_left, bisect_right
import hashlib
import json
import math
from pathlib import Path
import struct
import tarfile


PROFILES = {
    "paper": [("2012-01-01", "2021-12-31"), ("2022-01-01", "2022-06-30"),
              ("2022-07-01", "2022-12-31"), ("2023-01-01", "2023-06-30")],
    "postcutoff": [("2012-01-01", "2023-12-31"), ("2024-01-01", "2024-12-31"),
                   ("2025-01-01", "2025-06-30"), ("2025-07-01", "2025-12-31")],
    "postrelease": [("2012-01-01", "2024-12-31"), ("2025-01-01", "2025-06-30"),
                    ("2025-07-01", "2025-12-31"), ("2026-01-01", "2026-06-30")],
}
FEATURES = ("open", "close", "high", "low", "volume", "vwap")


def inspect_data(path, profile="paper", instruments="csi300", backtrack=100, future=30):
    path = Path(path).expanduser().resolve()
    archive = tarfile.open(path) if path.is_file() else None
    try:
        if archive:
            members = {m.name.rstrip("/"): m for m in archive.getmembers()}
            roots = [n[:-len("calendars/day.txt")] for n in members if n.endswith("calendars/day.txt")]
            if len(roots) != 1:
                raise ValueError("Archive must contain exactly one Qlib calendar")
            root = roots[0]
            def read(name):
                return archive.extractfile(members[root + name]).read()
            def exists(name):
                return root + name in members
            with path.open("rb") as handle:
                metadata = dict(archive_sha256=hashlib.file_digest(handle, "sha256").hexdigest())
        else:
            def read(name):
                return (path / name).read_bytes()
            def exists(name):
                return (path / name).is_file()
            metadata = {}
        calendar_raw = read("calendars/day.txt")
        universe_raw = read(f"instruments/{instruments}.txt")
        calendar = calendar_raw.decode().splitlines()
        if calendar != sorted(set(calendar)) or not calendar:
            raise ValueError("Calendar must be nonempty, sorted and unique")
        intervals = [line.split() for line in universe_raw.decode().splitlines() if line.strip()]
        if any(len(row) != 3 for row in intervals):
            raise ValueError("Instrument records must contain symbol/start/end")
        spans = []
        selected = set()
        for start, end in PROFILES[profile]:
            lo, hi = bisect_left(calendar, start), bisect_right(calendar, end) - 1
            if lo < backtrack or hi < lo or hi + future >= len(calendar):
                raise ValueError(f"Insufficient calendar coverage/buffers for {start}..{end}")
            symbols = {symbol.lower() for symbol, first, last in intervals if first <= end and last >= start}
            if not symbols:
                raise ValueError(f"No {instruments} constituents in {start}..{end}")
            selected.update(symbols)
            spans.append(dict(start=start, end=end, trading_days=hi-lo+1, symbols=len(symbols)))
        missing, nonfinite = [], []
        feature_digest = hashlib.sha256()
        # Inspect every required feature, not just the archive calendar.
        for symbol in sorted(selected):
            for feature in sorted(FEATURES):
                name = f"features/{symbol}/{feature}.day.bin"
                if not exists(name):
                    missing.append(name)
                    continue
                content = read(name)
                feature_digest.update(name.encode())
                feature_digest.update(content)
                if len(content) < 8 or len(content) % 4:
                    raise ValueError(f"Malformed Qlib float32 file: {name}")
                offset = struct.unpack_from("<f", content)[0]
                if not math.isfinite(offset) or not offset.is_integer() or offset < 0:
                    raise ValueError(f"Invalid calendar offset: {name}")
                if offset + len(content) // 4 - 1 > len(calendar):
                    raise ValueError(f"Feature extends beyond the calendar: {name}")
                # An all-NaN required feature is unusable even with a valid filename.
                if not any(math.isfinite(v[0]) for v in struct.iter_unpack("<f", content[4:])):
                    nonfinite.append(name)
        if missing or nonfinite:
            raise ValueError(f"Missing features ({len(missing)}): {missing[:8]}; all-nonfinite ({len(nonfinite)}): {nonfinite[:8]}")
        return dict(path=str(path), profile=profile, instruments=instruments,
                    calendar_start=calendar[0], calendar_end=calendar[-1], spans=spans,
                    required_symbols=len(selected), checked_features=len(selected)*len(FEATURES),
                    calendar_sha256=hashlib.sha256(calendar_raw).hexdigest(),
                    universe_sha256=hashlib.sha256(universe_raw).hexdigest(),
                    features_sha256=feature_digest.hexdigest(),
                    provenance="Source and adjustment policy not established by this archive", **metadata)
    finally:
        if archive:
            archive.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path")
    parser.add_argument("--profile", choices=PROFILES, default="paper")
    parser.add_argument("--instruments", default="csi300")
    parser.add_argument("--output")
    args = parser.parse_args()
    result = inspect_data(args.path, args.profile, args.instruments)
    text = json.dumps(result, indent=2)
    print(text)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
