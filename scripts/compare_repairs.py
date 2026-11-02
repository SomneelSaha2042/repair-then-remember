"""Run paired HARLA / stateless repair / episodic repair conditions."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime

from scripts.check_data import inspect_data
from alphagen_llm.client.openai_client import DEFAULT_MODEL


CONDITIONS = {
    "harla": dict(feedback_loop=False, repair_memory="none"),
    "repair": dict(feedback_loop=True, repair_memory="none"),
    "memory": dict(feedback_loop=True, repair_memory="nearest"),
    "random": dict(feedback_loop=True, repair_memory="random"),
    "shuffled": dict(feedback_loop=True, repair_memory="shuffled"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--profile", choices=("paper", "postcutoff", "postrelease"), default="postrelease")
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=["harla", "repair", "memory"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--steps", type=int, default=250000)
    parser.add_argument("--llm-updates", type=int, default=20)
    parser.add_argument("--llm-max-calls", type=int, default=1000)
    parser.add_argument("--memory-k", type=int, default=1)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", default="out/comparisons")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not Path(args.data).expanduser().is_dir():
        parser.error("--data must be an extracted Qlib directory")
    manifest = inspect_data(args.data, args.profile)
    root = Path(args.output) / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    root.mkdir(parents=True)
    (root / "data_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    config = vars(args)
    config["python"] = sys.version
    config["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    config["git_dirty"] = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    (root / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    for seed in args.seeds:
        for condition in args.conditions:
            directory = root / f"{condition}_seed{seed}"
            command = [sys.executable, "-m", "scripts.rl", f"--random_seeds={seed}",
                       "--use_llm=True", "--feedback_mode=fix-only", "--pool_capacity=20",
                       "--drop_rl_n=5", "--llm_replace_n=5", f"--steps={args.steps}", "--llm_every_n_steps=25000",
                       f"--qlib_data_path={manifest['path']}", f"--data_profile={args.profile}",
                       f"--device={args.device}", f"--llm_model={args.model}",
                       f"--llm_max_calls={args.llm_max_calls}", f"--llm_updates={args.llm_updates}",
                       f"--memory_k={args.memory_k}", f"--output_dir={directory}"]
            command += [f"--{key}={value}" for key, value in CONDITIONS[condition].items()]
            print(subprocess.list2cmdline(command), flush=True)
            if args.dry_run:
                continue
            directory.mkdir(parents=True)
            with (directory / "stdout.log").open("w", encoding="utf-8") as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    if not args.dry_run:
        results = [json.loads(p.read_text()) for p in root.glob("*/*/final.json")]
        (root / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Comparison artifacts: {root}")


if __name__ == "__main__":
    main()
