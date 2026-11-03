"""Compare frozen models on identical recorded HARLA generation prompts, without PPO."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import hashlib
from itertools import product
import json
from pathlib import Path
import random
import re
from statistics import fmean
import subprocess

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

from alphagen_llm.client import ChatConfig, OpenAIClient
from alphagen_llm.repair_memory import expression_shape
from alphagen.utils import get_logger
from scripts.natural_repair_benchmark import inspect_generations, write_json


MODELS = ("gpt-4o-mini-2024-07-18", "gpt-4o-2024-11-20")
PROMPT_INDICES = (0, 1, 7, 14, 20, 21, 28, 35, 40)


def load_cases(path):
    text = Path(path).read_text(encoding="utf-8")
    headers = list(re.finditer(
        r"(?m)^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+-DEBUG-\[(\w+)\] ", text))
    systems, users = [], []
    for i, header in enumerate(headers):
        end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        content = text[header.end():end].rstrip()
        if header.group(1) == "system":
            systems.append(content)
        elif header.group(1) == "user":
            users.append(content)
    if len(set(systems)) != 1 or len(users) != 41:
        raise ValueError("Expected one system prompt and 41 generation requests from the HARLA pilot")
    words = dict(zip(("one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"), range(1, 11)))
    cases = []
    for index in PROMPT_INDICES:
        prompt = users[index]
        match = re.search(r"please generate (\w+)(?: more)? alphas?", prompt, re.I)
        if match is None or "Ensemble on test" in prompt:
            raise ValueError(f"Unexpected generation prompt at source index {index}")
        number = match.group(1).lower()
        requested = int(number) if number.isdigit() else words[number]
        case = dict(prompt_id=index, system=systems[0], prompt=prompt, requested=requested,
                    phase="initial_pool" if index <= 20 else "pool_refresh")
        case["sha256"] = hashlib.sha256(json.dumps(case, sort_keys=True).encode()).hexdigest()
        cases.append(case)
    return cases


def collect(factory, root, index, model, temperature, seed, case):
    client = factory(root / f"request_{index:03d}", model, seed, temperature, case["system"])
    row = dict(batch=index, model=model, temperature=temperature, seed=seed,
               prompt_id=case["prompt_id"], prompt_sha256=case["sha256"], requested=case["requested"])
    try:
        row["response"] = client.chat_complete(case["prompt"])
        row["ok"] = True
    except (OpenAIError, ValueError) as error:
        row.update(ok=False, error_type=type(error).__name__, status_code=getattr(error, "status_code", None))
    row["usage"] = client.usage
    row["request_usage"] = client.request_usage
    return row


def summarize(rows, models, temperatures):
    cells, failures = [], []
    for model, temperature in product(models, temperatures):
        scheduled = [r for r in rows if r["model"] == model and r["temperature"] == temperature]
        completed = [r for r in scheduled if r["ok"]]
        inspection, invalid = inspect_generations(completed)
        counts = inspection["by_temperature"].get(str(temperature), {})
        shapes = [expression_shape(line.strip()) for row in completed
                  for line in row["response"].splitlines() if line.strip()]
        complexity = {key: dict(mean=fmean(row[key] for row in shapes), maximum=max(row[key] for row in shapes))
                      for key in shapes[0]} if shapes else {}
        cells.append(dict(model=model, temperature=temperature, scheduled_calls=len(scheduled),
                          completed_calls=len(completed), api_errors=len(scheduled)-len(completed),
                          truncated_calls=sum(r["request_usage"][-1].get("finish_reason") == "length"
                                              for r in completed),
                          returned=counts.get("returned", 0), valid=counts.get("valid", 0),
                          invalid=counts.get("invalid", 0), unique_valid=inspection["unique_valid"],
                          unique_invalid=inspection["unique_invalid"], error_classes=inspection["error_classes"],
                          complexity=complexity,
                          usage={key: sum(r["usage"].get(key, 0) for r in scheduled)
                                 for key in ("calls", "prompt_tokens", "completion_tokens", "cached_tokens")}))
        failures.extend(dict(row, model=model) for row in invalid)
    return dict(cells=cells, unique_failure_cases_by_model_and_temperature=failures,
                api_errors=sum(not row["ok"] for row in rows),
                limitations=["Fixed training-pool contexts from one HARLA trajectory; no new financial outcomes.",
                             "Sampling seeds are repeated API draws, not independent market datasets or PPO seeds.",
                             "Parser acceptance is separate from numerical executability and financial quality.",
                             "Output truncation is reported separately; no invalid expressions are manufactured."])


def run_benchmark(factory, root, cases, models=MODELS, temperatures=(0.0, 1.0), seeds=(0, 1, 2), workers=3,
                  provenance=None):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    if provenance is not None:
        write_json(root / "provenance.json", provenance)
    jobs = list(product(models, temperatures, seeds, cases))
    random.Random(0).shuffle(jobs)
    write_json(root / "prompts.json", cases)
    write_json(root / "protocol.json", dict(
        models=models, temperatures=temperatures, sampling_seeds=seeds, calls=len(jobs),
        prompt_indices=[c["prompt_id"] for c in cases], output_limit=2048, store=False,
        request_order_seed=0, retries=0, memory=False, repair=False,
        requested_formulas=sum(c["requested"] for c in cases)*len(models)*len(temperatures)*len(seeds),
        training_pool_statistics_sent=True, raw_market_data_sent=False, held_out_metrics_sent=False,
        complexity_scope="All returned nonempty candidate lines"))
    rows = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        pending = [executor.submit(collect, factory, root, i, model, temp, seed, case)
                   for i, (model, temp, seed, case) in enumerate(jobs)]
        for future in as_completed(pending):
            row = future.result()
            rows.append(row)
            with (root / "generations.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
            if len(rows) % 6 == 0 or len(rows) == len(jobs):
                print(f"Completed {len(rows)}/{len(jobs)} generation calls", flush=True)
    result = summarize(rows, models, temperatures)
    write_json(root / "results.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-log", required=True)
    parser.add_argument("--output", default="out/model-validity/" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    args = parser.parse_args()
    cases = load_cases(args.source_log)
    load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
    api = OpenAI(max_retries=0, timeout=120)

    def factory(directory, model, seed, temperature, system):
        return OpenAIClient(api, ChatConfig(system_prompt=system,
            logger=get_logger(str(directory.resolve()), str(directory / "llm.log"))),
            model=model, seed=seed, temperature=temperature, max_calls=1, max_tokens=2048,
            trim_to_token_limit=False)

    repository = Path(__file__).resolve().parents[1]
    git = ["git", "-c", f"safe.directory={repository.as_posix()}", "-C", str(repository)]
    provenance = dict(source_log=str(Path(args.source_log).resolve()),
                      source_log_sha256=hashlib.sha256(Path(args.source_log).read_bytes()).hexdigest(),
                      git_commit=subprocess.check_output(git + ["rev-parse", "HEAD"], text=True).strip(),
                      git_status=subprocess.check_output(git + ["status", "--porcelain"], text=True).splitlines(),
                      script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      client_sha256=hashlib.sha256((Path(__file__).resolve().parents[1] / "alphagen_llm/client/openai_client.py").read_bytes()).hexdigest())
    result = run_benchmark(factory, args.output, cases, provenance=provenance)
    print(json.dumps(result, indent=2))
    print(f"Artifacts: {args.output}")
    if result["api_errors"]:
        raise SystemExit("API errors occurred; consult saved records before comparing cells")


if __name__ == "__main__":
    main()
