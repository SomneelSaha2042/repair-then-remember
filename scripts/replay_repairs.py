"""Compare repair memory on a fixed natural-failure corpus, without new generation or PPO."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

from alphagen_llm.client import ChatConfig, OpenAIClient
from alphagen_llm.prompts.system_prompt import EXPLAIN_WITH_TEXT_DESC
from alphagen_llm.repair_memory import RepairEngine, build_parser, error_class, parse_candidate
from alphagen.utils import get_logger
from scripts.natural_repair_benchmark import MODES, replay, write_json


MODEL = "gpt-4o-mini-2024-07-18"
TEMPERATURE = 1.0
ORDERS = (0, 1, 2)


def load_cases(path):
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise ValueError("Expected a nonempty JSON list of natural failure cases")
    parser, cases, seen = build_parser(), [], set()
    for record in records:
        text = record["expression"].strip()
        expression, message = parse_candidate(parser, text)
        if not text or "\n" in text or expression is not None or text in seen:
            raise ValueError("Each case must be one unique parser-invalid expression")
        seen.add(text)
        normalized = re.sub(r"^-\s+(?=[A-Za-z_])", "", text).removesuffix(":").strip()
        formatting = normalized != text and parse_candidate(parser, normalized)[0] is not None
        # Historical repairs and all other source fields are deliberately excluded.
        cases.append(dict(case_id=record["case_id"], response=text,
                          parser_message=message, error_class=error_class(message),
                          formatting_only=formatting, normalized_expression=normalized))
    if len({row["case_id"] for row in cases}) != len(cases):
        raise ValueError("Case IDs must be unique")
    return cases


def summarize(results, cases):
    indexed = {row["response"]: row for row in cases}
    conditions, paired = [], []
    for mode in MODES:
        runs = [r for r in results if r["mode"] == mode]
        for group in ("all", "nonformat", "formatting_only"):
            selected = {text for text, case in indexed.items()
                        if group == "all" or case["formatting_only"] == (group == "formatting_only")}
            events = [e for run in runs for e in run["events"] if e["invalid_fragment"] in selected]
            successes = sum(e["success"] for e in events)
            prompt_tokens = sum(e["request_usage"].get("prompt_tokens", 0) or 0 for e in events)
            completion_tokens = sum(e["request_usage"].get("completion_tokens", 0) or 0 for e in events)
            conditions.append(dict(mode=mode, group=group, attempts=len(events), successes=successes,
                success_rate=successes/len(events) if events else None,
                prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                tokens_per_success=(prompt_tokens+completion_tokens)/successes if successes else None,
                retrieved_examples=sum(len(e["retrieved_ids"]) for e in events),
                calls_with_retrieval=sum(bool(e["retrieved_ids"]) for e in events)))
    for seed in sorted({r["order_seed"] for r in results}):
        baseline = next(r for r in results if r["mode"] == "none" and r["order_seed"] == seed)
        base = {e["invalid_fragment"]: e["success"] for e in baseline["events"]}
        for run in (r for r in results if r["mode"] != "none" and r["order_seed"] == seed):
            assert run["input_order"] == baseline["input_order"]
            for group in ("all", "nonformat", "formatting_only"):
                events = [e for e in run["events"] if group == "all" or
                          indexed[e["invalid_fragment"]]["formatting_only"] == (group == "formatting_only")]
                paired.append(dict(mode=run["mode"], order_seed=seed, group=group,
                    wins=sum(e["success"] and not base[e["invalid_fragment"]] for e in events),
                    losses=sum(base[e["invalid_fragment"]] and not e["success"] for e in events)))
    usage = {key: sum(r["usage"].get(key, 0) for r in results)
             for key in ("calls", "prompt_tokens", "completion_tokens", "cached_tokens")}
    return dict(conditions=conditions, paired=paired, usage=usage)


def run_replay(factory, root, cases, provenance=None):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "cases.json", cases)
    write_json(root / "protocol.json", dict(model=MODEL, temperature=TEMPERATURE,
        modes=MODES, order_seeds=ORDERS, api_seeds=ORDERS, sampling_seed_coupled_to_order=True,
        case_count=len(cases), formatting_only=sum(c["formatting_only"] for c in cases),
        nonformat=sum(not c["formatting_only"] for c in cases),
        total_call_ceiling=len(cases)*len(MODES)*len(ORDERS), repair_calls_per_case=1,
        max_output_tokens=2048, retries=0, store=False, retrieval_k=1,
        dsl_id=RepairEngine(build_parser()).dsl_id,
        memory="Empty for every condition/order; finalized successful earlier repairs only",
        financial_data_sent=False, historical_repaired_answers_sent=False,
        formatting_normalization="Zero-call diagnostic baseline only; all four arms receive unchanged raw inputs",
        controls="Equal cases, order and repair-call ceiling; prompt tokens and retrieval availability differ",
        shuffled="Wrong antecedent/consequent pairing in the prompt; stored repairs remain verified. One-entry memory cannot be shuffled.",
        provenance=provenance))
    results = []
    with ThreadPoolExecutor(max_workers=3) as executor:
        pending = [executor.submit(replay, factory, root, mode, seed, cases, temperature=TEMPERATURE)
                   for seed in ORDERS for mode in MODES]
        for future in as_completed(pending):
            result = future.result()
            results.append(result)
            assert len(result["events"]) == len(cases)
            assert result["usage"]["calls"] == len(cases)
            print(f"Completed {result['mode']} order {result['order_seed']}: "
                  f"{result['metrics'].get('repair_successes', 0)}/{len(cases)}", flush=True)
    final = summarize(results, cases)
    assert final["usage"]["calls"] == len(cases)*len(MODES)*len(ORDERS)
    final["normalization_baseline"] = dict(cases=len(cases), calls=0,
        parser_passes=sum(c["formatting_only"] for c in cases),
        scope="Structural parsing only; no financial or semantic-preservation claim")
    final["per_order"] = [{key: value for key, value in r.items() if key not in ("events", "request_usage", "input_order")}
                          for r in sorted(results, key=lambda r: (r["order_seed"], r["mode"]))]
    final["limitations"] = [
        "Exploratory fixed corpus from one PPO seed; three repeated orders are not independent datasets.",
        "No equal-token claim; memory changes prompt content, length and availability.",
        "Formatting labels define reporting subgroups, not a gold semantic-correctness oracle.",
        "Successful parsing does not establish semantic preservation or downstream pool performance.",
        "Per-case processing increments the repair engine's generation_calls metric, but no generation API calls occur."]
    write_json(root / "results.json", final)
    return final


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--output", default="out/repair-replay/" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    args = parser.parse_args()
    source = Path(args.cases).resolve()
    cases = load_cases(source)
    repository = Path(__file__).resolve().parents[1]
    git = ["git", "-c", f"safe.directory={repository.as_posix()}"]
    provenance = dict(source_file=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        git_commit=subprocess.check_output(git + ["rev-parse", "HEAD"], text=True).strip(),
        git_dirty=bool(subprocess.check_output(git + ["status", "--porcelain"], text=True).strip()))
    load_dotenv(repository / ".env", override=False)

    def factory(directory, calls, seed, temperature):
        return OpenAIClient(OpenAI(max_retries=0, timeout=120), ChatConfig(
            system_prompt=EXPLAIN_WITH_TEXT_DESC,
            logger=get_logger(str(directory.resolve()), str(directory / "llm.log"))),
            model=MODEL, seed=seed, temperature=temperature, max_calls=calls, max_tokens=2048)

    try:
        result = run_replay(factory, args.output, cases, provenance=provenance)
    except OpenAIError as error:
        raise SystemExit(f"Replay stopped: {type(error).__name__}; status={getattr(error, 'status_code', None)}. Completed artifacts were preserved.") from None
    print(json.dumps(result["conditions"], indent=2))
    print(f"Replay artifacts: {args.output}")


if __name__ == "__main__":
    main()
