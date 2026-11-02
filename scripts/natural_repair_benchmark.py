"""Collect unmodified GPT generations and compare causal repair on identical failures."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import json
import random
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

from alphagen_llm.client import ChatConfig, OpenAIClient
from alphagen_llm.client.openai_client import DEFAULT_MODEL
from alphagen_llm.prompts.system_prompt import EXPLAIN_WITH_TEXT_DESC
from alphagen_llm.repair_memory import RepairEngine, build_parser, error_class, parse_candidate
from alphagen.utils import get_logger


THEMES = ["short-term price reversal", "price trends", "price-volume interaction",
          "volatility", "intraday high-low ranges", "deviation from VWAP"]
MODES = ("none", "nearest", "random", "shuffled")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def collect_batch(factory, root, index, theme, temperature, count):
    client = factory(root / f"batch_{index}", 1, index, temperature)
    prompt = (f"Please generate {count} different formulaic alphas that you think would be "
              f"indicative of future stock price trend. Focus on {theme}. "
              "Each alpha should be on its own line without numbering. "
              "Please do not output anything else.")
    response = client.chat_complete(prompt)
    return dict(batch=index, theme=theme, temperature=temperature, seed=index,
                requested=count, response=response, usage=client.usage,
                request_usage=client.request_usage)


def inspect_generations(batches):
    parser, groups, errors = build_parser(), {}, Counter()
    failures, seen_failures, unique_valid = [], set(), set()
    for batch in sorted(batches, key=lambda row: row["batch"]):
        key = str(batch["temperature"])
        counts = groups.setdefault(key, Counter())
        counts["calls"] += 1
        counts["requested"] += batch["requested"]
        for line_number, line in enumerate(batch["response"].splitlines()):
            text = line.strip()
            if not text:
                continue
            counts["returned"] += 1
            expr, message = parse_candidate(parser, text)
            if expr is not None:
                counts["valid"] += 1
                unique_valid.add(str(expr))
            else:
                counts["invalid"] += 1
                errors[error_class(message)] += 1
                if text not in seen_failures:
                    seen_failures.add(text)
                    failures.append(dict(response=text, parser_message=message,
                                         error_class=error_class(message), batch=batch["batch"],
                                         line=line_number, temperature=batch["temperature"]))
    return dict(by_temperature={key: dict(value) for key, value in groups.items()},
                error_classes=dict(errors), unique_valid=len(unique_valid),
                unique_invalid=len(failures)), failures


def replay(factory, root, mode, seed, cases):
    directory = root / f"{mode}_order{seed}"
    client = factory(directory, max(1, len(cases)), seed, 0.0)
    engine = RepairEngine(build_parser(), mode, directory / "repair_memory.jsonl",
                          seed=seed, source_model=client.model)
    ordered = list(cases)
    random.Random(seed).shuffle(ordered)
    for row in ordered:
        engine.process(row["response"], client, repair=True)
    result = dict(mode=mode, order_seed=seed, input_order=[row["response"] for row in ordered],
                  metrics=engine.summary(), usage=client.usage,
                  request_usage=client.request_usage, events=engine.events)
    write_json(directory / "results.json", result)
    return result


def run_benchmark(factory, root, count=25, repeats=2, order_seeds=(0, 1, 2), workers=3, max_cases=100):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    jobs = [(theme, temperature) for temperature in (0.0, 0.7)
            for _ in range(repeats) for theme in THEMES]
    write_json(root / "protocol.json", dict(
        model=DEFAULT_MODEL, themes=THEMES, temperatures=[0.0, 0.7], repeats=repeats,
        requested_formulas=len(jobs)*count, formulas_per_call=count,
        modes=MODES, order_seeds=order_seeds, max_cases=max_cases,
        selection="First unique parser-invalid strings in batch order; never mutate outputs",
        financial_data_sent=False, memory="Empty per condition/order; prior finalized repairs only"))
    batches = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        pending = [executor.submit(collect_batch, factory, root, i, theme, temp, count)
                   for i, (theme, temp) in enumerate(jobs)]
        for future in as_completed(pending):
            batch = future.result()
            batches.append(batch)
            with (root / "generations.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(batch) + "\n")
            print(f"Generation batch {len(batches)}/{len(jobs)} complete", flush=True)
    summary, failures = inspect_generations(batches)
    write_json(root / "generation_summary.json", summary)
    with (root / "failures.jsonl").open("w", encoding="utf-8") as handle:
        for row in failures:
            handle.write(json.dumps(row) + "\n")
    print(json.dumps(summary), flush=True)
    cases = failures[:max_cases]
    results = []
    if cases:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            pending = [executor.submit(replay, factory, root, mode, seed, cases)
                       for seed in order_seeds for mode in MODES]
            for future in as_completed(pending):
                result = future.result()
                results.append(result)
                print(f"Repair {result['mode']} order {result['order_seed']}: "
                      f"{result['metrics'].get('repair_successes', 0)}/{len(cases)}", flush=True)
    usage = [batch["usage"] for batch in batches] + [r["usage"] for r in results]
    totals = {key: sum(row.get(key, 0) for row in usage)
              for key in ("calls", "prompt_tokens", "completion_tokens", "cached_tokens")}
    paired = []
    for seed in order_seeds:
        baseline = next((r for r in results if r["order_seed"] == seed and r["mode"] == "none"), None)
        if baseline is None:
            continue
        base = {e["invalid_fragment"]: e["success"] for e in baseline["events"]}
        for result in (r for r in results if r["order_seed"] == seed and r["mode"] != "none"):
            outcomes = {e["invalid_fragment"]: e["success"] for e in result["events"]}
            paired.append(dict(mode=result["mode"], order_seed=seed,
                               wins=sum(outcomes[key] and not base[key] for key in base),
                               losses=sum(base[key] and not outcomes[key] for key in base)))
    final = dict(generation=summary, cases_replayed=len(cases), usage=totals,
                 repairs=[{key: value for key, value in r.items() if key not in ("events", "request_usage", "input_order")}
                          for r in sorted(results, key=lambda r: (r["order_seed"], r["mode"]))],
                 paired=paired, limitation="Exploratory DSL benchmark; no financial outcomes, equal-token claim or independent-seed significance claim.")
    write_json(root / "results.json", final)
    return final


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="out/natural-repair/" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    args = parser.parse_args()
    load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)

    def factory(directory, calls, seed, temperature):
        return OpenAIClient(OpenAI(max_retries=0, timeout=120), ChatConfig(
            system_prompt=EXPLAIN_WITH_TEXT_DESC,
            logger=get_logger(str(directory.resolve()), str(directory / "llm.log"))),
            seed=seed, temperature=temperature, max_calls=calls, max_tokens=4096)
    try:
        result = run_benchmark(factory, args.output)
    except OpenAIError as error:
        raise SystemExit(f"Benchmark stopped: {type(error).__name__}. Completed artifacts were preserved.") from None
    print(json.dumps(result, indent=2))
    print(f"Benchmark artifacts: {args.output}")


if __name__ == "__main__":
    main()
