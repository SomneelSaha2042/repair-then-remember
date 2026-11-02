"""Repair a fixed ordered corpus, keeping original generations identical across arms."""
import argparse
import json
from pathlib import Path
from dotenv import load_dotenv

from alphagen_llm.repair_memory import RepairEngine, build_parser
from alphagen_llm.client.openai_client import DEFAULT_MODEL


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", help="JSONL records containing a response field (original model output)")
    parser.add_argument("--mode", choices=("none", "nearest", "random", "shuffled"), default="none")
    parser.add_argument("--output", required=True, help="New empty output directory")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-calls", type=int, default=1000)
    parser.add_argument("--memory-input", help="Verified source-model memory to transfer; copied and revalidated")
    args = parser.parse_args()
    load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
    from openai import OpenAI
    from alphagen_llm.client import OpenAIClient, ChatConfig
    from alphagen_llm.prompts.system_prompt import EXPLAIN_WITH_TEXT_DESC
    from alphagen.utils import get_logger
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    corpus = Path(args.corpus).read_text(encoding="utf-8")
    (output / "corpus.jsonl").write_text(corpus, encoding="utf-8")
    if args.memory_input:
        (output / "memory.jsonl").write_bytes(Path(args.memory_input).read_bytes())
    client = OpenAIClient(OpenAI(max_retries=0), ChatConfig(
        system_prompt=EXPLAIN_WITH_TEXT_DESC,
        logger=get_logger("replay", str(output / "llm.log"))),
        model=args.model, seed=args.seed, max_calls=args.max_calls)
    engine = RepairEngine(build_parser(), args.mode, output / "memory.jsonl", args.k, args.seed, args.model)
    for row in corpus.splitlines():
        if row.strip():
            engine.process(json.loads(row)["response"], client, repair=True)
    (output / "results.json").write_text(json.dumps(dict(
        config=vars(args), structural_metrics=engine.summary(), llm_usage=client.usage,
        request_usage=client.request_usage,
        events=engine.events), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
