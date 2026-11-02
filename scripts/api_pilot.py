"""Small live API check: generate formulas, then compare repairs on fixed DSL failures."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path

from openai import OpenAI, OpenAIError
from dotenv import load_dotenv
from alphagen_llm.client import OpenAIClient, ChatConfig
from alphagen_llm.client.openai_client import DEFAULT_MODEL
from alphagen_llm.prompts.system_prompt import EXPLAIN_WITH_TEXT_DESC
from alphagen_llm.repair_memory import RepairEngine, build_parser
from alphagen.utils import get_logger


# Synthetic syntax failures ensure the wiring can be checked even if generation is valid.
REPAIR_CASES = ["Mean(close)", "Mean(open)", "Corr(close,volume)", "Corr(high,low)",
                "Ref(close)", "Ref(volume)", "Log(close,20)", "Log(volume,20)"]


def run_pilot(factory, output, model=DEFAULT_MODEL):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    generation_client = factory(output / "generation", 1)
    response = generation_client.chat_complete(
        "Generate ten different formulaic alphas using only the DSL. "
        "One formula per line, without numbering or explanations.")
    engine = RepairEngine(build_parser(), source_model=model)
    engine.process(response, generation_client)
    corpus = [dict(response=text) for text in REPAIR_CASES]
    (output / "corpus.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in corpus), encoding="utf-8")
    results = dict(model=model, generation=dict(response=response, metrics=engine.summary(),
                                               usage=generation_client.usage), repairs={})
    for mode in ("none", "nearest"):
        client = factory(output / mode, len(REPAIR_CASES))
        engine = RepairEngine(build_parser(), mode, output / mode / "repair_memory.jsonl",
                              source_model=model)
        for row in corpus:
            engine.process(row["response"], client, repair=True)
        results["repairs"][mode] = dict(metrics=engine.summary(), usage=client.usage,
                                       request_usage=client.request_usage)
    results["interpretation"] = "Integration check using synthetic DSL errors; not evidence of a memory benefit."
    (output / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output", default="out/api-pilot/" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    args = parser.parse_args()
    load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
    if not os.environ.get("OPENAI_API_KEY"):
        parser.error("Set OPENAI_API_KEY privately in this terminal before running; do not paste it into chat.")

    def factory(directory, max_calls):
        return OpenAIClient(OpenAI(max_retries=0), ChatConfig(
            system_prompt=EXPLAIN_WITH_TEXT_DESC,
            logger=get_logger("pilot", str(directory / "llm.log"))),
            model=args.model, max_calls=max_calls)

    try:
        results = run_pilot(factory, args.output, args.model)
    except OpenAIError as error:
        raise SystemExit(f"API pilot failed ({type(error).__name__}); verify key, model access and billing.") from None
    print(json.dumps(results, indent=2))
    print(f"Pilot artifacts: {args.output}")


if __name__ == "__main__":
    main()
