"""Causal repair episodes; verification uses the same DSL parser as generation."""
import hashlib
import json
import random
import re
from collections import Counter
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from uuid import uuid4

from alphagen.data.parser import ExpressionParser
from alphagen.data.expression import Operators, Greater, Less, Sub


def build_parser():
    return ExpressionParser(
        Operators, ignore_case=True, non_positive_time_deltas_allowed=False,
        additional_operator_mapping={"Max": [Greater], "Min": [Less], "Delta": [Sub]},
    )


def parse_candidate(parser, text):
    try:
        expr = parser.parse(text.replace("$", ""))
        if not expr.is_featured:
            raise ValueError("A candidate must be a featured expression")
        return expr, None
    except Exception as error:
        return None, f"{type(error).__name__}: {error}"


def error_class(message):
    if re.search(r"expects \d+ operand", message):
        return "operator_arity"
    if "DeltaTime" in message or "time difference" in message:
        return "window_type"
    if "featured" in message or "normal expression" in message:
        return "operand_type"
    if "operator/feature" in message or "find the feature" in message:
        return "unknown_symbol"
    if any(word in message for word in ("parenthesis", "stack", "tokens left")):
        return "parenthesis"
    return "syntax"


def operator_name(text, message):
    match = re.search(r"(\w+) expects", message) or re.search(r"(\w+)\s*\(", text)
    return match.group(1).lower() if match else "unknown"


def token_edit_distance(before, after):
    """Observed token Levenshtein distance, not a proof of minimal repair."""
    tokenize = lambda text: re.findall(r"[A-Za-z_]\w*|\d+(?:\.\d+)?|[^\s]", text)
    a, b = tokenize(before), tokenize(after)
    row = list(range(len(b) + 1))
    for i, left in enumerate(a, 1):
        new = [i]
        for j, right in enumerate(b, 1):
            new.append(min(new[-1] + 1, row[j] + 1, row[j - 1] + (left != right)))
        row = new
    return row[-1]


def expression_shape(text):
    """Lexical sizes work on invalid expressions too; depth counts parentheses."""
    depth = maximum = 0
    for char in text:
        depth += (char == "(") - (char == ")")
        maximum = max(maximum, depth)
    return dict(characters=len(text),
                lexical_tokens=len(re.findall(r"[A-Za-z_]\w*|\d+(?:\.\d+)?|[^\s]", text)),
                operators=len(re.findall(r"[A-Za-z_]\w*\s*\(", text)),
                nesting_depth=maximum)


class RepairEngine:
    def __init__(self, parser, mode="none", path=None, k=1, seed=0, source_model="unknown"):
        if mode not in ("none", "nearest", "random", "shuffled") or k < 1:
            raise ValueError("Expected mode none/nearest/random/shuffled and k >= 1")
        self.parser, self.mode, self.k = parser, mode, k
        self.source_model = source_model
        self.path = Path(path) if path else None
        self.rng = random.Random(seed)
        self.records, self.events, self.generations = [], [], []
        self.context = dict(phase="standalone", ppo_timestep=0)
        self.shapes = []
        self.counts, self.errors, self.unique = Counter(), Counter(), set()
        # Include parser flags, aliases and implementation in compatibility identity.
        identity = {key: repr(value) for key, value in vars(parser).items()
                    if key not in ("_stack", "_tokens")}
        code = Path(__file__).parents[1] / "alphagen" / "data"
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode())
        for name in ("parser.py", "expression.py"):
            digest.update((code / name).read_text(encoding="utf-8").encode())
        self.dsl_id = digest.hexdigest()
        if self.path and self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                record = json.loads(line)
                if record["dsl_id"] != self.dsl_id:
                    raise ValueError("Memory DSL fingerprint differs from this parser")
                invalid, _ = parse_candidate(parser, record["invalid_fragment"])
                repaired, _ = parse_candidate(parser, record["repaired_fragment"])
                if invalid is not None or repaired is None:
                    raise ValueError("Memory contains an unverified repair")
                self.records.append(record)

    def retrieve(self, text, message):
        if self.mode == "none":
            return []
        candidates = list(self.records)
        if self.mode in ("random", "shuffled"):
            self.rng.shuffle(candidates)
        else:
            candidates = [r for r in candidates if r["error_class"] == error_class(message)]
            candidates.sort(key=lambda r: (
                r["operator"] == operator_name(text, message),
                SequenceMatcher(None, text, r["invalid_fragment"]).ratio(),
            ), reverse=True)
        selected = candidates[:self.k]
        if self.mode == "shuffled" and selected:
            # Placebo changes the mapping only in the prompt; never poison stored repairs.
            selected = [dict(r, repaired_fragment=candidates[(i + 1) % len(candidates)]["repaired_fragment"])
                        for i, r in enumerate(selected)]
        return selected

    def repair(self, client, text, message):
        from alphagen_llm.prompts.common import get_fixer_prompt
        original, verified_error = parse_candidate(self.parser, text)
        if original is not None:
            raise ValueError("Repair requires an invalid expression")
        message = verified_error
        examples = self.retrieve(text, message)  # Snapshot before this episode is finalized.
        prompt = get_fixer_prompt(text, message)
        if examples:
            prompt += "\nPrior repair cases (use only if applicable):\n" + "\n".join(
                f"{r['invalid_fragment']} -> {r['repaired_fragment']}" for r in examples)
        self.counts["repair_calls"] += 1
        self.counts["retrieved_cases"] += len(examples)
        client.reset()
        try:
            result = client.chat_complete(prompt).strip()
        finally:
            client.reset()
        expr, repair_error = parse_candidate(self.parser, result)
        event = dict(invalid_fragment=text, parser_message=message,
                     error_class=error_class(message), retrieved_ids=[r["episode_id"] for r in examples],
                     repaired_fragment=result, success=expr is not None, repair_error=repair_error,
                     request_usage=getattr(client, "request_usage", [None])[-1])
        self.events.append(event)
        self._log("repair_events.jsonl", event)
        if expr is not None:
            self.counts["repair_successes"] += 1
            # Arity/types are checked by operator.validate_parameters inside parser.parse.
            record = dict(event, episode_id=uuid4().hex, dsl_id=self.dsl_id,
                          operator=operator_name(text, message),
                          repair_operation="verified_expression_rewrite",
                          type_signature_before="invalid", type_signature_after="feature",
                          verification=dict(parser_pass=True, type_checker_pass=True,
                                            token_edit_distance=token_edit_distance(text, result)),
                          source_model=self.source_model, prompt_language="English",
                          finalized_at=datetime.now(timezone.utc).isoformat())
            if self.mode != "none":
                self.records.append(record)
                if self.path:
                    self.path.parent.mkdir(parents=True, exist_ok=True)
                    with self.path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(record) + "\n")
        return result

    def process(self, response, client, repair=False):
        expressions = []
        self.counts["generation_calls"] += 1
        self.generations.append(dict(response=response, source_model=self.source_model, **self.context))
        self._log("generations.jsonl", self.generations[-1])
        # Finalize repairs only after inspecting all initial outputs in this generation.
        candidates = [(line.strip(), *parse_candidate(self.parser, line.strip()))
                      for line in response.splitlines() if line.strip()]
        for text, expr, message in candidates:
            shape = expression_shape(text)
            self.shapes.append(shape)
            self._log("candidates.jsonl", dict(expression=text, parser_pass=expr is not None,
                                              parser_error=message, shape=shape, **self.context))
            self.counts["initial_candidates"] += 1
            self.counts["initial_valid"] += int(expr is not None)
            if expr is None:
                category = error_class(message)
                self.counts["recurrent_errors"] += int(self.errors[category] > 0)
                self.errors[category] += 1
                if repair:
                    expr, _ = parse_candidate(self.parser, self.repair(client, text, message))
            if expr is not None:
                expressions.append(expr)
                self.unique.add(str(expr))
        self.counts["final_valid"] += len(expressions)
        return expressions

    def _log(self, filename, event):
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with (self.path.parent / filename).open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event) + "\n")

    def summary(self):
        shapes = {key: dict(mean=sum(row[key] for row in self.shapes)/len(self.shapes),
                            maximum=max(row[key] for row in self.shapes))
                  for key in self.shapes[0]} if self.shapes else {}
        return dict(self.counts, unique_valid=len(self.unique), error_classes=dict(self.errors),
                    memory_size=len(self.records), mode=self.mode, retrieval_k=self.k,
                    dsl_id=self.dsl_id, expression_shapes=shapes)
