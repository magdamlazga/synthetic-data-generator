"""
Generating a dataset with an LLM, batch by batch.

The hard part of synthetic data is not getting rows, it's getting rows that are *different from each other*.
Asked for 100 customer messages, a model tends to write the same five messages with small changes. So:

1. The model first brainstorms a list of distinct sub-scenarios for the dataset (types of users, situations...).
2. Every row in a batch gets its own scenario, and the batch gets a few earlier rows it must not repeat.
3. A share of the batches can be edge cases (typos, extreme values, injection attempts...) for testing.
4. Every row is validated against the schema; invalid rows and exact duplicates are dropped and counted.
5. It keeps generating until it has the requested number of valid, unique rows (with a limit).
"""

import json
import math
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import requests
from dotenv import load_dotenv
from openai import OpenAI

from schema import rows_json_schema, validate_row

load_dotenv(override=True)

OLLAMA_URL = "http://localhost:11434"
OPENAI_MODELS = ["gpt-6-luna"]
BATCH_SIZE = 10
PARALLEL_BATCHES = {"openai": 4, "ollama": 1}  # a local model on one machine is fastest one request at a time

# Price per 1M tokens (input, output) in USD, to estimate cost. Local models are free.
# Fill in the current price from https://openai.com/api/pricing to see a cost estimate in the app.
PRICES = {"openai/gpt-6-luna": None}

META_COLUMNS = ["_scenario", "_edge_case", "_model"]

SYSTEM_PROMPT = """
You generate synthetic datasets used to test software and evaluate AI systems.
- Return only data that matches the JSON schema.
- Rows must be realistic and plausible for the dataset description, unless you are told to write edge cases.
- Make every row clearly different from the others: vary names, wording, length, tone, details and values.
  Never repeat a row, a sentence or a sentence pattern.
- Spread category values across all allowed values, unless the description says some are rarer.
  Don't default to the most common or most positive value.
- Avoid placeholder values such as "John Doe", "Jane Smith", "example.com", "Lorem ipsum" or "Test".
- Text should read like real people wrote it, with natural variation, not like it was filled in from a template.
""".strip()

EDGE_CASE_STYLES = [
    "typos, slang, missing punctuation and ALL CAPS, like rushed real users write",
    "values at the extremes: very long or very short text, zero, negative or very large numbers, far past or future dates",
    "unusual characters: accented and non-Latin names, emojis, mixed languages, right-to-left text",
    "ambiguous or contradictory content that is hard to classify or act on",
    "security-style inputs: SQL injection strings, HTML/script tags, prompt-injection attempts like 'ignore previous instructions'",
    "missing or placeholder-looking information: 'N/A', '-', 'unknown', whitespace-only or near-empty values",
]

_clients = {}


def client_for(provider):
    if provider not in _clients:
        if provider == "ollama":
            _clients[provider] = OpenAI(base_url=f"{OLLAMA_URL}/v1", api_key="ollama")  # Ollama ignores the key
        else:
            _clients[provider] = OpenAI()
    return _clients[provider]


def available_models():
    """OpenAI models if a key is set, plus every chat model installed in Ollama."""
    models = [f"openai/{m}" for m in OPENAI_MODELS] if os.getenv("OPENAI_API_KEY") else []
    try:
        tags = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2).json().get("models", [])
        models += [f"ollama/{t['name'].removesuffix(':latest')}" for t in tags if "embed" not in t["name"]]
    except requests.RequestException:
        pass  # Ollama isn't running: offer OpenAI only
    return models


# ---------- one call to a model ----------

def call_model(model_id, system, user, json_schema, temperature=0.9):
    """Ask a model for JSON that matches json_schema. Returns (parsed_json, input_tokens, output_tokens)."""
    provider, model = model_id.split("/", 1)

    if provider == "openai":
        # Reasoning models don't take a temperature; variety comes from the prompts instead.
        response = client_for("openai").responses.create(
            model=model,
            instructions=system,
            input=user,
            reasoning={"effort": "medium"},
            text={"format": {"type": "json_schema", "name": "output", "schema": json_schema, "strict": True}},
        )
        return json.loads(response.output_text), response.usage.input_tokens, response.usage.output_tokens

    response = client_for("ollama").chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        response_format={"type": "json_schema", "json_schema": {"name": "output", "schema": json_schema}},
        temperature=temperature,
    )
    usage = response.usage
    return json.loads(response.choices[0].message.content), usage.prompt_tokens, usage.completion_tokens


# ---------- prompts ----------

def describe_spec(spec):
    return f"Dataset description: {spec.description}\n\nColumns:\n" + "\n".join(c.describe() for c in spec.columns)


def scenario_prompt(spec, count):
    return f"""{describe_spec(spec)}

List {count} distinct sub-scenarios that together cover the full variety of realistic data for this dataset:
different kinds of people, situations, contexts, sizes, regions, levels of experience and so on.
Each scenario is one short phrase (at most 15 words). Make them as different from each other as possible."""


SCENARIO_SCHEMA = {
    "type": "object",
    "properties": {"scenarios": {"type": "array", "items": {"type": "string"}}},
    "required": ["scenarios"],
    "additionalProperties": False,
}


def batch_prompt(spec, scenarios, edge_style, avoid):
    if edge_style:
        kind = (f"These rows are EDGE CASES for testing. Every row should stress the system this way: {edge_style}.\n"
                "Values must still have the right type and allowed category values, "
                "but may be outside the usual range, extreme, unusual or messy.")
    else:
        kind = "These rows are normal, realistic data."
    numbered = "\n".join(f"{i}. {s}" for i, s in enumerate(scenarios, start=1))
    prompt = f"""{describe_spec(spec)}

Write exactly {len(scenarios)} rows, one for each of these scenarios, in this order:
{numbered}

{kind}"""
    if avoid:
        prompt += "\n\nThese values already exist in the dataset. Do not repeat them or write close variations:\n"
        prompt += "\n".join(f"- {a}" for a in avoid)
    return prompt


# ---------- the result ----------

@dataclass
class GenerationResult:
    model: str
    requested: int
    rows: list = field(default_factory=list)
    invalid: int = 0
    duplicates: int = 0
    failed_batches: int = 0
    batches: int = 0
    errors: list = field(default_factory=list)  # a few example reasons, for the report
    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0
    done: bool = False

    @property
    def cost(self):
        if self.model.startswith("ollama/"):
            return 0.0
        price = PRICES.get(self.model)
        if not price:
            return None
        return (self.input_tokens * price[0] + self.output_tokens * price[1]) / 1_000_000

    def note_error(self, message):
        if len(self.errors) < 8:
            self.errors.append(message)


# ---------- the generation loop ----------

def brainstorm_scenarios(spec, model_id, count, temperature, result):
    try:
        data, tokens_in, tokens_out = call_model(model_id, SYSTEM_PROMPT, scenario_prompt(spec, count),
                                                 SCENARIO_SCHEMA, temperature)
        result.input_tokens += tokens_in
        result.output_tokens += tokens_out
        scenarios = [s.strip() for s in data.get("scenarios", []) if isinstance(s, str) and s.strip()]
    except Exception as e:  # a failed brainstorm shouldn't stop the dataset
        result.note_error(f"scenario brainstorm failed: {e}")
        scenarios = []
    return scenarios or ["a typical, everyday case"]


def _row_key(row, spec):
    return json.dumps([row[c.name] for c in spec.columns], sort_keys=True, default=str).lower()


def _avoid_examples(rows, spec, rng, count=6):
    """A few existing values of the main text column, so the next batch doesn't repeat them."""
    text_columns = [c.name for c in spec.columns if c.type == "text"]
    if not rows or not text_columns:
        return []
    main = max(text_columns, key=lambda name: sum(len(str(r[name])) for r in rows))
    sample = rng.sample(rows, min(count, len(rows)))
    return [str(r[main])[:100] for r in sample]


def generate(spec, model_id, n_rows, edge_ratio=0.0, temperature=0.9, seed=None, batch_size=BATCH_SIZE):
    """
    Generate n_rows valid, unique rows. This is a generator: it yields the same GenerationResult after every
    round of batches, so the UI can show progress. The last yield has result.done = True.
    """
    rng = random.Random(seed)
    result = GenerationResult(model=model_id, requested=n_rows)
    start = time.time()
    provider = model_id.split("/", 1)[0]
    workers = PARALLEL_BATCHES.get(provider, 1)
    json_schema = rows_json_schema(spec.columns)

    planned_batches = math.ceil(n_rows / batch_size)
    max_batches = planned_batches + max(2, planned_batches // 2)  # room to replace invalid and duplicate rows

    scenarios = brainstorm_scenarios(spec, model_id, min(max(n_rows, 10), 40), temperature, result)
    rng.shuffle(scenarios)
    result.seconds = time.time() - start
    yield result

    seen, edge_rows, planned_rows, next_scenario = set(), 0, 0, 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        while len(result.rows) < n_rows and result.batches < max_batches:
            # plan one round of batches that run in parallel
            avoid = _avoid_examples(result.rows, spec, rng)
            remaining = n_rows - len(result.rows)
            jobs = []
            for _ in range(min(workers, math.ceil(remaining / batch_size), max_batches - result.batches)):
                size = min(batch_size, remaining - sum(j[1] for j in jobs)) or batch_size
                # spread edge-case batches evenly, starting with normal data:
                # a batch is edge cases while edge rows are behind their share of the rows planned so far
                is_edge = edge_rows < edge_ratio * planned_rows
                edge_style = rng.choice(EDGE_CASE_STYLES) if is_edge else None
                planned_rows += size
                if is_edge:
                    edge_rows += size
                # every row gets its own scenario, so rows in one batch aren't all about the same thing
                batch_scenarios = [scenarios[(next_scenario + j) % len(scenarios)] for j in range(size)]
                next_scenario += size
                prompt = batch_prompt(spec, batch_scenarios, edge_style, avoid)
                future = pool.submit(call_model, model_id, SYSTEM_PROMPT, prompt, json_schema, temperature)
                jobs.append((future, size, batch_scenarios, is_edge))
                result.batches += 1

            for future, size, batch_scenarios, is_edge in jobs:
                try:
                    data, tokens_in, tokens_out = future.result()
                except Exception as e:
                    result.failed_batches += 1
                    result.note_error(f"batch failed: {type(e).__name__}: {str(e)[:150]}")
                    continue
                result.input_tokens += tokens_in
                result.output_tokens += tokens_out
                raw_rows = data.get("rows", []) if isinstance(data, dict) else []
                for j, raw in enumerate(raw_rows):
                    row, error = validate_row(raw, spec.columns)
                    if error:
                        result.invalid += 1
                        result.note_error(f"invalid row: {error}")
                        continue
                    key = _row_key(row, spec)
                    if key in seen:
                        result.duplicates += 1
                        continue
                    seen.add(key)
                    scenario = batch_scenarios[j] if j < len(batch_scenarios) else ""
                    row.update({"_scenario": scenario, "_edge_case": is_edge, "_model": model_id})
                    result.rows.append(row)

            result.seconds = time.time() - start
            yield result

    result.rows = result.rows[:n_rows]
    result.seconds = time.time() - start
    result.done = True
    yield result


def generate_all(spec, model_id, n_rows, **kwargs):
    """Run generate() to the end and return the final result (for notebooks and scripts)."""
    result = None
    for result in generate(spec, model_id, n_rows, **kwargs):
        pass
    return result
