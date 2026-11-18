"""
An LLM judge that reads a sample of the dataset and scores it like a human reviewer would.

The numbers in quality.py measure variety, but they can't tell whether a row is *believable*.
A strong model reads a random sample and scores realism and variety, and lists concrete problems.
"""

import json
import random

from generator import call_model, describe_spec

JUDGE_MODEL = "openai/gpt-6-luna"
SAMPLE_SIZE = 20

JUDGE_SYSTEM = """
You review synthetic datasets for a QA team. Be strict and specific: a score of 10 means a human
could not tell the data from real data. Each row has an extra "edge_case" field added by the generator
(it is not part of the dataset). Rows with edge_case true are meant to be unusual or messy;
judge them on whether they are useful, believable test cases, not on whether they are clean.
""".strip()

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "realism": {"type": "integer", "description": "1-10: how believable the rows are as real data"},
        "variety": {"type": "integer", "description": "1-10: how different the rows are from each other"},
        "summary": {"type": "string", "description": "One sentence verdict"},
        "issues": {"type": "array", "items": {"type": "string"},
                   "description": "Up to 5 concrete problems, quoting the rows they come from"},
    },
    "required": ["realism", "variety", "summary", "issues"],
    "additionalProperties": False,
}


def judge_dataset(spec, rows, seed=None):
    """Score a random sample of rows. Returns the judge's dict, with token counts added."""
    sample = random.Random(seed).sample(rows, min(SAMPLE_SIZE, len(rows)))
    data_columns = [c.name for c in spec.columns]
    lines = [json.dumps({**{k: r[k] for k in data_columns}, "edge_case": r.get("_edge_case", False)},
                        ensure_ascii=False, default=str) for r in sample]
    prompt = f"""{describe_spec(spec)}

Here is a random sample of {len(sample)} rows from the generated dataset, one JSON object per line:

{chr(10).join(lines)}

Score the dataset and list its most important problems."""
    verdict, tokens_in, tokens_out = call_model(JUDGE_MODEL, JUDGE_SYSTEM, prompt, JUDGE_SCHEMA)
    verdict["realism"] = max(1, min(10, int(verdict["realism"])))
    verdict["variety"] = max(1, min(10, int(verdict["variety"])))
    verdict["issues"] = verdict["issues"][:5]
    verdict["tokens"] = (tokens_in, tokens_out)
    return verdict
