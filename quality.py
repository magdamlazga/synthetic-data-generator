"""
Measuring how good a synthetic dataset is, without trusting the model that wrote it.

- Validity:   how many rows the model returned that didn't fit the schema
- Uniqueness: how many rows were exact duplicates
- Variety of text:
    * distinct-2: share of word pairs that are unique across a column (1.0 = no phrase is ever reused)
    * similarity: average TF-IDF cosine similarity between rows (0 = nothing in common, 1 = identical),
      and how many pairs of rows are near-duplicates (similarity >= 0.8)
- Balance of categories: normalised entropy (1.0 = all values used equally, 0 = always the same value)
- Numbers: min, max, mean and values outside the usual range

These are combined into one diversity score from 0 to 100.
"""

import math
import re
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from schema import out_of_range

NEAR_DUPLICATE = 0.8
WORD = re.compile(r"\w+", re.UNICODE)


def distinct_2(values):
    pairs = []
    for value in values:
        words = WORD.findall(str(value).lower())
        pairs += list(zip(words, words[1:]))
    return len(set(pairs)) / len(pairs) if pairs else None


def text_similarity(texts):
    """(mean pairwise similarity, share of near-duplicate pairs), or (None, None) if it can't be measured."""
    texts = [t for t in texts if t.strip()]
    if len(texts) < 2:
        return None, None
    try:
        matrix = TfidfVectorizer().fit_transform(texts)
    except ValueError:  # only stop words / empty vocabulary
        return None, None
    sims = cosine_similarity(matrix)
    upper = sims[np.triu_indices(len(texts), k=1)]
    return float(upper.mean()), float((upper >= NEAR_DUPLICATE).mean())


def category_balance(values, options):
    counts = Counter(values)
    if len(options) < 2 or not values:
        return None, counts
    total = len(values)
    entropy = -sum((n / total) * math.log(n / total) for n in counts.values())
    return entropy / math.log(len(options)), counts


def quality_report(rows, columns, invalid=0, duplicates=0):
    """Compute the quality metrics of a list of row dicts. Returns a dict (see report_markdown)."""
    n = len(rows)
    returned = n + invalid + duplicates
    report = {
        "rows": n,
        "edge_case_rows": sum(1 for r in rows if r.get("_edge_case")),
        "valid_rate": (returned - invalid) / returned if returned else None,
        "duplicate_rate": duplicates / (returned - invalid) if returned - invalid else None,
        "text": {}, "categories": {}, "numbers": {},
    }

    text_columns = [c for c in columns if c.type == "text"]
    for column in text_columns:
        values = [str(r[column.name]) for r in rows]
        report["text"][column.name] = {
            "distinct_2": distinct_2(values),
            "avg_words": float(np.mean([len(WORD.findall(v)) for v in values])) if values else 0.0,
        }
    # similarity is measured on whole rows (all text columns together), which is what a reader perceives
    row_texts = [" ".join(str(r[c.name]) for c in text_columns) for r in rows]
    report["mean_similarity"], report["near_duplicate_pairs"] = text_similarity(row_texts) if text_columns else (None, None)

    for column in columns:
        if column.type == "category":
            balance, counts = category_balance([r[column.name] for r in rows], column.options)
            report["categories"][column.name] = {
                "balance": balance,
                "coverage": len(counts) / len(column.options),
                "counts": {o: counts.get(o, 0) for o in column.options},
            }
        elif column.type in ("integer", "number") and rows:
            values = [r[column.name] for r in rows]
            outside = [r for r in rows if out_of_range(r[column.name], column)]
            report["numbers"][column.name] = {
                "min": min(values), "max": max(values), "mean": float(np.mean(values)),
                "out_of_range": len(outside),
                "out_of_range_in_edge_cases": sum(1 for r in outside if r.get("_edge_case")),
            }
        elif column.type == "boolean" and rows:
            true_share = sum(1 for r in rows if r[column.name]) / n
            report["categories"][column.name] = {
                "balance": category_balance([r[column.name] for r in rows], [True, False])[0],
                "coverage": len({r[column.name] for r in rows}) / 2,
                "counts": {"true": round(true_share * n), "false": n - round(true_share * n)},
            }

    report["diversity_score"] = diversity_score(report)
    return report


def diversity_score(report):
    """Average of the variety signals that could be measured, as 0-100."""
    parts = []
    if report["mean_similarity"] is not None:
        parts.append(1 - report["mean_similarity"])
    d2 = [t["distinct_2"] for t in report["text"].values() if t["distinct_2"] is not None]
    if d2:
        parts.append(float(np.mean(d2)))
    balances = [c["balance"] for c in report["categories"].values() if c["balance"] is not None]
    if balances:
        parts.append(float(np.mean(balances)))
    if report["duplicate_rate"] is not None:
        parts.append(1 - report["duplicate_rate"])
    return round(100 * float(np.mean(parts)), 1) if parts else None


# ---------- presenting it ----------

def _pct(x):
    return "n/a" if x is None else f"{100 * x:.0f}%"


def _num(x, digits=2):
    return "n/a" if x is None else f"{x:.{digits}f}"


def report_markdown(report, result=None, judge=None):
    lines = [f"### Quality report: diversity score **{report['diversity_score'] if report['diversity_score'] is not None else 'n/a'} / 100**", ""]
    lines += [
        "| Check | Result |", "|---|---|",
        f"| Rows | {report['rows']} ({report['edge_case_rows']} edge cases) |",
        f"| Valid rows (fit the schema) | {_pct(report['valid_rate'])} |",
        f"| Exact duplicates removed | {_pct(report['duplicate_rate'])} |",
        f"| Average similarity between rows (lower is better) | {_num(report['mean_similarity'])} |",
        f"| Near-duplicate pairs (similarity ≥ {NEAR_DUPLICATE}) | {_pct(report['near_duplicate_pairs'])} |",
    ]
    if result is not None:
        cost = "free (local)" if result.cost == 0 else ("set a price in generator.py" if result.cost is None else f"${result.cost:.4f}")
        lines += [
            f"| Time | {result.seconds:.0f} s ({60 * report['rows'] / max(result.seconds, 0.1):.0f} rows/min) |",
            f"| Tokens (in / out) | {result.input_tokens:,} / {result.output_tokens:,} |",
            f"| Cost | {cost} |",
        ]
        if result.failed_batches:
            lines.append(f"| Failed batches | {result.failed_batches} of {result.batches} |")
    if judge:
        lines.append(f"| LLM judge: realism | {judge['realism']} / 10 |")
        lines.append(f"| LLM judge: variety | {judge['variety']} / 10 |")

    if report["text"]:
        lines += ["", "**Text columns**", "", "| Column | Unique word pairs (distinct-2) | Avg. words |", "|---|---|---|"]
        lines += [f"| {name} | {_pct(t['distinct_2'])} | {t['avg_words']:.0f} |" for name, t in report["text"].items()]
    if report["categories"]:
        lines += ["", "**Categories**", "", "| Column | Balance | Values used | Counts |", "|---|---|---|---|"]
        for name, c in report["categories"].items():
            counts = ", ".join(f"{k}: {v}" for k, v in c["counts"].items())
            lines.append(f"| {name} | {_pct(c['balance'])} | {_pct(c['coverage'])} | {counts} |")
    if report["numbers"]:
        lines += ["", "**Numbers**", "", "| Column | Min | Max | Mean | Outside usual range |", "|---|---|---|---|---|"]
        for name, x in report["numbers"].items():
            outside = f"{x['out_of_range']} ({x['out_of_range_in_edge_cases']} in edge cases)" if x["out_of_range"] else "0"
            lines.append(f"| {name} | {x['min']} | {x['max']} | {x['mean']:.1f} | {outside} |")
    if judge and judge.get("issues"):
        lines += ["", f"**LLM judge says:** {judge['summary']}", ""] + [f"- {issue}" for issue in judge["issues"]]
    if result is not None and result.errors:
        lines += ["", "<details><summary>Problems the generator ran into</summary>", ""]
        lines += [f"- {e}" for e in result.errors] + ["", "</details>"]
    return "\n".join(lines)
