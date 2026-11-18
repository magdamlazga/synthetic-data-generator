"""
Synthetic Test Data Generator.

Describe a dataset and its columns, pick a model, and get realistic, varied rows plus edge cases for testing,
with a quality report. A second tab runs the same dataset on several models and compares them.

Run:  python app.py
"""

import re
import time
from pathlib import Path

import gradio as gr
import pandas as pd

from generator import META_COLUMNS, available_models, generate
from judge import JUDGE_MODEL, judge_dataset
from presets import DEFAULT_PRESET, PRESETS
from quality import quality_report, report_markdown
from schema import COLUMN_TYPES, DatasetSpec, parse_columns

HERE = Path(__file__).parent
DATASETS_DIR = HERE / "datasets"
REPORTS_DIR = HERE / "reports"
COLUMN_HEADERS = ["name", "type", "description", "options or range"]


# ---------- helpers ----------

def slugify(text):
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:40] or "dataset"


def build_spec(description, columns_table, preset_name):
    if not description or not description.strip():
        raise gr.Error("Describe the dataset first.")
    rows = columns_table.values.tolist() if isinstance(columns_table, pd.DataFrame) else columns_table
    try:
        columns = parse_columns(rows)
    except ValueError as e:
        raise gr.Error(str(e))
    return DatasetSpec(description=description.strip(), columns=columns, name=slugify(preset_name or description))


def to_frame(rows, spec, include_meta=True):
    order = [c.name for c in spec.columns] + (META_COLUMNS if include_meta else [])
    return pd.DataFrame(rows, columns=order) if rows else pd.DataFrame(columns=order)


def save_dataset(frame, spec, model_id):
    DATASETS_DIR.mkdir(exist_ok=True)
    stem = DATASETS_DIR / f"{spec.name}_{slugify(model_id.split('/')[-1])}_{time.strftime('%Y%m%d_%H%M%S')}"
    frame.to_csv(f"{stem}.csv", index=False)
    frame.to_json(f"{stem}.json", orient="records", indent=2, force_ascii=False)
    return [f"{stem}.csv", f"{stem}.json"]


def seed_value(seed):
    return int(seed) if seed not in (None, "") else None


def run_judge(spec, rows, seed, wanted):
    if not wanted or not rows:
        return None, ""
    try:
        return judge_dataset(spec, rows, seed=seed), ""
    except Exception as e:
        return None, f"\n\n_The LLM judge failed: {e}_"


def progress_line(result):
    note = f", {result.invalid} invalid, {result.duplicates} duplicates dropped" if result.invalid or result.duplicates else ""
    return f"⏳ **{result.model}**: {len(result.rows)} / {result.requested} rows ({result.batches} batches, {result.seconds:.0f} s{note})"


# ---------- tab 1: generate ----------

def run_generate(preset, description, columns_table, model_id, n_rows, edge_pct, temperature, seed, use_judge, include_meta):
    if not model_id:
        raise gr.Error("No model available. Set OPENAI_API_KEY in .env or start Ollama.")
    spec = build_spec(description, columns_table, preset)
    seed = seed_value(seed)

    result = None
    for result in generate(spec, model_id, int(n_rows), edge_ratio=edge_pct / 100, temperature=temperature, seed=seed):
        yield progress_line(result), to_frame(result.rows, spec), "", None

    if not result.rows:
        details = "\n".join(f"- {e}" for e in result.errors)
        yield f"❌ No valid rows were generated.\n\n{details}", to_frame([], spec), "", None
        return

    yield "⏳ Measuring quality…", to_frame(result.rows, spec), "", None
    report = quality_report(result.rows, spec.columns, result.invalid, result.duplicates)
    judge, judge_note = run_judge(spec, result.rows, seed, use_judge)
    frame = to_frame(result.rows, spec, include_meta)
    files = save_dataset(frame, spec, model_id)
    status = f"✅ **{len(result.rows)} rows** from **{model_id}** in {result.seconds:.0f} s. Saved to `datasets/`."
    yield status, to_frame(result.rows, spec), report_markdown(report, result, judge) + judge_note, files


# ---------- tab 2: compare models ----------

def comparison_row(result, report, judge):
    text = [t["distinct_2"] for t in report["text"].values() if t["distinct_2"] is not None]
    balance = [c["balance"] for c in report["categories"].values() if c["balance"] is not None]

    def pct(x):
        return None if x is None else round(100 * x, 1)

    return {
        "Model": result.model,
        "Rows": len(result.rows),
        "Valid %": pct(report["valid_rate"]),
        "Duplicates %": pct(report["duplicate_rate"]),
        "Diversity score": report["diversity_score"],
        "Avg similarity ↓": None if report["mean_similarity"] is None else round(report["mean_similarity"], 3),
        "Near-dup pairs %": pct(report["near_duplicate_pairs"]),
        "Distinct-2 %": pct(sum(text) / len(text)) if text else None,
        "Category balance %": pct(sum(balance) / len(balance)) if balance else None,
        "Judge realism": judge["realism"] if judge else None,
        "Judge variety": judge["variety"] if judge else None,
        "Time (s)": round(result.seconds, 1),
        "Rows/min": round(60 * len(result.rows) / max(result.seconds, 0.1), 1),
        "Output tokens": result.output_tokens,
        "Cost ($)": None if result.cost is None else round(result.cost, 4),
    }


# metric -> True if higher is better
BEST_DIRECTION = {
    "Valid %": True, "Duplicates %": False, "Diversity score": True, "Avg similarity ↓": False,
    "Near-dup pairs %": False, "Distinct-2 %": True, "Category balance %": True,
    "Judge realism": True, "Judge variety": True, "Rows/min": True, "Cost ($)": False,
}


def comparison_summary(table):
    if len(table) < 2:
        return ""
    lines = ["### Who wins what", ""]
    for metric, higher in BEST_DIRECTION.items():
        values = table[["Model", metric]].dropna()
        if len(values) < 2 or values[metric].nunique() == 1:
            continue
        best = values.loc[values[metric].idxmax() if higher else values[metric].idxmin()]
        lines.append(f"- **{metric}**: {best['Model']} ({best[metric]})")
    return "\n".join(lines)


def save_comparison(table, spec):
    REPORTS_DIR.mkdir(exist_ok=True)
    path = REPORTS_DIR / f"comparison_{spec.name}_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    table.to_csv(path, index=False)
    return str(path)


def run_compare(preset, description, columns_table, models, n_rows, edge_pct, temperature, seed, use_judge):
    if len(models or []) < 1:
        raise gr.Error("Pick at least one model.")
    spec = build_spec(description, columns_table, preset)
    seed = seed_value(seed)
    rows, samples, done_lines = [], [], []

    for model_id in models:
        result = None
        for result in generate(spec, model_id, int(n_rows), edge_ratio=edge_pct / 100, temperature=temperature, seed=seed):
            yield "\n\n".join(done_lines + [progress_line(result)]), pd.DataFrame(rows), "", pd.DataFrame(samples), None
        if not result.rows:
            done_lines.append(f"❌ **{model_id}**: no valid rows ({'; '.join(result.errors[:2])})")
            continue
        report = quality_report(result.rows, spec.columns, result.invalid, result.duplicates)
        judge, _ = run_judge(spec, result.rows, seed, use_judge)
        rows.append(comparison_row(result, report, judge))
        samples += [{k: r[k] for k in [c.name for c in spec.columns] + ["_model"]} for r in result.rows[:5]]
        done_lines.append(f"✅ **{model_id}**: {len(result.rows)} rows in {result.seconds:.0f} s")
        yield "\n\n".join(done_lines), pd.DataFrame(rows), "", pd.DataFrame(samples), None

    table = pd.DataFrame(rows)
    file = save_comparison(table, spec) if rows else None
    yield "\n\n".join(done_lines), table, comparison_summary(table) if rows else "", pd.DataFrame(samples), file


# ---------- UI ----------

def load_preset(name):
    preset = PRESETS[name]
    return preset["description"], pd.DataFrame(preset["columns"], columns=COLUMN_HEADERS), preset["edge"]


MODELS = available_models()
JUDGE_AVAILABLE = JUDGE_MODEL in MODELS
start = PRESETS[DEFAULT_PRESET]

with gr.Blocks(title="Synthetic Test Data Generator", theme=gr.themes.Soft()) as ui:
    gr.Markdown("# 🧪 Synthetic Test Data Generator\n"
                "Describe a dataset, define its columns, and generate realistic, varied rows and edge cases "
                "for testing apps and AI systems. Every dataset comes with a quality report, "
                "and you can compare models on the same task.")
    if not MODELS:
        gr.Markdown("> ⚠️ **No models found.** Add `OPENAI_API_KEY` to `.env` or start Ollama, then restart the app.")

    with gr.Group():
        preset = gr.Dropdown(list(PRESETS), value=DEFAULT_PRESET, label="Start from an example")
        description = gr.Textbox(start["description"], label="Dataset description", lines=2,
                                 info="What the data is, where it comes from and what it will be used for.")
        columns_table = gr.Dataframe(
            value=pd.DataFrame(start["columns"], columns=COLUMN_HEADERS), headers=COLUMN_HEADERS,
            datatype=["str"] * 4, col_count=(4, "fixed"), row_count=(len(start["columns"]), "dynamic"),
            interactive=True, wrap=True, label=f"Columns. Types: {', '.join(COLUMN_TYPES)}. "
            "Options: comma-separated values for a category, or a usual range like 1..5 for numbers.")

    with gr.Row():
        edge_pct = gr.Slider(0, 100, value=start["edge"], step=5, label="Edge cases (%)",
                             info="Share of rows that are deliberately messy or extreme, for testing")
        temperature = gr.Slider(0.0, 1.5, value=0.9, step=0.1, label="Temperature (local models)",
                                info="OpenAI reasoning models ignore it")
        seed = gr.Number(value=42, precision=0, label="Seed", info="Same seed, same scenarios order. Clear for random.")

    with gr.Tabs():
        with gr.Tab("Generate"):
            with gr.Row():
                model = gr.Dropdown(MODELS, value=MODELS[0] if MODELS else None, label="Model")
                n_rows = gr.Slider(10, 500, value=50, step=10, label="Rows")
            with gr.Row():
                use_judge = gr.Checkbox(JUDGE_AVAILABLE, interactive=JUDGE_AVAILABLE,
                                        label=f"Score realism with an LLM judge ({JUDGE_MODEL}, 1 extra call)")
                include_meta = gr.Checkbox(True, label="Include _scenario, _edge_case, _model columns in the download")
            generate_btn = gr.Button("Generate dataset", variant="primary")
            status = gr.Markdown()
            data = gr.Dataframe(label="Dataset", wrap=True, interactive=False, max_height=500)
            with gr.Row():
                with gr.Column(scale=3):
                    report = gr.Markdown()
                with gr.Column(scale=1):
                    files = gr.File(label="Download", file_count="multiple")

        with gr.Tab("Compare models"):
            gr.Markdown("Run the same dataset definition, seed and settings on several models, "
                        "and compare validity, variety, speed and cost side by side.")
            with gr.Row():
                models = gr.CheckboxGroup(MODELS, value=MODELS, label="Models")
                compare_rows = gr.Slider(10, 200, value=30, step=10, label="Rows per model")
            compare_judge = gr.Checkbox(JUDGE_AVAILABLE, interactive=JUDGE_AVAILABLE,
                                        label=f"Score realism with an LLM judge ({JUDGE_MODEL})")
            compare_btn = gr.Button("Compare models", variant="primary")
            compare_status = gr.Markdown()
            comparison = gr.Dataframe(label="Comparison", interactive=False)
            summary = gr.Markdown()
            samples = gr.Dataframe(label="First 5 rows from each model", wrap=True, interactive=False, max_height=400)
            comparison_file = gr.File(label="Download comparison")

    preset.change(load_preset, preset, [description, columns_table, edge_pct])
    generate_btn.click(
        run_generate,
        [preset, description, columns_table, model, n_rows, edge_pct, temperature, seed, use_judge, include_meta],
        [status, data, report, files])
    compare_btn.click(
        run_compare,
        [preset, description, columns_table, models, compare_rows, edge_pct, temperature, seed, compare_judge],
        [compare_status, comparison, summary, samples, comparison_file])

if __name__ == "__main__":
    ui.launch(inbrowser=True)
