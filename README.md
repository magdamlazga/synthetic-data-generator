# 🧪 Synthetic Test Data Generator

**Generate realistic test data with AI, and measure how good it is before you use it.**

Describe the data you need, for example *"messages customers send to a support chatbot"*. Pick a model, and you get a ready-to-use dataset with a quality report.

## The problem

Testers and AI teams always need data they don't have: chatbot messages, sign-up inputs, orders, reviews. Real data is often private, and writing it by hand is slow.

AI can write it in seconds, but the result usually has problems:

- 🔁 **It repeats itself.** 100 messages turn out to be the same 5, reworded.
- ✨ **It's too clean.** No typos, no weird inputs, no one trying to break things.
- 🙈 **Nobody checks it.** It looks fine, so it goes straight into the tests.

This app solves all three.

## What it does

- **Any dataset you describe.** You choose the columns: text, numbers, categories, dates, yes/no.
- **Varied data.** Every row is written for a different scenario, and the model is told what not to repeat.
- **Edge cases on demand.** A slider adds messy rows: typos, emojis, extreme values, SQL injection, "ignore previous instructions"…
- **Quality report.** It measures duplicates, how similar the rows are, and whether categories are used evenly, and gives one diversity score from 0 to 100.
- **AI judge.** A strong model reads a sample, scores realism and variety from 1 to 10, and points out concrete problems.
- **Model comparison.** It runs the same task on several models and shows which one wins where.
- **Five ready-made examples** to start from: chatbot messages, bug reports, orders, sign-up forms and product reviews.

## Results

Same task (30 support chatbot messages) on two models:

| | gpt-6-luna (cloud) | llama3.2 (local) |
|---|---|---|
| Valid rows | 100% | 100% |
| Diversity score | **92** | 82 |
| Realism (AI judge) | **6/10** | 2/10 |
| Speed | **82 rows/min** | 5 rows/min |
| Cost | paid | **free** |

**In short:** the cloud model writes much more realistic data. The local model is free and private, but slower, and it sometimes labels rows wrongly. Wrong labels in test data are dangerous, because tests then pass or fail for the wrong reason.

### 💡 What measuring taught me

My first version scored well on numbers, but the AI judge gave it **3/10 for realism**. Looking at the data showed why: a whole batch was about the same thing, elderly customers asking about one kettle. I changed it so every row gets its own scenario:

**Realism 3 → 6 · Variety 2 → 6 · Diversity 84 → 92**

*You can only improve synthetic data if you measure it.*

## How it works

1. **Brainstorm.** The model first lists many different scenarios for your dataset.
2. **Generate.** Rows are written in batches of 10, each row for its own scenario. Some batches are edge cases.
3. **Check.** Every row is validated against your columns. Broken rows and duplicates are thrown out and replaced.
4. **Measure.** The quality report is calculated, and the AI judge reviews a sample.
5. **Download** the result as CSV or JSON.

## Quick start

```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add your OpenAI API key. Then:

```bash
python app.py
```

**Want free local models too?** Install [Ollama](https://ollama.com) and run `ollama pull llama3.2`. Every local model you install appears in the app automatically.

To run the tests (no API calls, no cost):

```bash
pytest
```

## Project files

| File | What's inside |
|---|---|
| `app.py` | The web app |
| `generator.py` | Talks to the models and builds the dataset |
| `quality.py` | The quality report |
| `judge.py` | The AI judge |
| `schema.py` | Columns and row validation |
| `presets.py` | The five examples |
| `tests/` | Automated tests |

## Ideas for later

- Catch rows that say the same thing in different words, not only reworded copies
- Choose target distributions, like "70% delivered, 10% returned"
- Send the generated chatbot messages straight into an automated chatbot test
