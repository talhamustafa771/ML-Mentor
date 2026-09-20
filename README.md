# ML Mentor

Ninety-one days from "what is a model" to building a neural network by hand.
Theory, a picture of the theory, and code that has been executed against real
data before you ever see it.

Built as the companion to the Python & DSA Mentor, and deliberately different
where machine learning is deliberately different.

---

## What makes it work

**Nothing is used before it is taught.** Every lesson is generated under a
knowledge horizon derived from the curriculum itself (`core/concepts.py`), and
then *checked against it*. A week-2 lesson cannot quietly cross-validate; a
week-4 lesson cannot regularise before overfitting has been seen. The Python
mentor learned this the expensive way — it opened day 1 with a function
definition taught on day 43 — so here the constraint is machine-enforced rather
than requested politely.

**Evaluation comes before models.** Weeks 1 and 2 cover the train/test split,
overfitting and the metrics before a single algorithm is fitted. Almost every
serious mistake in applied ML is an evaluation mistake, not an algorithm one.

**Every lesson's code is run before you see it.** Four gates, in
`core/verify.py`: shape, horizon, execution, and claims. That last one is the
quiet failure — a lesson that says "the AUC comes out around 0.71" because the
author expected 0.71 while the code printed 0.68. Every number the commentary
asserts is checked against what the code actually printed, and a lesson that
fails is sent back with the traceback attached.

**The maths is answerable.** Seventy-seven terms are written into the app with
a formula, a worked example using small numbers, and a sentence on what it
unlocks — instant, free, and impossible to hallucinate. Thirty-one symbols are
named, because most of the time the blocker is that nobody said what the
squiggle is called. Anything else goes to the model under a prompt that forbids
explaining one unfamiliar thing with another. Every worked example in the bank
is recomputed by the test suite.

**The data is bundled and its truth is known.** Two of the seven datasets are
synthetic, so the real relationship is written down: a model can be checked
against reality rather than merely scored. They contain, on purpose, missing
values that are not missing at random, a rare category, a leaking column, a
skewed target, genuine outliers and a redundant feature.

---

## Running it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open http://localhost:8501. Paste a free [Groq](https://console.groq.com/keys)
key into Settings to generate lessons. Everything else — the curriculum, the
datasets, the written Math Helper answers, the 37 challenges, the visualisers —
works without a key.

### On your phone, on the same Wi-Fi

```bash
streamlit run app.py --server.address 0.0.0.0
```

Then open `http://<your-laptop's-IP>:8501` on the phone. Not `0.0.0.0` — that
is an instruction to the laptop, not an address a phone can reach.

### Anywhere, on anything

Push to GitHub, deploy on [Streamlit Community Cloud](https://share.streamlit.io),
and point `TURSO_DATABASE_URL` / `TURSO_AUTH_TOKEN` at a free
[Turso](https://turso.tech) database so your progress survives the container
being recycled. See `.streamlit/secrets.toml.example`.

---

## Layout

```
app.py                  navigation; everything global happens once here
core/
  curriculum.py         91 days, 13 weeks, 263 hours
  concepts.py           the knowledge horizon — what may appear on which day
  mathpad.py            the Math Helper: 77 written terms, 31 symbols
  datasets.py           the seven bundled datasets
  challenges.py         37 curated challenges with hidden graders
  generators.py         writes lessons, questions, cards, challenges
  verify.py             the four gates a lesson passes before it is shown
  sandbox.py            runs notebooks; captures stdout and figures
  db.py / dbdriver.py   SQLite locally, libSQL when hosted
  scheduler.py          mastery, spacing, what to do next
  gamify.py             XP, levels, and badges that reward judgement
  ui.py                 the visual system
views/                  one file per page
datasets/               the CSVs, and the script that generates two of them
tests/                  run each directly: python tests/test_smoke.py
```

---

## The tests

```bash
python tests/test_smoke.py       # ~290 checks, fast
python tests/test_horizon.py     # ~540 checks, the knowledge horizon
python tests/test_mathpad.py     # ~2700 checks, every worked example recomputed
python tests/test_challenges.py  # ~440 checks, slow: every challenge really runs
```

`test_challenges.py` is the one worth understanding. It executes every
challenge's reference solution against that challenge's own hidden grader, and
then attempts every challenge with an empty answer to confirm the grader
actually rejects it. A challenge that cannot be solved, or that passes when you
do nothing, is worse than no challenge at all.

---

## The datasets

| Dataset | Rows | For |
|---|---|---|
| `customer_churn` | 4200 | missing-not-at-random, a rare category, target leakage, class balance |
| `house_prices` | 2600 | a skewed target, genuine outliers, a redundant feature, log scales |
| `breast_cancer` | 569 | thirty correlated features, scaling, precision against recall |
| `iris` | 150 | a first classifier, small enough to print whole |
| `wine` | 178 | features on wildly different scales, three classes |
| `diabetes` | 442 | an honest R-squared that is not close to 1 |
| `digits` | 1797 | images as vectors, PCA, clustering without labels |

Load any of them with:

```python
from core.datasets import load
df = load('customer_churn')
```

Nothing downloads anything. `read_csv`, `requests` and every `fetch_*` are
blocked in the sandbox on purpose: a lesson that downloads its data cannot be
verified and breaks the first time the container has no network.
