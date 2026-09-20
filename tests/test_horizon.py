"""The knowledge horizon, tested against the failure it exists to prevent.

The Python mentor shipped a day-1 lesson that defined a function taught on day
43. It was caught by a person reading it, three times, which is not a system.
This file is the system: every rule the horizon claims to enforce is checked
here against code written specifically to break it.

Run:  python tests/test_horizon.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import concepts, curriculum                    # noqa: E402

PASS, FAIL = [], []


def check(name: str, condition: bool, detail: str = "") -> None:
    (PASS if condition else FAIL).append(f"{name}{' - ' + detail if detail else ''}")


def flags(code: str, day: int) -> list[str]:
    return [v.symbol for v in concepts.check_code(code, day)]


# ---------------------------------------------------------------------------
# The module agrees with the curriculum
# ---------------------------------------------------------------------------
problems = concepts.validate()
check("nothing in the horizon contradicts the curriculum", not problems,
      "; ".join(problems[:5]))

check("every curriculum day is covered",
      {t["day"] for t in curriculum.TOPICS} == set(range(1, 92)))
check("concepts are derived from the curriculum, never duplicated",
      all(concepts.EVERYTHING[key].day ==
          min(t["day"] for t in curriculum.TOPICS
              if key in t["concepts"] or key in t["maths"])
          for key in list(concepts.EVERYTHING)[:80]))

# An idea must arrive exactly once. Two topics both claiming to introduce
# cross-validation means one of them is teaching it a second time as though it
# were new, which is the ordering bug this file exists to catch.
counts: dict[str, list[int]] = {}
for topic in curriculum.TOPICS:
    for key in topic["concepts"]:
        counts.setdefault(key, []).append(topic["day"])
repeats = {k: v for k, v in counts.items() if len(v) > 1}
check("no concept is introduced twice", not repeats,
      "; ".join(f"{k} on days {v}" for k, v in list(repeats.items())[:4]))


# ---------------------------------------------------------------------------
# The exact failure that prompted this: a lesson reaching forward
# ---------------------------------------------------------------------------
REACHING = """
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

pipe = Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression())])
print(cross_val_score(pipe, X, y, cv=5).mean())
print(classification_report(y_test, pipe.predict(X_test)))
"""
caught = flags(REACHING, day=5)
for symbol in ("StandardScaler", "Pipeline", "LogisticRegression",
               "cross_val_score", "classification_report"):
    check(f"day 5 rejects {symbol}", symbol in caught, f"caught {caught}")

check("day 5 accepts train_test_split, which day 5 teaches",
      "train_test_split" not in caught)
check("day 91 accepts all of it", not flags(REACHING, day=91))


# ---------------------------------------------------------------------------
# Honest lessons must pass untouched
# ---------------------------------------------------------------------------
HONEST = [
    (2, """
from core.datasets import load
df = load("customer_churn")
print(df.shape)
print(df.dtypes)
print(df.describe())
print(df.isna().sum())
"""),
    (3, """
import numpy as np
import matplotlib.pyplot as plt
from core.datasets import load
df = load("house_prices")
fig, ax = plt.subplots(1, 2, figsize=(9, 3))
ax[0].hist(df["price"], bins=40, alpha=0.8)
ax[1].hist(np.log(df["price"]), bins=40, alpha=0.8)
print(df["price"].skew(), np.log(df["price"]).skew())
"""),
    (5, """
from core.datasets import load
from sklearn.model_selection import train_test_split
df = load("customer_churn")
train, test = train_test_split(df, test_size=0.2, stratify=df["churned"],
                               random_state=7)
print(len(train), len(test), train["churned"].mean(), test["churned"].mean())
"""),
    (18, """
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
print(X_scaled.mean(axis=0).round(3), X_scaled.std(axis=0).round(3))
"""),
    (37, """
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=0)
forest.fit(X_train, y_train)
print(sorted(zip(forest.feature_importances_, X_train.columns), reverse=True))
"""),
    (46, """
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
coords = pca.fit_transform(X_scaled)
print(pca.explained_variance_ratio_)
"""),
]
for day, code in HONEST:
    found = flags(code, day)
    check(f"an honest day-{day} cell passes", not found, f"flagged {found}")


# ---------------------------------------------------------------------------
# The rules that hold on every day
# ---------------------------------------------------------------------------
for day in (1, 30, 91):
    check(f"day {day} refuses read_csv", "read_csv" in flags(
        "import pandas as pd\ndf = pd.read_csv('x.csv')", day))
    check(f"day {day} refuses a download", "fetch_openml" in flags(
        "from sklearn.datasets import fetch_openml\nd = fetch_openml('x')", day))
    check(f"day {day} refuses torch", "torch" in flags("import torch", day))
    check(f"day {day} refuses invented data", "make_classification" in flags(
        "from sklearn.datasets import make_classification\n"
        "X, y = make_classification(n_samples=100)", day))

check("the always-forbidden list explains itself",
      all(len(reason) > 20 for reason in concepts.FORBIDDEN_ALWAYS.values()))


# ---------------------------------------------------------------------------
# It must not cry wolf
# ---------------------------------------------------------------------------
INNOCENT = [
    (1, "print('hello')\nx = [1, 2, 3]\nprint(sum(x) / len(x))"),
    (1, "import numpy as np\nprint(np.array([1, 2, 3]).mean())"),
    # alpha= is transparency here, not regularisation
    (4, "import matplotlib.pyplot as plt\nplt.scatter(a, b, alpha=0.3)"),
    # a data pipeline in a sentence, not a sklearn Pipeline
    (2, "# the data pipeline at work feeds this table every night\nprint(df.head())"),
    (1, "total = 0\nfor row in rows:\n    total += row\nprint(total)"),
    (4, "print(df.groupby('contract')['churned'].mean())"),
    (2, "print(df['price'].quantile([0.25, 0.5, 0.75]))"),
]
for day, code in INNOCENT:
    found = flags(code, day)
    check(f"day {day} does not cry wolf", not found, f"flagged {found}")

check("syntactically broken code is not reported as reaching ahead",
      not concepts.check_code("def (((:", 40))


# ---------------------------------------------------------------------------
# Keyword arguments
# ---------------------------------------------------------------------------
check("cv= is caught before cross-validation is taught",
      "cv" in flags("cross_val_score(m, X, y, cv=5)", 7))
check("cv= passes after it is taught",
      "cv" not in flags("cross_val_score(m, X, y, cv=5)", 12))
check("random_state= is allowed from the split onwards",
      not flags("train_test_split(X, y, random_state=0)", 5))
check("max_depth= is caught before trees",
      "max_depth" in flags("DecisionTreeClassifier(max_depth=3)", 20))
check("class_weight= is caught before imbalance is taught",
      "class_weight" in flags("LogisticRegression(class_weight='balanced')", 30))


# ---------------------------------------------------------------------------
# Reaching ahead by hand, without naming a forbidden symbol
# ---------------------------------------------------------------------------
# The day-1 lesson this app actually produced imported nothing forbidden and
# still taught four ideas from later weeks: it shuffled the rows itself, sliced
# off a quarter, scored the result as "test accuracy", cut probabilities at
# 0.5, and counted the answers into a confusion matrix. The symbol scan saw
# nothing, because no symbol was used. So the commentary is read too.

HAND_ROLLED = [
    (1, "We create a random split, map the rates to the test rows, turn them "
        "into predictions with a 0.5 threshold, and print the proportion "
        "correct.", ("train_test_split", "threshold")),
    (1, "We build a table counting each combination of actual and predicted "
        "class.", ()),
    (1, "Test accuracy: 0.592", ("accuracy",)),
    (1, "The confusion matrix shows where it went wrong.", ("confusion_matrix",)),
    (2, "The intercept and the coefficients of a linear regression.",
     ("linear_model", "intercept", "coefficient")),
    (5, "We cross-validate over five folds.", ("cross_validation",)),
]
for day, text, expected in HAND_ROLLED:
    found = {key for key, _, _ in concepts.prose_reaches_ahead(text, day, 1)}
    for key in expected:
        check(f"day {day} catches hand-rolled {key}", key in found,
              f"found {sorted(found)}")

# And it must not cry wolf on a lesson describing its own day's material.
ON_TOPIC = [
    (2, "We load the data and print its shape, types and missing values."),
    (3, "The price column is right-skewed; logging it pulls the tail in."),
    (4, "Area and price rise together, so their correlation is positive."),
    (5, "We split into a training set and a test set, keeping the balance."),
    (6, "We fit a baseline and measure its accuracy on the held-out rows."),
    (8, "The model overfits: it scores better on training than on test."),
    (10, "Five-fold cross-validation gives five scores, not one."),
    (11, "The confusion matrix shows precision and recall differ."),
    (12, "Moving the threshold trades precision against recall."),
    (22, "The intercept is where the line starts; each coefficient is a slope."),
    (25, "The residuals fan out, so the variance is not constant."),
    (37, "A random forest averages many decision trees."),
    (46, "The first principal component is the direction of greatest spread."),
]
for day, text in ON_TOPIC:
    found = concepts.prose_reaches_ahead(text, day, 1)
    check(f"day {day} prose about its own material passes", not found,
          f"flagged {[k for k, _, _ in found]}")

# A whole future topic can be taught without any one word being repeated.
# This is the day-1 lesson that slipped through a per-word threshold: "linear
# equation" once, "intercept" once, "weight" twice — no term three times, and
# day 22's lesson delivered in full.
LINEAR_ON_DAY_ONE = (
    "One of the earliest ways to describe a relationship is a linear "
    "equation. w0 is the intercept, a constant that shifts the prediction. "
    "wi is the weight for feature xi. If the weight for tenure_months is "
    "negative, longer tenure reduces the predicted chance.")
_caught = {key for key, _, _ in
           concepts.prose_reaches_ahead(LINEAR_ON_DAY_ONE, 1, min_hits=3)}
check("a future topic taught under several names is caught",
      {"linear_model", "intercept", "coefficient"} <= _caught,
      f"caught {sorted(_caught)}")
check("the same words are fine on the day they belong to",
      not concepts.prose_reaches_ahead(LINEAR_ON_DAY_ONE, 22, min_hits=1))
check("and fine later still",
      not concepts.prose_reaches_ahead(LINEAR_ON_DAY_ONE, 57, min_hits=1))
check("'weights' does not become a day-57 idea in week 4",
      not concepts.prose_reaches_ahead("the weights of the model", 22, 1),
      "a regression lesson must be able to say 'weights'")

# Counting by topic must not make a single passing mention trip the wire.
for _text, _day in [
    ("Later you will measure accuracy against a baseline.", 1),
    ("We look at the shape, the types and the missing values.", 2),
    ("The price is right-skewed, so we take its log.", 3),
]:
    check(f"day {_day} passing mention stays quiet",
          not concepts.prose_reaches_ahead(_text, _day, min_hits=3),
          str(concepts.prose_reaches_ahead(_text, _day, 3)))

# Day 1 has no scikit-learn tool at all, which is exactly why a model asked
# for three code cells goes looking for something to fit. The horizon has to
# say what IS possible, not only what is not.
_day1_horizon = concepts.horizon(curriculum.TOPICS[0])
check("an early horizon says what the code can do",
      "WHAT THE CODE CAN DO TODAY" in _day1_horizon)
check("it admits there is no scikit-learn tool yet",
      "No scikit-learn tool is unlocked yet" in _day1_horizon)
check("a late horizon does not waste room on that",
      "WHAT THE CODE CAN DO TODAY" not in concepts.horizon(
          next(t for t in curriculum.TOPICS if t["day"] == 40)))
check("day 1 really has no tools", not concepts.allowed_api(1))


# A passing forward reference in the main theory is legitimate and must
# survive; teaching the thing is not.
signpost = "In three weeks you will cross-validate this properly."
check("one forward reference is allowed in the theory",
      not concepts.prose_reaches_ahead(signpost, 1, min_hits=3))
teaching = ("Cross-validation splits the data five ways. Each cross-validation "
            "fold is scored. We then average the cross-validation scores.")
check("teaching it three times over is not",
      any(k == "cross_validation"
          for k, _, _ in concepts.prose_reaches_ahead(teaching, 1, min_hits=3)))

# The prompt must say so as well, since detection alone only rejects after
# the tokens have been spent.
check("the horizon forbids hand-rolling explicitly",
      "route around this" in concepts.horizon(curriculum.TOPICS[0]))


# ---------------------------------------------------------------------------
# reaches_ahead, across the shapes a generated item takes
# ---------------------------------------------------------------------------
check("a code field is scanned",
      concepts.reaches_ahead({"code": "from sklearn.svm import SVC"}, 10))
check("a fenced block in prose is scanned",
      concepts.reaches_ahead(
          {"explanation": "Try this:\n```python\nfrom sklearn.svm import SVC\n```"},
          10))
check("steps are scanned",
      concepts.reaches_ahead(
          {"steps": [{"code": "m = SVC()"}, {"code": "print(1)"}]}, 10))
check("prose alone is never scanned",
      not concepts.reaches_ahead(
          {"explanation": "In week 5 you will meet the SVC class."}, 10))
check("a clean item passes",
      not concepts.reaches_ahead({"code": "print(df.shape)"}, 2))


# ---------------------------------------------------------------------------
# The prompt fragment
# ---------------------------------------------------------------------------
for topic in curriculum.TOPICS:
    text = concepts.horizon(topic)
    day = topic["day"]
    check(f"day {day} horizon states the day", f"day {day} of 91" in text)
    check(f"day {day} horizon bans downloads", "read_csv" in text)
    check(f"day {day} horizon bans torch", "torch" in text)
    check(f"day {day} horizon stays readable", len(text) < 9000,
          f"{len(text)} characters")
    check(f"day {day} horizon tells it to go deeper, not forward",
          "go deeper rather than forward" in text)

# The attractors are the whole reason the Python version failed. Whatever else
# is truncated, these must be named while they are still in the future.
for symbol in concepts.ATTRACTORS:
    arrives = concepts.day_for_symbol(symbol)
    if arrives > 1:
        text = concepts.horizon(next(t for t in curriculum.TOPICS if t["day"] == 1))
        check(f"day 1 names the attractor {symbol}", symbol in text)

late = concepts.horizon(next(t for t in curriculum.TOPICS if t["day"] == 80))
check("a late horizon summarises rather than listing 300 ideas",
      "everything in days 1 to 79" in late)
early = concepts.horizon(next(t for t in curriculum.TOPICS if t["day"] == 4))
check("an early horizon lists what is available outright",
      "TAUGHT SO FAR, and therefore available" in early)


# ---------------------------------------------------------------------------
# Timeline sanity: the orderings the course promises
# ---------------------------------------------------------------------------
def arrives(key: str) -> int:
    return concepts.introduced_on(key)


# (earlier, later, why, must_be_a_different_day)
ORDERINGS = [
    ("train_test_split", "fit", "you split before you fit", True),
    ("overfitting", "cross_validation", "overfitting motivates cross-validation", True),
    ("accuracy", "precision", "accuracy's failure motivates precision", True),
    ("confusion_matrix", "roc_curve", "the matrix comes before the curve", True),
    ("roc_curve", "auc", "the curve comes before its area", False),
    ("overfitting", "ridge", "overfitting comes before regularisation", True),
    ("linear_model", "logistic_regression", "linear before logistic", True),
    ("gradient_descent", "backpropagation", "descent before backprop", True),
    ("decision_tree", "random_forest", "a tree before a forest", True),
    ("random_forest", "gradient_boosting", "bagging before boosting", True),
    ("kmeans", "silhouette", "clusters before judging them", False),
    ("pca", "tsne", "linear projection before the nonlinear one", True),
    ("cross_validation", "grid_search", "you can score before you search", True),
    ("data_leakage", "target_leakage", "the idea before the special case", True),
    ("perceptron", "hidden_layer", "one neuron before a layer", True),
    ("sigmoid", "logistic_regression", "the squash before the model using it", False),
    ("standardisation", "knn", "scaling before a distance-based model", True),
    ("pipeline", "grid_search", "pipelines before searching over them", True),
]
for first, second, why, strict in ORDERINGS:
    a, b = arrives(first), arrives(second)
    ok = 0 < a < b if strict else 0 < a <= b
    check(f"ordering: {why}", ok, f"{first}={a}, {second}={b}")

# Evaluation before models, which is the curriculum's stated second principle.
first_model_day = min(concepts.day_for_symbol(s) for s in
                      ("LinearRegression", "LogisticRegression",
                       "DecisionTreeClassifier", "KNeighborsClassifier")
                      if concepts.day_for_symbol(s) > 0)
for metric in ("train_test_split", "overfitting", "cross_validation",
               "confusion_matrix", "auc", "r_squared"):
    check(f"{metric} is taught before the first real model",
          0 < arrives(metric) < first_model_day,
          f"{metric}={arrives(metric)}, first model day={first_model_day}")

check("every API symbol becomes available at some point",
      all(concepts.day_for_symbol(s) > 0 for s in concepts._API))
check("the last day allows everything", not concepts.forbidden_api(91))
check("day 1 allows almost nothing", len(concepts.allowed_api(1)) == 0,
      f"{concepts.allowed_api(1)}")


# ---------------------------------------------------------------------------
print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
for line in FAIL:
    print("  FAIL:", line)
sys.exit(1 if FAIL else 0)
