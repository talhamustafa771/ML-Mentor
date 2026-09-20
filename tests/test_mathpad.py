"""The Math Helper's written answers, checked rather than trusted.

A glossary that quietly contains a wrong number is worse than no glossary: the
reader has no way to catch it, and the whole point of the written bank is that
it cannot hallucinate. So every worked example carrying arithmetic is
recomputed here from the formula it claims to demonstrate, and the test fails
if the file and the maths disagree.

Run:  python tests/test_mathpad.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import concepts, curriculum, mathpad          # noqa: E402

PASS, FAIL = [], []


def check(name: str, condition: bool, detail: str = "") -> None:
    (PASS if condition else FAIL).append(f"{name}{' - ' + detail if detail else ''}")


def near(a: float, b: float, tol: float = 0.005) -> bool:
    return abs(a - b) <= tol


def contains(entry_key: str, *fragments: str) -> None:
    """The worked example must contain the numbers it claims."""
    entry = mathpad.BANK[entry_key]
    text = f"{entry.worked} {entry.plain} {entry.why}"
    for fragment in fragments:
        check(f"{entry_key} states {fragment!r}", fragment in text,
              f"not found in: {entry.worked[:70]}...")


# ---------------------------------------------------------------------------
# The arithmetic in the worked examples
# ---------------------------------------------------------------------------
xs = [2, 4, 9]
mean = sum(xs) / len(xs)
var = sum((x - mean) ** 2 for x in xs) / len(xs)

check("mean of 2,4,9 is 5", near(mean, 5.0))
contains("mean", "15", "5")

check("variance of 2,4,9 is 8.67", near(var, 8.6667, 0.001))
contains("variance", "26", "8.67")
check("sd of 2,4,9 is 2.94", near(math.sqrt(var), 2.9439, 0.001))
contains("standard_deviation", "26", "8.67", "2.94")

check("median of 2,4,9,100 is 6.5", near((4 + 9) / 2, 6.5))
check("mean of 2,4,9,100 is 28.75", near((2 + 4 + 9 + 100) / 4, 28.75))
contains("median", "6.5", "28.75")

check("log10(1000) is 3", near(math.log10(1000), 3.0))
check("ln(250000) is about 12.4", near(math.log(250_000), 12.43, 0.02))
contains("logarithm", "3", "12.4")
check("exp(12.4) is about 242,800", near(math.exp(12.4), 242_801, 500))
contains("exponential", "242,800")

errors = [2, -1, 3]
mae = sum(abs(e) for e in errors) / 3
mse = sum(e ** 2 for e in errors) / 3
check("MAE of +2,-1,+3 is 2", near(mae, 2.0))
contains("absolute_error", "6", "2")
check("MSE of +2,-1,+3 is 4.67", near(mse, 4.6667, 0.001))
contains("squared_error", "14", "4.67")
check("RMSE is 2.16", near(math.sqrt(mse), 2.1602, 0.001))
contains("rmse", "2.16")
check("RMSE exceeds MAE here", math.sqrt(mse) > mae)

y = [10, 12, 20]
pred = [12, 13, 17]
ybar = sum(y) / 3
ss_tot = sum((v - ybar) ** 2 for v in y)
ss_res = sum((a - b) ** 2 for a, b in zip(y, pred))
check("R2 worked example: SS_tot is 56", near(ss_tot, 56))
check("R2 worked example: SS_res is 14", near(ss_res, 14))
check("R2 worked example: R2 is 0.75", near(1 - ss_res / ss_tot, 0.75))
contains("r_squared_formula", "56", "14", "0.75")

precision = 8 / 10
recall = 8 / 20
f1 = 2 * precision * recall / (precision + recall)
check("precision 8/10 is 0.8", near(precision, 0.8))
check("recall 8/20 is 0.4", near(recall, 0.4))
check("F1 is 0.53", near(f1, 0.5333, 0.001))
check("F1 is below the plain average", f1 < (precision + recall) / 2)
contains("precision_formula", "0.8")
contains("recall_formula", "0.4")
contains("harmonic_mean", "0.53", "0.6")
check("FPR 2/100 is 0.02", near(2 / 100, 0.02))
contains("false_positive_rate", "0.02")

check("log loss at p=0.9 is 0.105", near(-math.log(0.9), 0.1054, 0.001))
check("log loss at p=0.1 is 2.303", near(-math.log(0.1), 2.3026, 0.001))
check("the confident-wrong penalty is >20x", -math.log(0.1) / -math.log(0.9) > 20)
contains("log_loss_formula", "0.105", "2.303")

check("Brier 0.9 vs 1 is 0.01", near((0.9 - 1) ** 2, 0.01))
check("Brier 0.9 vs 0 is 0.81", near((0.9 - 0) ** 2, 0.81))
contains("brier_formula", "0.01", "0.81")

posterior = (0.99 * 0.01) / (0.99 * 0.01 + 0.05 * 0.99)
check("Bayes worked example is 0.167", near(posterior, 0.1667, 0.001))
check("Bayes numerator is 0.0099", near(0.99 * 0.01, 0.0099, 1e-6))
check("Bayes denominator is 0.0594", near(0.99 * 0.01 + 0.05 * 0.99, 0.0594, 1e-6))
contains("bayes_theorem", "0.0099", "0.0594", "0.167")

check("conditional worked example is 0.45", near(90 / 200, 0.45))
contains("conditional_probability", "0.45")

check("z-score of 80 is 2", near((80 - 60) / 10, 2.0))
contains("z_score_formula", "2")

check("euclidean (0,0)-(3,4) is 5", near(math.hypot(3, 4), 5.0))
check("manhattan (0,0)-(3,4) is 7", near(3 + 4, 7.0))
check("manhattan is never shorter", 3 + 4 >= math.hypot(3, 4))
contains("euclidean_distance", "25", "5")
contains("manhattan_distance", "7", "5")

check("dot product is 32", sum(a * b for a, b in zip([1, 2, 3], [4, 5, 6])) == 32)
contains("dot_product", "32")

check("L2 of [3,-4] is 5", near(math.sqrt(9 + 16), 5.0))
check("L1 of [3,-4] is 7", near(abs(3) + abs(-4), 7.0))
contains("l2_norm", "25", "5")
contains("l1_norm", "7")

check("derivative of x^2 at 3 is 6", near(2 * 3, 6.0))
contains("derivative", "6")

check("odds at p=0.8 is 4", near(0.8 / 0.2, 4.0))
check("log-odds at p=0.8 is 1.39", near(math.log(4), 1.3863, 0.001))
check("log-odds at p=0.5 is 0", near(math.log(1), 0.0))
contains("odds", "4", "1.39")

sig = lambda z: 1 / (1 + math.exp(-z))                    # noqa: E731
check("sigmoid(0) is 0.5", near(sig(0), 0.5))
check("sigmoid(2) is 0.88", near(sig(2), 0.8808, 0.001))
check("sigmoid(-2) is 0.12", near(sig(-2), 0.1192, 0.001))
contains("sigmoid_formula", "0.5", "0.88", "0.12")

check("sigmoid slope at 0 is 0.25", near(sig(0) * (1 - sig(0)), 0.25))
check("sigmoid slope at 6 is 0.0025", near(sig(6) * (1 - sig(6)), 0.00247, 0.0002))
check("sigmoid(6) is 0.9975", near(sig(6), 0.99753, 0.0001))
contains("derivative_of_sigmoid", "0.25", "0.9975", "0.0025")

ent = lambda ps: -sum(p * math.log2(p) for p in ps if p > 0)   # noqa: E731
check("entropy 50/50 is 1 bit", near(ent([0.5, 0.5]), 1.0))
check("entropy 90/10 is 0.47", near(ent([0.9, 0.1]), 0.4690, 0.001))
check("log2(0.9) is -0.152", near(math.log2(0.9), -0.152, 0.001))
check("log2(0.1) is -3.322", near(math.log2(0.1), -3.3219, 0.001))
contains("entropy_formula", "0.47", "0.152", "3.322")

gini = lambda ps: 1 - sum(p * p for p in ps)                   # noqa: E731
check("gini 50/50 is 0.5", near(gini([0.5, 0.5]), 0.5))
check("gini 90/10 is 0.18", near(gini([0.9, 0.1]), 0.18))
contains("gini_formula", "0.5", "0.18", "0.81")

check("centroid of (1,1),(3,3),(5,2) is (3,2)",
      near((1 + 3 + 5) / 3, 3.0) and near((1 + 3 + 2) / 3, 2.0))
contains("centroid", "(3, 2)")

check("standard error 10/sqrt(100) is 1", near(10 / math.sqrt(100), 1.0))
contains("standard_error", "1")
check("CI 50 +/- 1.96 is 48.0 to 52.0",
      near(50 - 1.96, 48.04, 0.01) and near(50 + 1.96, 51.96, 0.01))
contains("confidence_interval", "48.0", "52.0")

check("bonferroni 0.05/5 is 0.01", near(0.05 / 5, 0.01))
check("family-wise error over 5 tests is 23%",
      near(1 - 0.95 ** 5, 0.2262, 0.001))
contains("bonferroni", "0.01", "23%")

check("idf of a word in all docs is 0", near(math.log(1000 / 1000), 0.0))
check("idf of a word in 100 of 1000 is 2.30", near(math.log(10), 2.3026, 0.001))
contains("inverse_document_frequency", "2.30", "0")

check("bootstrap leaves out about 37%", near((1 - 1 / 4200) ** 4200, 0.3679, 0.01))
contains("bootstrap_sampling", "37%")

check("linear worked example is 268,000",
      40_000 + 120 * 1500 + 8_000 * 6 == 268_000)
contains("linear_equation", "268,000", "180,000", "48,000")

check("expected value worked example is 46",
      near(0.3 * 200 - 0.7 * 20, 46.0))
contains("expected_value", "60", "14", "46")

check("MLE of 7 heads in 10 is 0.7", near(7 / 10, 0.7))
contains("maximum_likelihood", "0.7")

check("accuracy trap: 97 of 100 is 97%", near(97 / 100, 0.97))
contains("accuracy_definition", "97%")


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------
unique = {entry.key: entry for entry in mathpad.BANK.values()}

for key, entry in unique.items():
    check(f"{key} has plain English", len(entry.plain) > 40)
    check(f"{key} has a worked example", len(entry.worked) > 40)
    check(f"{key} says why it matters", len(entry.why) > 30)
    check(f"{key} has a spoken reading", len(entry.reads_as) > 15)
    check(f"{key} formula carries no dollar signs", "$" not in entry.formula)
    # The one rule the whole module exists to enforce.
    check(f"{key} does not brush the reader off",
          not any(phrase in (entry.plain + entry.why).lower() for phrase in
                  ("don't worry", "do not worry", "simply put", "it's easy",
                   "it is easy", "as you know", "obviously")))

check("every alias resolves", all(mathpad.lookup(alias) is not None
                                  for entry in unique.values()
                                  for alias in entry.also))
check("every title resolves", all(mathpad.lookup(entry.title) is not None
                                  for entry in unique.values()))
check("every key resolves to itself",
      all(mathpad.lookup(key).key == key for key in unique))

for symbol in mathpad.SYMBOLS:
    check(f"symbol {symbol.name} is named", len(symbol.name) > 1)
    check(f"symbol {symbol.name} says what it does", len(symbol.says) > 15)
    check(f"symbol {symbol.name} is findable by name",
          mathpad.symbol_for(symbol.name.split(" (")[0]) is not None)
    check(f"symbol {symbol.name} is findable by glyph",
          mathpad.symbol_for(symbol.glyph) is not None)


# ---------------------------------------------------------------------------
# Questions the way a person actually types them
# ---------------------------------------------------------------------------
TYPED = {
    "what is r squared": "R squared",
    "r squared formula": "R squared",
    "explain the gradient to me": "The gradient",
    "how does entropy work": "Entropy",
    "AUC": "AUC",
    "f1 score": "F1, and the harmonic mean",
    "y hat": "y-hat",
    "theta": "theta",
    "Σ": "sigma (capital)",
    "nabla": "nabla, or grad",
    "what does sigma mean": "Standard deviation",
    "l2 norm": "The L2 norm",
    "rbf": "A kernel",
    "the roc curve": "The ROC curve",
    "log odds": "Odds and log-odds",
    "bayes theorem": "Bayes' theorem",
}
for typed, expected in TYPED.items():
    got = mathpad.explain(typed, use_model=False)
    check(f"typed {typed!r} finds {expected!r}", got.title == expected,
          f"got {got.title!r}")

check("an unknown term does not invent an answer",
      not mathpad.explain("quzzlefrump", use_model=False).ok)

# A core word must not widen into a longer entry. "model" matched the alias
# "linear_model" and answered with a linear equation — which then went into
# day 1's prompt as "the mathematics this day touches", and produced a day-22
# equation on day 1 three lessons running.
for _word in ("model", "features", "target", "prediction", "data", "split",
              "test", "training", "label", "output"):
    check(f"'{_word}' does not resolve to a longer entry",
          mathpad.lookup(_word) is None,
          f"resolved to {mathpad.lookup(_word).title if mathpad.lookup(_word) else ''!r}")
    check(f"'{_word}' is not offered as a written term",
          mathpad.lookup_exact(_word) is None)

# The forgiving matcher must stay forgiving for a person typing a question.
for _typed, _want in [("l1", "The L1 norm"), ("rbf", "A kernel"),
                      ("what is r squared", "R squared"),
                      ("how does entropy work", "Entropy")]:
    check(f"typing {_typed!r} still finds {_want!r}",
          mathpad.lookup(_typed) is not None
          and mathpad.lookup(_typed).title == _want)

# What a day offers as buttons must be exact matches only.
for _topic in curriculum.TOPICS:
    for _key, _label, _banked in mathpad.offer_for(_topic):
        if _banked:
            check(f"day {_topic['day']} offers {_key} only on an exact match",
                  mathpad.lookup_exact(_key) is not None,
                  f"{_key} was offered as written but only matches loosely")

_day1 = curriculum.TOPICS[0]
check("day 1 offers no written maths term at all",
      not any(banked for _, _, banked in mathpad.offer_for(_day1)),
      str([(k, l) for k, l, b in mathpad.offer_for(_day1) if b]))


# ---------------------------------------------------------------------------
# Wiring to the curriculum
# ---------------------------------------------------------------------------
check("the bank covers most of the course's maths", mathpad.coverage()["share"] > 0.75,
      f"share is {mathpad.coverage()['share']}")

for topic in curriculum.TOPICS:
    offers = mathpad.offer_for(topic)
    keys = [key for key, _, _ in offers]
    check(f"day {topic['day']} offers no duplicates", len(keys) == len(set(keys)))
    titles = [title for _, title, _ in offers]
    check(f"day {topic['day']} offers no duplicate answers",
          len(titles) == len(set(titles)))

lessons = [t for t in curriculum.TOPICS if t["kind"] == "lesson"]
check("every lesson offers at least one term",
      all(mathpad.offer_for(t) for t in lessons))

# The ladder must only ever point backwards.
for topic in curriculum.TOPICS:
    for key in topic.get("concepts", ()):
        for step in mathpad.ladder(key, topic["day"]):
            check(f"{key}'s ladder step {step['key']} is earlier",
                  step["day"] < topic["day"],
                  f"day {step['day']} vs topic day {topic['day']}")

check("cache ids are stable", mathpad.cache_id("auc", 12) == mathpad.cache_id("AUC", 12))
check("cache ids separate by day", mathpad.cache_id("auc", 12) != mathpad.cache_id("auc", 13))

prompt = mathpad.prompt_for("kernel trick", lessons[40])
check("the prompt states the day", "day " in prompt)
check("the prompt lists prior maths", "already been shown" in prompt)
check("the system prompt forbids jargon-on-jargon",
      "Never explain one unfamiliar term with another" in mathpad.SYSTEM)
check("the system prompt demands a worked example",
      "worked example" in mathpad.SYSTEM)


# ---------------------------------------------------------------------------
print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
for line in FAIL:
    print("  FAIL:", line)
sys.exit(1 if FAIL else 0)
