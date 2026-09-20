"""The curated challenge bank.

The Python mentor's bank held 150 interview problems, each a function with
hidden test cases, and that shape works because "reverse a linked list" has one
right answer you can check by calling it.

Machine learning does not decompose that way. The skill is almost never "write
this function" and almost always "do the right thing to this data, and notice
what it tells you". There is no single correct program, only a correct thing to
have found out.

So a challenge here is a task against a bundled dataset. Your code runs, and
then a hidden check runs in the same namespace and asserts against whatever you
left behind. Two consequences follow, and both are the point.

The check can grade the *method*, not just the number. A model that leaves the
leaking column in scores 0.99 on the churn data and is worthless; a check that
only asserted `auc > 0.9` would congratulate you for the worst possible answer.
Several checks here assert an upper bound for exactly that reason, and their
failure messages say why.

And every assert carries a message that teaches. Failing one should tell you
what you did, not merely that you were wrong — the message is the lesson, and
it is often the only part of a challenge you will remember.

Each entry is a dict:

    id          stable, so re-seeding updates rather than duplicates
    skill       a concept slug from core/curriculum.py; it decides which day
                the challenge unlocks on
    difficulty  beginner | intermediate | advanced
    dataset     which bundled dataset it uses
    task        what to do, naming the variables the check looks for
    setup       code run before yours, so everyone starts from the same place
    check       the hidden grader: asserts only
    hints       two or three nudges, never the answer
    solution    a reference answer that passes the check

Every solution in this file is executed against its own check by
tests/test_challenges.py. A challenge whose reference answer fails its own
grader is unsolvable, and the reader would spend an hour proving it.
"""
from __future__ import annotations

from typing import Any

from . import concepts

CHALLENGES: list[dict[str, Any]] = []


def _c(cid: str, skill: str, difficulty: str, dataset: str, task: str,
       check: str, solution: str, hints: tuple[str, ...] = (),
       setup: str = "") -> None:
    setup, solution = setup.strip(), solution.strip()
    CHALLENGES.append({
        "id": f"ch-{cid}", "skill": skill, "difficulty": difficulty,
        "dataset": dataset, "task": task.strip(), "setup": setup,
        "check": check.strip(), "hints": list(hints),
        "solution": solution,
        # When this challenge becomes available. Not the day its skill is
        # taught, but the first day on which its own reference answer would be
        # legal — which is often later, because a task about cross-validation
        # still has to fit SOMETHING, and the thing it fits may arrive weeks
        # after the idea does. Reading this off the code means it cannot drift
        # when the curriculum moves, and it cannot be got wrong by an author
        # who forgot which week logistic regression lands in.
        "unlock_day": max(concepts.introduced_on(skill),
                          concepts.earliest_day_for(setup),
                          concepts.earliest_day_for(solution)),
    })


B, I, A = "beginner", "intermediate", "advanced"

LOAD_CHURN = "from core.datasets import load\ndf = load('customer_churn')"
LOAD_HOUSE = "from core.datasets import load\ndf = load('house_prices')"


# ---------------------------------------------------------------------------
# Week 1 — reading data before touching it
# ---------------------------------------------------------------------------

_c("shape", "dataframe", B, "customer_churn", """
Find the shape of the churn data. Leave the number of rows in `n_rows` and the
number of columns in `n_cols`.
""", setup=LOAD_CHURN, check="""
assert 'n_rows' in dir(), "Leave the row count in a variable called n_rows."
assert n_rows == 4200, f"There are 4200 rows, not {n_rows}."
assert n_cols == 10, f"There are 10 columns, not {n_cols}."
""", solution="""
n_rows, n_cols = df.shape
""", hints=("df.shape gives both numbers at once.",
            "It returns a tuple: (rows, columns)."))

_c("missing", "missing_values", B, "customer_churn", """
Which column has the most missing values? Leave its name in `worst_column` and
the number missing in `worst_count`.
""", setup=LOAD_CHURN, check="""
assert worst_column == 'total_charges', (
    f"You found '{worst_column}'. Check df.isna().sum() again — total_charges "
    "is missing far more often than anything else, and the reason matters: "
    "customers who joined recently have no billing total yet.")
assert worst_count > 200, f"That count ({worst_count}) looks too low."
""", solution="""
counts = df.isna().sum()
worst_column = counts.idxmax()
worst_count = int(counts.max())
""", hints=("df.isna() gives a frame of True and False.",
            "Summing it counts the Trues, one number per column.",
            "idxmax() gives the label of the largest value, not the value."))

_c("mnar", "mnar", I, "customer_churn", """
The missing total_charges are not missing at random. Show it: compare the mean
tenure of rows where total_charges is missing against rows where it is present.
Leave them in `tenure_missing` and `tenure_present`.
""", setup=LOAD_CHURN, check="""
assert tenure_missing < tenure_present, (
    "The customers with no billing total should have SHORTER tenure — that is "
    "what makes this 'not missing at random'. Check which group you assigned "
    "to which variable.")
assert tenure_present - tenure_missing > 5, (
    "The gap should be large. Did you filter on the right column?")
""", solution="""
missing = df['total_charges'].isna()
tenure_missing = df.loc[missing, 'tenure_months'].mean()
tenure_present = df.loc[~missing, 'tenure_months'].mean()
""", hints=("Build a boolean mask with .isna() on the one column.",
            "~mask flips a boolean mask.",
            "df.loc[mask, 'column'] selects the rows the mask is True for."))

_c("impossible", "outlier", B, "customer_churn", """
Some rows contain values that cannot be real: a negative age, or a monthly
charge of 9999. Count them. Leave the number of impossible ages in `bad_ages`
and the number of impossible charges in `bad_charges`.
""", setup=LOAD_CHURN, check="""
assert bad_ages == 14, f"There are 14 rows with an impossible age, not {bad_ages}."
assert bad_charges == 9, (
    f"There are 9 rows charged 9999, not {bad_charges}. Note that these are "
    "not outliers to be trimmed — they are placeholder values standing in for "
    "missing data, and averaging them in would be wrong.")
""", solution="""
bad_ages = int((df['age'] < 0).sum())
bad_charges = int((df['monthly_charges'] == 9999).sum())
""", hints=("A comparison on a column gives a boolean column.",
            "Summing booleans counts the Trues."))

_c("skew", "skew", B, "house_prices", """
House prices are right-skewed. Measure it, then measure the skew of the logged
price. Leave them in `skew_raw` and `skew_logged`.
""", setup=LOAD_HOUSE, check="""
assert skew_raw > 1.5, (
    f"The raw skew should be well above 1.5, you got {skew_raw:.2f}.")
assert abs(skew_logged) < 0.5, (
    f"The logged skew should be close to zero, you got {skew_logged:.2f}. "
    "That collapse is the whole reason for logging a price.")
""", solution="""
import numpy as np
skew_raw = df['price'].skew()
skew_logged = np.log(df['price']).skew()
""", hints=("A pandas Series has a .skew() method.",
            "np.log applies to the whole column at once."))

_c("correlation", "correlation", B, "house_prices", """
Which numeric column correlates most strongly with price, ignoring price
itself? Leave the column name in `strongest` and the correlation in `r`.

Before you run it, write down your guess. Most people say area.
""", setup=LOAD_HOUSE, check="""
assert strongest == 'quality_score', (
    f"You found '{strongest}'. It is quality_score, at about 0.61, ahead of "
    "area at about 0.46 — and the reason is worth more than the answer. "
    "Correlation only measures a STRAIGHT-LINE relationship. Area drives "
    "price multiplicatively here, so on the raw price its relationship curves "
    "and a straight-line measure understates it. Log the price and area's "
    "correlation behaves quite differently. A low r never means 'no "
    "relationship'; it means 'no straight line'.")
assert r > 0.5, f"That correlation ({r:.2f}) looks too weak."
""", solution="""
correlations = df.corr(numeric_only=True)['price'].drop('price')
strongest = correlations.abs().idxmax()
r = correlations[strongest]
""", hints=("df.corr(numeric_only=True) gives the whole matrix.",
            "Take the 'price' column of it, then drop 'price' itself.",
            "Use .abs() before idxmax so a strong negative would also win."))


# ---------------------------------------------------------------------------
# Week 2 — evaluation, before any model
# ---------------------------------------------------------------------------

_c("split-stratify", "stratify", B, "customer_churn", """
Split the churn data 75/25, keeping the proportion of churners the same in both
halves. Leave the churn rate of each half in `rate_train` and `rate_test`.
""", setup=LOAD_CHURN, check="""
assert abs(rate_train - rate_test) < 0.02, (
    f"The two rates differ by {abs(rate_train - rate_test):.3f}. That is more "
    "than stratifying should allow — did you pass stratify=?")
assert 0.4 < rate_train < 0.52, (
    f"A churn rate of {rate_train:.2f} is not what this data has. Check you "
    "measured the churned column.")
""", solution="""
from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size=0.25, stratify=df['churned'],
                               random_state=0)
rate_train = train['churned'].mean()
rate_test = test['churned'].mean()
""", hints=("train_test_split can split a whole DataFrame, not just X and y.",
            "stratify takes the column whose balance you want preserved."))

_c("baseline", "baseline", B, "customer_churn", """
Before fitting anything, find the accuracy of always predicting the majority
class. Leave it in `baseline_accuracy`. Any model that cannot beat this has
learned nothing.
""", setup=LOAD_CHURN, check="""
assert 0.5 <= baseline_accuracy <= 0.56, (
    f"You got {baseline_accuracy:.3f}. The majority class here is 'did not "
    "churn' at about 54 per cent — that is the number to beat.")
""", solution="""
rate = df['churned'].mean()
baseline_accuracy = max(rate, 1 - rate)
""", hints=("The majority class is whichever is more common.",
            "Its share is its accuracy if you always guess it."))

_c("confusion", "confusion_matrix", I, "customer_churn", """
A model flags 900 customers. 500 of them really churned; 1432 churners were
missed entirely; 1368 non-churners were correctly left alone. Compute precision
and recall by hand from those four numbers, into `precision` and `recall`.
No model needed — this is about reading a confusion matrix.
""", check="""
assert abs(precision - 500/900) < 0.01, (
    f"Precision is TP/(TP+FP) = 500/900 = {500/900:.3f}, you got {precision:.3f}.")
assert abs(recall - 500/1932) < 0.01, (
    f"Recall is TP/(TP+FN) = 500/(500+1432) = {500/1932:.3f}, you got "
    f"{recall:.3f}. Note how much lower it is than precision here — the model "
    "is cautious, and misses most of what it is looking for.")
""", solution="""
tp, fp, fn, tn = 500, 400, 1432, 1368
precision = tp / (tp + fp)
recall = tp / (tp + fn)
""", hints=("900 flagged, 500 of them right, so 400 were wrong.",
            "Precision looks along the flagged; recall looks along the real."))

_c("cv-spread", "cross_validation", I, "customer_churn", """
Cross-validate a simple model over 5 folds and look at the spread, not just the
average. Leave the mean score in `cv_mean` and the standard deviation in
`cv_std`. Use only tenure_months and support_calls as features.
""", setup=LOAD_CHURN, check="""
assert 0.55 < cv_mean < 0.80, (
    f"A mean of {cv_mean:.3f} is outside what these two features can do.")
assert cv_std < 0.05, (
    f"A standard deviation of {cv_std:.3f} across folds is very high. With "
    "4200 rows the folds should agree closely; a big spread usually means the "
    "folds were not stratified.")
assert cv_std > 0, "The folds cannot all have scored identically."
""", solution="""
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
X = df[['tenure_months', 'support_calls']]
y = df['churned']
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
scores = cross_val_score(LogisticRegression(max_iter=1000), X, y, cv=folds)
cv_mean, cv_std = scores.mean(), scores.std()
""", hints=("cross_val_score returns one score per fold.",
            "Pass a StratifiedKFold so each fold has the same class balance.",
            "The array has .mean() and .std()."))


# ---------------------------------------------------------------------------
# Week 3 — cleaning, encoding, pipelines
# ---------------------------------------------------------------------------

_c("clean-impossible", "imputation", I, "customer_churn", """
Clean the impossible values properly: turn the negative ages and the 9999
charges into missing values, then impute all missing numeric columns with the
median. Leave the cleaned frame in `clean` and the number of remaining missing
numeric values in `still_missing`.
""", setup=LOAD_CHURN, check="""
assert still_missing == 0, f"{still_missing} missing values remain."
assert clean['age'].min() >= 0, "There is still a negative age."
assert clean['monthly_charges'].max() < 500, (
    "A charge of 9999 survived. Replacing it with NaN before imputing is the "
    "point: leaving it in drags the median and every later mean with it.")
assert len(clean) == 4200, (
    "You dropped rows. Imputing keeps them — and dropping the short-tenure "
    "customers would bias the sample, because that is exactly who is missing.")
""", solution="""
import numpy as np
clean = df.copy()
clean.loc[clean['age'] < 0, 'age'] = np.nan
clean.loc[clean['monthly_charges'] == 9999, 'monthly_charges'] = np.nan
numeric = clean.select_dtypes('number').columns
clean[numeric] = clean[numeric].fillna(clean[numeric].median())
still_missing = int(clean[numeric].isna().sum().sum())
""", hints=("Mark the impossible values as NaN first, with np.nan.",
            "select_dtypes('number') gives the numeric columns.",
            "fillna accepts a Series of per-column values, like .median()."))

_c("rare-category", "unseen_category", I, "customer_churn", """
The internet_service column has a rare level. Find it, and count how many rows
have it. Leave the level name in `rare_level` and the count in `rare_count`.
""", setup=LOAD_CHURN, check="""
assert rare_level == 'satellite', (
    f"You found '{rare_level}'. 'satellite' is the rare one, and it matters "
    "because a random split can put all of it on one side — which is how an "
    "encoder meets a category at test time it never saw while fitting.")
assert 50 < rare_count < 130, f"A count of {rare_count} is not right."
""", solution="""
counts = df['internet_service'].value_counts()
rare_level = counts.idxmin()
rare_count = int(counts.min())
""", hints=("value_counts() orders the levels by frequency.",
            "The last one is the rarest."))

_c("leak-in-scaling", "preprocessing_leak", A, "customer_churn", """
Show why scaling before the split leaks. Fit a StandardScaler on ALL the data,
then on the training half only, and compare what each thinks the mean of
tenure_months is. Leave them in `mean_all` and `mean_train`.
""", setup=LOAD_CHURN, check="""
assert mean_all != mean_train, (
    "The two should differ. If they are identical you fitted the same scaler "
    "twice — the second must see only the training rows.")
assert abs(mean_all - mean_train) < 3, (
    "That gap is implausibly large; check you are comparing the same column.")
""", solution="""
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
data = df[['tenure_months']].fillna(0)
train, test = train_test_split(data, test_size=0.25, random_state=0)
mean_all = StandardScaler().fit(data).mean_[0]
mean_train = StandardScaler().fit(train).mean_[0]
""", hints=("A fitted StandardScaler exposes .mean_, an array.",
            "Fit one on everything and one on the training half only.",
            "The difference is information about the test set, which is "
            "exactly what must not reach the model."))

_c("pipeline-build", "pipeline", I, "customer_churn", """
Build a Pipeline that imputes missing numeric values with the median, scales
them, and fits a logistic regression. Fit it on a training split and leave the
test accuracy in `accuracy`. Use tenure_months, monthly_charges and
support_calls only.
""", setup=LOAD_CHURN, check="""
assert 0.55 < accuracy < 0.80, (
    f"An accuracy of {accuracy:.3f} is outside the plausible range for these "
    "three features. Anything above 0.8 means something leaked.")
assert 'pipe' in dir() or 'pipeline' in dir(), (
    "Leave the fitted pipeline in a variable called pipe, so the check can "
    "confirm the preprocessing is inside it rather than applied beforehand.")
_p = pipe if 'pipe' in dir() else pipeline
assert hasattr(_p, 'steps'), "pipe should be a Pipeline."
assert len(_p.steps) >= 2, (
    "The pipeline needs at least the preprocessing and the model. Doing the "
    "imputation outside it is what leaks on the next dataset.")
""", solution="""
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

X = df[['tenure_months', 'monthly_charges', 'support_calls']]
y = df['churned']
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25,
                                          stratify=y, random_state=0)
pipe = Pipeline([
    ('impute', SimpleImputer(strategy='median')),
    ('scale', StandardScaler()),
    ('model', LogisticRegression(max_iter=1000)),
])
pipe.fit(X_tr, y_tr)
accuracy = pipe.score(X_te, y_te)
""", hints=("Pipeline takes a list of (name, step) pairs.",
            "Every step except the last must have a transform method.",
            "pipe.score does predict and compare in one call."))


# ---------------------------------------------------------------------------
# Week 4 — regression
# ---------------------------------------------------------------------------

_c("fit-line", "linear_model", B, "house_prices", """
Fit a straight line predicting price from area_sqft alone. Leave the slope in
`slope` and the intercept in `intercept`, and say what one extra square foot is
worth by leaving that in `per_sqft`.
""", setup=LOAD_HOUSE, check="""
assert slope > 0, "Bigger houses cost more; the slope should be positive."
assert abs(per_sqft - slope) < 1e-6, (
    "The value of one extra square foot IS the slope. That is what a "
    "coefficient means: the change in the target per one-unit change in the "
    "feature, holding everything else fixed.")
assert 20 < slope < 400, f"A slope of {slope:.1f} per square foot is not right."
""", solution="""
from sklearn.linear_model import LinearRegression
X = df[['area_sqft']]
y = df['price']
model = LinearRegression().fit(X, y)
slope = model.coef_[0]
intercept = model.intercept_
per_sqft = slope
""", hints=("X must be two-dimensional: df[['area_sqft']], not df['area_sqft'].",
            "A fitted LinearRegression has .coef_ and .intercept_."))

_c("log-target", "log_transform", I, "house_prices", """
Fit the same model twice: once on price, once on log(price). Compare the test
R-squared each achieves ON THE LOG SCALE, so the comparison is fair. Leave them
in `r2_raw` and `r2_logged`.
""", setup=LOAD_HOUSE, check="""
assert r2_logged > r2_raw, (
    f"Logging should help here: you got {r2_raw:.3f} raw and {r2_logged:.3f} "
    "logged. The relationship is multiplicative, which a straight line on the "
    "raw price cannot follow.")
assert r2_logged > 0.6, f"An R-squared of {r2_logged:.3f} is lower than expected."
""", solution="""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

X = df[['area_sqft', 'quality_score', 'age_years']].fillna(df.median(numeric_only=True))
y = df['price']
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0)

raw = LinearRegression().fit(X_tr, y_tr)
r2_raw = r2_score(np.log(y_te), np.log(raw.predict(X_te).clip(1)))

logged = LinearRegression().fit(X_tr, np.log(y_tr))
r2_logged = r2_score(np.log(y_te), logged.predict(X_te))
""", hints=("Fit the second model on np.log(y_train).",
            "To compare fairly, score both against np.log(y_test).",
            "The raw model's predictions need logging before comparison."))

_c("redundant-feature", "multicollinearity", I, "house_prices", """
Bedrooms should add nothing once area is known — a house has more bedrooms
because it is bigger. Show it properly: drop the eleven enormous houses above
10,000 sq ft, predict the LOGGED price from area, quality and age, then add
bedrooms and compare test R-squared. Leave them in `r2_without` and `r2_with`.

(Try it without dropping the giants first, and see what happens. That
difference is the second lesson here.)
""", setup=LOAD_HOUSE, check="""
assert r2_without > 0.6, (
    f"An R-squared of {r2_without:.3f} is too low — check you logged the "
    "price and kept quality and age as features.")
assert abs(r2_with - r2_without) < 0.005, (
    f"Adding bedrooms changed R-squared by {abs(r2_with - r2_without):.4f}. "
    "Once the giant houses are gone it should not move at all. If yours did, "
    "you left them in — and then bedrooms LOOKS useful, because being capped "
    "at eight is quietly telling the model which houses are enormous. A "
    "feature that only helps through the outliers is not a feature, it is an "
    "outlier detector in disguise.")
""", solution="""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = df.fillna(df.median(numeric_only=True))
typical = data[data['area_sqft'] < 10000]
y = np.log(typical['price'])
base = ['area_sqft', 'quality_score', 'age_years']

def r2_for(columns):
    X_tr, X_te, y_tr, y_te = train_test_split(typical[columns], y,
                                              test_size=0.25, random_state=0)
    return LinearRegression().fit(X_tr, y_tr).score(X_te, y_te)

r2_without = r2_for(base)
r2_with = r2_for(base + ['bedrooms'])
""", hints=("Filter with data[data['area_sqft'] < 10000] before anything else.",
            "Use the same random_state for both splits so the rows match.",
            ".score on a regressor returns R-squared."))

_c("ridge-shrink", "ridge", I, "house_prices", """
Show what ridge does to coefficients. Fit an ordinary linear regression and a
ridge with alpha=1000 on the same scaled features, and compare the total size
of the coefficients. Leave them in `size_plain` and `size_ridge`.
""", setup=LOAD_HOUSE, check="""
assert size_ridge < size_plain, (
    "Ridge should SHRINK the coefficients — that is what the penalty does. "
    f"You got {size_ridge:.2f} against {size_plain:.2f}.")
assert size_ridge > 0, (
    "Ridge shrinks towards zero but never reaches it. A total of exactly zero "
    "means something else went wrong — lasso is the one that zeroes features.")
""", solution="""
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler

X = df[['area_sqft', 'bedrooms', 'quality_score', 'age_years']].fillna(
    df.median(numeric_only=True))
X = StandardScaler().fit_transform(X)
y = np.log(df['price'])

size_plain = float(np.abs(LinearRegression().fit(X, y).coef_).sum())
size_ridge = float(np.abs(Ridge(alpha=1000).fit(X, y).coef_).sum())
""", hints=("Scale first, or the penalty falls unevenly across features.",
            "np.abs(coefs).sum() is the total size."))


# ---------------------------------------------------------------------------
# Week 5 — classification
# ---------------------------------------------------------------------------

_c("leakage-auc", "target_leakage", A, "customer_churn", """
The last_call_outcome column is recorded AFTER the customer decided. Fit two
models, one with it and one without, and leave their test AUCs in `auc_leaked`
and `auc_honest`. Then set `is_usable` to the one you would actually ship.
""", setup=LOAD_CHURN, check="""
assert auc_leaked > 0.95, (
    f"The leaking model should score near-perfectly: you got {auc_leaked:.3f}. "
    "Check that last_call_outcome is encoded and included.")
assert 0.6 < auc_honest < 0.85, (
    f"The honest model should land well short of perfect: you got "
    f"{auc_honest:.3f}.")
assert abs(is_usable - auc_honest) < 1e-9, (
    "The usable model is the HONEST one. The leaked score is unreachable in "
    "production, because at the moment you need a prediction the last call "
    "has not happened yet. A score you cannot reproduce in production is not "
    "a score.")
""", solution="""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

base = df[['tenure_months', 'monthly_charges', 'support_calls']].fillna(0)
y = df['churned']
leaked = pd.concat([base, pd.get_dummies(df['last_call_outcome'])], axis=1)

def auc_for(X):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25,
                                              stratify=y, random_state=0)
    model = LogisticRegression(max_iter=2000).fit(X_tr, y_tr)
    return roc_auc_score(y_te, model.predict_proba(X_te)[:, 1])

auc_honest = auc_for(base)
auc_leaked = auc_for(leaked)
is_usable = auc_honest
""", hints=("pd.get_dummies turns a text column into indicator columns.",
            "roc_auc_score needs probabilities: predict_proba(X)[:, 1].",
            "Ask yourself when each column becomes known, relative to the "
            "moment you need the prediction."))

_c("threshold-move", "threshold", I, "customer_churn", """
Fit a classifier, then find the threshold that maximises F1 on the test set.
Leave it in `best_threshold` and the F1 it achieves in `best_f1`. Also record
the F1 at the default 0.5 in `f1_at_default`.
""", setup=LOAD_CHURN, check="""
assert 0.05 < best_threshold < 0.95, (
    f"A threshold of {best_threshold} is not a probability cut-off.")
assert best_f1 >= f1_at_default - 1e-9, (
    "The best threshold cannot score worse than the default — 0.5 is one of "
    "the candidates you searched.")
assert best_f1 > 0.4, f"An F1 of {best_f1:.3f} is lower than expected."
""", solution="""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

X = df[['tenure_months', 'monthly_charges', 'support_calls']].fillna(0)
y = df['churned']
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25,
                                          stratify=y, random_state=0)
model = LogisticRegression(max_iter=2000).fit(X_tr, y_tr)
probabilities = model.predict_proba(X_te)[:, 1]

f1_at_default = f1_score(y_te, probabilities >= 0.5)
candidates = np.arange(0.05, 0.96, 0.01)
scores = [f1_score(y_te, probabilities >= t) for t in candidates]
best_index = int(np.argmax(scores))
best_threshold = float(candidates[best_index])
best_f1 = float(scores[best_index])
""", hints=("probabilities >= t gives a boolean prediction array.",
            "Loop over candidate thresholds and score each.",
            "np.argmax gives the position of the best, not the best value."))

_c("scaling-knn", "distance_metric", I, "breast_cancer", """
k-NN is a distance method, so scale matters. Fit a 5-neighbour classifier with
and without standardising the features, and leave the test accuracies in
`accuracy_raw` and `accuracy_scaled`.
""", setup="from core.datasets import load\ndf = load('breast_cancer')",
   check="""
assert accuracy_scaled > accuracy_raw, (
    f"Scaling should help clearly here: you got {accuracy_raw:.3f} raw and "
    f"{accuracy_scaled:.3f} scaled. These features span several orders of "
    "magnitude, so unscaled the largest one decides every distance by itself.")
assert accuracy_scaled > 0.9, f"{accuracy_scaled:.3f} is lower than expected."
""", solution="""
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

y = df['diagnosis']
X = df.drop(columns=['diagnosis']).select_dtypes('number')
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25,
                                          stratify=y, random_state=0)
accuracy_raw = KNeighborsClassifier(5).fit(X_tr, y_tr).score(X_te, y_te)
accuracy_scaled = Pipeline([
    ('scale', StandardScaler()),
    ('knn', KNeighborsClassifier(5)),
]).fit(X_tr, y_tr).score(X_te, y_te)
""", hints=("Put the scaler in a Pipeline so it is fitted on training rows only.",
            "drop(columns=...) removes the target from the features."))


# ---------------------------------------------------------------------------
# Week 6 — trees and ensembles
# ---------------------------------------------------------------------------

_c("overfit-tree", "max_depth", I, "customer_churn", """
Grow a decision tree with no depth limit and one limited to depth 4. Leave
their TRAINING accuracies in `train_deep` and `train_shallow`, and their TEST
accuracies in `test_deep` and `test_shallow`.
""", setup=LOAD_CHURN, check="""
assert train_deep > train_shallow, (
    "The unlimited tree should fit the training data better — it can memorise "
    "it.")
assert train_deep > 0.95, (
    f"An unlimited tree should reach near-perfect training accuracy, you got "
    f"{train_deep:.3f}.")
assert test_deep < train_deep - 0.15, (
    "The gap between training and test on the deep tree is what overfitting "
    "looks like. Yours is smaller than expected — check you scored the "
    "training set with the deep tree.")
""", solution="""
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

X = df[['tenure_months', 'monthly_charges', 'support_calls', 'age']].fillna(0)
y = df['churned']
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25,
                                          stratify=y, random_state=0)
deep = DecisionTreeClassifier(random_state=0).fit(X_tr, y_tr)
shallow = DecisionTreeClassifier(max_depth=4, random_state=0).fit(X_tr, y_tr)
train_deep, test_deep = deep.score(X_tr, y_tr), deep.score(X_te, y_te)
train_shallow, test_shallow = shallow.score(X_tr, y_tr), shallow.score(X_te, y_te)
""", hints=("A DecisionTreeClassifier with no max_depth grows until pure.",
            "Score each model on both halves."))

_c("importance-noise", "feature_importance", A, "customer_churn", """
Age has NO real effect on churn in this data — it was generated with a
coefficient of exactly zero. Fit a random forest on tenure_months,
support_calls and age, and leave age's importance in `age_importance` and the
name of the highest-ranked feature in `most_important`.

Predict what you will see before you run it.
""", setup=LOAD_CHURN, check="""
assert age_importance > 0.2, (
    f"Age scores {age_importance:.3f}. It should come out HIGH — around 0.5 — "
    "despite having no effect whatsoever. Check you fitted on all three "
    "columns.")
assert most_important == 'age', (
    f"You found '{most_important}'. It is age, and that is the entire point "
    "of this challenge. A pure-noise column ranks FIRST. Impurity importance "
    "rewards a feature for being splittable, and age takes seventy distinct "
    "values while support_calls takes about eight — so age offers the tree "
    "far more chances to carve out a lucky-looking subgroup. This is why "
    ".feature_importances_ must never be read as 'what matters', and why "
    "permutation importance, on day 41, exists.")
""", solution="""
from sklearn.ensemble import RandomForestClassifier
X = df[['tenure_months', 'support_calls', 'age']].fillna(0)
y = df['churned']
forest = RandomForestClassifier(n_estimators=200, random_state=0).fit(X, y)
importances = dict(zip(X.columns, forest.feature_importances_))
age_importance = importances['age']
most_important = max(importances, key=importances.get)
""", hints=("A fitted forest exposes .feature_importances_, in column order.",
            "zip the columns with it to get a mapping."))

_c("forest-vs-tree", "random_forest", I, "customer_churn", """
Compare one unlimited tree against a forest of 200 of them, on the same split.
Leave the test accuracies in `tree_accuracy` and `forest_accuracy`.
""", setup=LOAD_CHURN, check="""
assert forest_accuracy > tree_accuracy, (
    f"The forest should beat the single tree: you got {forest_accuracy:.3f} "
    f"against {tree_accuracy:.3f}. Averaging many high-variance trees is what "
    "removes the variance — that is the whole idea of bagging.")
""", solution="""
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

X = df[['tenure_months', 'monthly_charges', 'support_calls', 'age']].fillna(0)
y = df['churned']
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25,
                                          stratify=y, random_state=0)
tree_accuracy = DecisionTreeClassifier(random_state=0).fit(
    X_tr, y_tr).score(X_te, y_te)
forest_accuracy = RandomForestClassifier(
    n_estimators=200, random_state=0).fit(X_tr, y_tr).score(X_te, y_te)
""", hints=("Use the same split for both, or the comparison means nothing.",
            "n_estimators is how many trees the forest grows."))

_c("grid-search", "grid_search", I, "customer_churn", """
Search over max_depth for a decision tree using 5-fold cross-validation on the
TRAINING data only. Leave the chosen depth in `best_depth` and the test
accuracy of the refitted model in `final_accuracy`.
""", setup=LOAD_CHURN, check="""
assert isinstance(best_depth, (int, type(None))), (
    "best_depth should be the depth itself, from best_params_.")
assert 0.55 < final_accuracy < 0.80, (
    f"A final accuracy of {final_accuracy:.3f} is outside the plausible range.")
assert final_accuracy < 0.95, (
    "Anything near-perfect means the search saw the test set. The grid must "
    "cross-validate within the training half only, and the test half must be "
    "touched exactly once, at the end.")
""", solution="""
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier

X = df[['tenure_months', 'monthly_charges', 'support_calls']].fillna(0)
y = df['churned']
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25,
                                          stratify=y, random_state=0)
search = GridSearchCV(DecisionTreeClassifier(random_state=0),
                      {'max_depth': [2, 3, 4, 6, 8, 12]}, cv=5)
search.fit(X_tr, y_tr)
best_depth = search.best_params_['max_depth']
final_accuracy = search.score(X_te, y_te)
""", hints=("GridSearchCV takes an estimator and a dict of parameter lists.",
            "It refits the best model on all the training data by default.",
            "best_params_ is a dict; take the value you searched over."))


# ---------------------------------------------------------------------------
# Week 7 — unsupervised
# ---------------------------------------------------------------------------

_c("elbow", "elbow_method", I, "customer_churn", """
Run k-means for k from 2 to 8 on the scaled numeric churn features and collect
the inertia for each. Leave the list in `inertias`, in order.
""", setup=LOAD_CHURN, check="""
assert len(inertias) == 7, f"Expected 7 values for k=2..8, got {len(inertias)}."
assert all(inertias[i] >= inertias[i + 1] for i in range(len(inertias) - 1)), (
    "Inertia must fall as k rises — more clusters always fit tighter. That is "
    "exactly why you cannot choose k by minimising it, and why the elbow is "
    "what you look for instead.")
""", solution="""
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

X = df[['tenure_months', 'monthly_charges', 'support_calls', 'age']].fillna(0)
X = StandardScaler().fit_transform(X)
inertias = [KMeans(n_clusters=k, n_init=10, random_state=0).fit(X).inertia_
            for k in range(2, 9)]
""", hints=("A fitted KMeans exposes .inertia_.",
            "Scale first: k-means measures distance.",
            "n_init=10 runs it ten times and keeps the best start."))

_c("pca-variance", "explained_variance", I, "breast_cancer", """
Run PCA on the scaled cancer features. How many components are needed to
explain at least 90 per cent of the variance? Leave the count in
`components_needed` and the share the first component alone explains in
`first_share`.
""", setup="from core.datasets import load\ndf = load('breast_cancer')",
   check="""
assert 1 <= components_needed <= 12, (
    f"{components_needed} components is outside the expected range — these "
    "features are heavily correlated, so a handful should carry most of it.")
assert 0.2 < first_share < 0.6, (
    f"The first component explains {first_share:.2f}; that is outside the "
    "expected range. Did you scale before running PCA?")
""", solution="""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

X = df.drop(columns=['diagnosis']).select_dtypes('number')
X = StandardScaler().fit_transform(X)
pca = PCA().fit(X)
shares = pca.explained_variance_ratio_
first_share = float(shares[0])
components_needed = int(np.searchsorted(np.cumsum(shares), 0.90) + 1)
""", hints=("explained_variance_ratio_ is one number per component.",
            "np.cumsum gives the running total.",
            "PCA without scaling just finds the largest-unit column."))


# ---------------------------------------------------------------------------
# Week 8 — leakage, time, calibration
# ---------------------------------------------------------------------------

_c("too-good", "too_good_to_be_true", A, "customer_churn", """
You are handed a model reporting 0.997 AUC on churn. Decide whether to ship it.
Set `ship` to True or False, and leave the name of the column you would remove
first in `suspect_column`.
""", check="""
assert ship is False, (
    "No. A near-perfect AUC on customer churn is not a triumph, it is a "
    "symptom. Real churn is driven by things that only partly determine the "
    "outcome; a score like that means a column is carrying the answer.")
assert suspect_column == 'last_call_outcome', (
    f"You named '{suspect_column}'. The suspect is last_call_outcome: it is "
    "recorded after the customer decided, so it is unavailable at the moment "
    "a prediction is actually needed.")
""", solution="""
ship = False
suspect_column = 'last_call_outcome'
""", hints=("Ask what the best achievable score plausibly is for this problem.",
            "Then ask which column could only be known after the fact."))

_c("temporal-split", "temporal_split", A, "customer_churn", """
For a time-ordered problem, a random split leaks the future. Using tenure_months
as a stand-in for time, build a split where the test set contains only the
NEWEST customers. Leave the maximum tenure in the test set in `test_max_tenure`
and the minimum tenure in the training set in `train_min_tenure`.
""", setup=LOAD_CHURN, check="""
assert test_max_tenure <= train_min_tenure, (
    "Every test row must be newer than every training row — that is what a "
    "temporal split means. Newer customers have SHORTER tenure, so the test "
    "set is the low-tenure end.")
""", solution="""
ordered = df.sort_values('tenure_months', ascending=False)
cut = int(len(ordered) * 0.75)
train = ordered.iloc[:cut]
test = ordered.iloc[cut:]
test_max_tenure = test['tenure_months'].max()
train_min_tenure = train['tenure_months'].min()
""", hints=("Sort by tenure, longest first, then cut.",
            "Long tenure means an old customer; short means a new one.",
            "iloc slices by position, not by label."))

_c("group-leak", "group_leakage", A, "customer_churn", """
If the same customer appears twice, a random split can put one row in training
and the other in test. Show the danger: count how many customer_ids would
appear in BOTH halves if you duplicated every row and split randomly. Leave it
in `shared_ids`.
""", setup=LOAD_CHURN, check="""
assert shared_ids > 0, (
    "With every row duplicated, a random split is almost certain to separate "
    "some pairs. That is group leakage: the model has effectively seen the "
    "test rows already.")
""", solution="""
import pandas as pd
from sklearn.model_selection import train_test_split
doubled = pd.concat([df, df], ignore_index=True)
train, test = train_test_split(doubled, test_size=0.25, random_state=0)
shared_ids = len(set(train['customer_id']) & set(test['customer_id']))
""", hints=("pd.concat([df, df]) duplicates every row.",
            "Set intersection finds the ids present on both sides."))


# ---------------------------------------------------------------------------
# Week 9 — neural networks from scratch
# ---------------------------------------------------------------------------

_c("gradient-descent", "gradient_descent", A, "house_prices", """
Implement gradient descent by hand for a single-feature linear model. Starting
from slope 0 and intercept 0, take 400 steps on the scaled area against the
logged price. Leave the final mean squared error in `final_mse` and the MSE
after the first step in `first_mse`.
""", setup=LOAD_HOUSE, check="""
assert final_mse < first_mse, (
    "The loss must fall. If it rose, the learning rate is too large and each "
    "step is overshooting the valley.")
assert final_mse < 1.0, (
    f"An MSE of {final_mse:.3f} on the logged price means it has barely "
    "moved. Check the gradient's sign: you step AGAINST the gradient.")
""", solution="""
import numpy as np
x = df['area_sqft'].to_numpy(dtype=float)
x = (x - x.mean()) / x.std()
y = np.log(df['price'].to_numpy(dtype=float))

w = b = 0.0
rate = 0.05
first_mse = None
for step in range(400):
    prediction = w * x + b
    error = prediction - y
    w -= rate * 2 * (error * x).mean()
    b -= rate * 2 * error.mean()
    if step == 0:
        first_mse = float((error ** 2).mean())
final_mse = float(((w * x + b - y) ** 2).mean())
""", hints=("The gradient of the mean squared error with respect to w is "
            "2 * mean(error * x).",
            "Subtract the gradient times the learning rate.",
            "Scale x first, or the gradient is enormous and it diverges."))

_c("perceptron-or", "perceptron", I, "customer_churn", """
A single perceptron can learn OR but not XOR. Implement the OR function with
hand-chosen weights and a bias: leave a function `predict(a, b)` that returns 1
or 0. No library needed.
""", check="""
assert predict(0, 0) == 0, "OR of 0 and 0 is 0."
assert predict(0, 1) == 1, "OR of 0 and 1 is 1."
assert predict(1, 0) == 1, "OR of 1 and 0 is 1."
assert predict(1, 1) == 1, "OR of 1 and 1 is 1."
""", solution="""
def predict(a, b):
    total = 1.0 * a + 1.0 * b - 0.5
    return 1 if total > 0 else 0
""", hints=("A perceptron computes w1*a + w2*b + bias, then checks the sign.",
            "For OR, any single input being 1 must push the total above zero.",
            "Try weights of 1 and a bias of -0.5."))


# ---------------------------------------------------------------------------
# Week 10 — shipping
# ---------------------------------------------------------------------------

_c("cost-threshold", "cost_matrix", A, "customer_churn", """
A retention call costs 20. A saved customer is worth 200. A missed churner
costs nothing directly but loses the 200. Given a predicted probability p, at
what p does calling break even? Leave it in `break_even_p`.
""", check="""
assert abs(break_even_p - 0.1) < 0.005, (
    f"You got {break_even_p:.3f}. Calling costs 20 regardless; it returns 200 "
    "with probability p. Break-even is where 200p = 20, so p = 0.1. Note how "
    "far that is from the default threshold of 0.5 — and that choosing 0.5 "
    "here would mean declining almost every profitable call.")
""", solution="""
call_cost = 20
saved_value = 200
break_even_p = call_cost / saved_value
""", hints=("Expected gain from calling is p times the value saved.",
            "Set that equal to the cost and solve for p."))

_c("drift", "data_drift", A, "customer_churn", """
Split the churn data into two halves by customer_id and check whether
monthly_charges has drifted between them. Leave the absolute difference in
means in `drift_amount` and a boolean `has_drifted` using a threshold of 5.
""", setup=LOAD_CHURN, check="""
assert drift_amount >= 0, "A difference in means, taken as an absolute value."
assert has_drifted is False or has_drifted is (drift_amount > 5), (
    "has_drifted should follow from drift_amount and the threshold.")
assert drift_amount < 5, (
    "These halves come from one sample, so there is no real drift to find — "
    "and that is worth seeing. A drift alarm that fires on random variation "
    "is an alarm nobody will listen to when it matters.")
""", solution="""
charges = df['monthly_charges'].replace(9999, None).dropna()
half = len(charges) // 2
drift_amount = abs(charges.iloc[:half].mean() - charges.iloc[half:].mean())
has_drifted = drift_amount > 5
""", hints=("Remove the 9999 placeholders first, or they dominate the mean.",
            "iloc slices by position."))

_c("save-load", "serialisation", I, "customer_churn", """
Fit a pipeline, serialise it to bytes in memory with joblib, load it back, and
confirm the reloaded model gives identical predictions. Leave the number of
disagreements in `disagreements`.
""", setup=LOAD_CHURN, check="""
assert disagreements == 0, (
    f"{disagreements} predictions changed after a round trip. A saved model "
    "that does not reproduce itself exactly is not saved.")
""", solution="""
import io
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

X = df[['tenure_months', 'support_calls']].fillna(0)
y = df['churned']
pipe = Pipeline([('scale', StandardScaler()),
                 ('model', LogisticRegression(max_iter=1000))]).fit(X, y)

buffer = io.BytesIO()
joblib.dump(pipe, buffer)
buffer.seek(0)
reloaded = joblib.load(buffer)
disagreements = int((pipe.predict(X) != reloaded.predict(X)).sum())
""", hints=("joblib.dump accepts a file-like object, so io.BytesIO works.",
            "seek(0) rewinds the buffer before loading.",
            "Comparing two prediction arrays gives a boolean array."))


# ---------------------------------------------------------------------------
# Week 11 — statistics
# ---------------------------------------------------------------------------

_c("bootstrap-ci", "resampling", A, "customer_churn", """
Put error bars on the churn rate. Bootstrap it 500 times and leave the 2.5th
and 97.5th percentiles in `ci_low` and `ci_high`.
""", setup=LOAD_CHURN, check="""
assert ci_low < ci_high, "The lower bound must be below the upper one."
assert ci_low < df['churned'].mean() < ci_high, (
    "The interval should contain the observed rate.")
assert ci_high - ci_low < 0.06, (
    f"An interval {ci_high - ci_low:.3f} wide is too wide for 4200 rows. "
    "Check that each resample is the same SIZE as the original — that is what "
    "makes the spread meaningful.")
""", solution="""
import numpy as np
values = df['churned'].to_numpy()
rng = np.random.default_rng(0)
means = [rng.choice(values, size=len(values), replace=True).mean()
         for _ in range(500)]
ci_low, ci_high = np.percentile(means, [2.5, 97.5])
""", hints=("Resample WITH replacement, at the original size.",
            "np.percentile takes a list of percentiles at once."))

_c("paired-folds", "same_folds", A, "customer_churn", """
Compare two models fairly: cross-validate both on the SAME folds and look at
the per-fold differences. Leave the mean difference in `mean_difference` and
the number of folds where the first model won in `folds_won`.
""", setup=LOAD_CHURN, check="""
assert 0 <= folds_won <= 5, "There are five folds."
assert isinstance(mean_difference, float), (
    "mean_difference should be one number: the average of the per-fold gaps.")
assert abs(mean_difference) < 0.2, (
    "A gap that large between two reasonable models is suspicious.")
""", solution="""
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

X = df[['tenure_months', 'monthly_charges', 'support_calls']].fillna(0)
y = df['churned']
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

a = cross_val_score(LogisticRegression(max_iter=2000), X, y, cv=folds)
b = cross_val_score(DecisionTreeClassifier(max_depth=4, random_state=0),
                    X, y, cv=folds)
differences = a - b
mean_difference = float(np.mean(differences))
folds_won = int((differences > 0).sum())
""", hints=("Build the StratifiedKFold once and pass the same object to both.",
            "Subtracting the two score arrays gives the per-fold differences."))


def unlocked_by(day: int) -> list[dict[str, Any]]:
    """The challenges available to someone on this day of the plan."""
    return [c for c in CHALLENGES if 0 < c["unlock_day"] <= day]


def by_skill() -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for challenge in CHALLENGES:
        out.setdefault(challenge["skill"], []).append(challenge)
    return out


def counts() -> dict[str, Any]:
    return {
        "total": len(CHALLENGES),
        "skills": len(by_skill()),
        "by_difficulty": {
            level: sum(1 for c in CHALLENGES if c["difficulty"] == level)
            for level in (B, I, A)
        },
        "datasets": sorted({c["dataset"] for c in CHALLENGES}),
        "first_unlock": min(c["unlock_day"] for c in CHALLENGES),
        "last_unlock": max(c["unlock_day"] for c in CHALLENGES),
    }
