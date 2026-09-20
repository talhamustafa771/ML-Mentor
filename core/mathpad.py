"""The Math Helper.

You said it plainly: maths is where you struggle, so you want to be able to ask
about any symbol or formula the moment it appears. This module is that.

It works in three layers, in this order.

A written bank. The sixty-odd terms this course actually leans on are explained
here, in the file, with the formula, a worked example using small numbers you
can check in your head, and a sentence on why it is in the course at all. These
answers need no API key, cost nothing, load instantly, and cannot be
hallucinated. Every worked example in the bank is verified by the test suite,
which recomputes the arithmetic and fails if a number here is wrong.

A symbol glossary. Most of the time the blocker is not the idea, it is that
nobody ever said out loud what the squiggle is called. Sigma, theta, y-hat,
nabla: each one gets a name, a pronunciation and a plain description of what it
does, so a formula stops being a wall of shapes.

A model, for anything else. Terms not in the bank go to the LLM under a prompt
that forbids it from explaining one unfamiliar thing with another, holds it to
what you have been taught by that day, and requires a worked example with real
numbers. Those answers are cached, so a term is only ever paid for once.

Underneath all three is the ladder. If you ask what AUC means and the true gap
is that nobody explained a false positive rate, the answer says so and offers
that question next — the chain comes from the curriculum's own prerequisites,
so it is the actual teaching order rather than a guess.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any

from . import concepts


# ---------------------------------------------------------------------------
# The written bank
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Entry:
    key: str
    title: str
    plain: str            # what it is, no symbols
    formula: str          # LaTeX, for st.latex
    reads_as: str         # the formula, spoken aloud
    worked: str           # small numbers, checkable by hand
    why: str              # why this course needs it
    symbols: tuple[str, ...] = ()
    also: tuple[str, ...] = ()      # other names for the same thing


BANK: dict[str, Entry] = {}


def _M(key: str, title: str, *, plain: str, formula: str, reads_as: str,
       worked: str, why: str, symbols: tuple[str, ...] = (),
       also: tuple[str, ...] = ()) -> None:
    entry = Entry(key, title, plain, formula, reads_as, worked, why, symbols, also)
    BANK[key] = entry
    for alias in also:
        BANK.setdefault(alias, entry)


# --- describing a column ---------------------------------------------------
_M("mean", "The mean",
   plain="Add everything up, then divide by how many things there were.",
   formula=r"\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i",
   reads_as="x-bar equals one over n, times the sum of all the x values",
   worked="Take 2, 4 and 9. They add to 15. There are 3 of them. "
          "15 / 3 = 5, so the mean is 5.",
   why="Nearly every other formula in the course starts by taking a mean of "
       "something: of errors, of scores, of distances.",
   symbols=(r"\bar{x}", r"\sum", "n"), also=("average",))

_M("median", "The median",
   plain="Sort the values and take the middle one. With an even count, "
         "average the two in the middle.",
   formula=r"\text{median} = x_{(\frac{n+1}{2})}",
   reads_as="the value sitting in position (n plus one) over two, once sorted",
   worked="Take 2, 4, 9, 100. Sorted, the middle two are 4 and 9, so the "
          "median is 6.5. The mean of those same numbers is 28.75 — one "
          "large value dragged it up, and the median ignored it.",
   why="It is the honest summary when a column is skewed, which house prices "
       "and incomes always are.",
   also=("quartile", "percentile"))

_M("standard_deviation", "Standard deviation",
   plain="The typical distance between a value and the mean. Small means the "
         "values huddle; large means they spread.",
   formula=r"\sigma = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(x_i - \bar{x})^2}",
   reads_as="sigma equals the square root of the average squared distance "
            "from the mean",
   worked="Take 2, 4, 9. The mean is 5. The distances are -3, -1 and 4. "
          "Square them: 9, 1, 16, adding to 26. Divide by 3: 8.67. Square "
          "root: about 2.94.",
   why="Scaling a feature means subtracting its mean and dividing by this. "
       "It is also what a standard error and a z-score are built from.",
   symbols=(r"\sigma", r"\bar{x}"), also=("sd", "sigma"))

_M("variance", "Variance",
   plain="Standard deviation before the square root: the average squared "
         "distance from the mean.",
   formula=r"\sigma^2 = \frac{1}{n}\sum_{i=1}^{n}(x_i - \bar{x})^2",
   reads_as="sigma squared equals the average of the squared distances from "
            "the mean",
   worked="Using 2, 4, 9 again: the squared distances were 9, 1 and 16, so "
          "the variance is 26 / 3 = 8.67. Standard deviation is its square "
          "root, 2.94.",
   why="Squaring is what makes variance split cleanly into parts, which is "
       "what the bias-variance decomposition needs. The price is that its "
       "units are squared, which is why you usually report the square root.",
   symbols=(r"\sigma^2",))

_M("skewness", "Skew",
   plain="Whether a column leans. A long tail to the right is positive skew; "
         "a long tail to the left is negative.",
   formula=r"\text{skew} = \frac{\frac{1}{n}\sum(x_i-\bar{x})^3}{\sigma^3}",
   reads_as="the average cubed distance from the mean, divided by sigma cubed",
   worked="You rarely compute it by hand. What matters is the sign and rough "
          "size: house prices in the bundled set come out around +2.2, which "
          "is heavily right-skewed. Taking the log drops it to about 0.2.",
   why="A skewed target breaks the assumption behind least squares and makes "
       "the residuals fan out. Spotting the skew is what tells you to log it.",
   also=("skew",))

_M("logarithm", "The logarithm",
   plain="The log answers: what power do I raise the base to, to get this "
         "number? It squashes large numbers hard and small numbers gently.",
   formula=r"\log_b(x) = y \iff b^{y} = x",
   reads_as="log base b of x equals y, exactly when b to the power y equals x",
   worked="log base 10 of 1000 is 3, because 10 x 10 x 10 = 1000. In this "
          "course log almost always means the natural log, ln, which uses "
          "the base e = 2.718. ln(1) = 0, and ln of a house price of 250,000 "
          "is about 12.4.",
   why="It turns multiplying into adding, which is how a model that "
       "multiplies effects becomes one a straight line can fit.",
   symbols=(r"\log", r"\ln", "e"), also=("log", "ln", "natural_log"))

_M("exponential", "The exponential",
   plain="The log run backwards. Raising e to a power undoes taking the natural log of it, so exp is how you get out of log units and back into the units of the thing itself.",
   formula=r"e^{x},\quad e^{\ln(x)} = x",
   reads_as="e to the power x; e to the power of the natural log of x gives "
            "back x",
   worked="If your model predicts a log-price of 12.4, the actual price is "
          "exp(12.4), which is about 242,800. You must exponentiate before "
          "reporting, or you will quote a price of twelve pounds.",
   why="Every model trained on a logged target predicts in log units, and "
       "this is the step back to money.",
   symbols=("e^x",), also=("exp",))

# --- two columns together --------------------------------------------------
_M("covariance", "Covariance",
   plain="Do two columns move together? Positive means they rise together, "
         "negative means one rises as the other falls.",
   formula=r"\mathrm{cov}(x,y) = \frac{1}{n}\sum (x_i-\bar{x})(y_i-\bar{y})",
   reads_as="the average of: x's distance from its mean, times y's distance "
            "from its mean",
   worked="When a house is bigger than average AND pricier than average, both "
          "brackets are positive and their product is positive. When it is "
          "bigger but cheaper, the product is negative. Averaging over every "
          "house gives the covariance.",
   why="Correlation is this, rescaled. The covariance matrix is what PCA "
       "takes apart.",
   symbols=(r"\mathrm{cov}",))

_M("pearson_correlation", "Correlation (Pearson's r)",
   plain="Covariance divided by both standard deviations, which forces it "
         "onto a scale from -1 to +1 and strips out the units.",
   formula=r"r = \frac{\mathrm{cov}(x,y)}{\sigma_x \sigma_y}",
   reads_as="r equals the covariance of x and y, over sigma-x times sigma-y",
   worked="r = 1 is a perfect straight line upward, r = -1 perfect downward, "
          "r = 0 no straight-line relationship. Area and price in the bundled "
          "houses come out around 0.7: strong, but far from everything.",
   why="It is the fastest read on which features might matter — and its "
       "famous limit is that it only sees straight lines. A perfect U-shape "
       "can score r = 0.",
   symbols=("r", r"\rho"), also=("correlation", "r"))

# --- error and fit ---------------------------------------------------------
_M("absolute_error", "Mean absolute error (MAE)",
   plain="How far off you are on average, ignoring which side you missed on.",
   formula=r"\mathrm{MAE} = \frac{1}{n}\sum_{i=1}^{n}\left|y_i - \hat{y}_i\right|",
   reads_as="the average of the absolute values of the gaps between the true "
            "y and the predicted y-hat",
   worked="Three predictions miss by +2, -1 and +3. Absolute values: 2, 1, 3. "
          "They add to 6, so MAE = 6 / 3 = 2. It is in the same units as the "
          "target: two pounds, two days, two customers.",
   why="It is the metric you can say out loud to a non-technical person, and "
       "the one that does not panic about a single large miss.",
   symbols=(r"\hat{y}", "|x|"), also=("mae",))

_M("squared_error", "Mean squared error (MSE)",
   plain="Square each miss before averaging, so a big miss hurts far more "
         "than several small ones.",
   formula=r"\mathrm{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2",
   reads_as="the average of the squared gaps between true and predicted",
   worked="The same misses of +2, -1, +3 square to 4, 1 and 9, adding to 14. "
          "MSE = 14 / 3 = 4.67. Note the single miss of 3 now contributes "
          "more than the other two combined.",
   why="Squaring is what makes the maths differentiable and solvable, which "
       "is why almost every regression minimises this rather than MAE.",
   also=("mse", "mean_squared_error", "least_squares"))

_M("rmse", "Root mean squared error (RMSE)",
   plain="The square root of MSE, which puts the number back into the units "
         "of the thing you are predicting.",
   formula=r"\mathrm{RMSE} = \sqrt{\frac{1}{n}\sum (y_i - \hat{y}_i)^2}",
   reads_as="the square root of the mean squared error",
   worked="MSE was 4.67, so RMSE is about 2.16. Compare that with the MAE of "
          "2.0 on the same predictions: RMSE is always the larger of the two, "
          "and the gap between them tells you how uneven your errors are.",
   why="It is MSE you can quote in pounds. Reporting both it and MAE is a "
       "cheap way to show whether a few bad misses are driving the average.",
   also=("root_mean_squared_error",))

_M("r_squared_formula", "R squared",
   plain="The share of the target's variation your model accounts for. 1 is "
         "perfect, 0 is no better than always guessing the mean, and negative "
         "means worse than that.",
   formula=r"R^2 = 1 - \frac{SS_{res}}{SS_{tot}}"
           r" = 1 - \frac{\sum(y_i-\hat{y}_i)^2}{\sum(y_i-\bar{y})^2}",
   reads_as="R squared equals one minus the residual sum of squares over the "
            "total sum of squares",
   worked="True values 10, 12, 20 have a mean of 14, so the total sum of "
          "squares is 16 + 4 + 36 = 56. Predictions 12, 13, 17 leave "
          "residuals of -2, -1, +3, so the residual sum of squares is "
          "4 + 1 + 9 = 14. R squared = 1 - 14/56 = 0.75.",
   why="It is the one regression number that needs no context to read. Its "
       "trap is that it never falls when you add a feature, which is why "
       "the adjusted version exists.",
   symbols=("R^2", "SS_{res}"), also=("r_squared", "adjusted_r_squared"))

_M("residual", "A residual",
   plain="One row's miss: what actually happened, minus what you predicted.",
   formula=r"e_i = y_i - \hat{y}_i",
   reads_as="e-sub-i equals y-sub-i minus y-hat-sub-i",
   worked="The house sold for 310,000 and you predicted 288,000. The residual "
          "is +22,000 — positive, so you under-predicted.",
   why="Plotting residuals is how you find the shape your model missed. A "
       "residual plot with a pattern in it is a model with a bug in it.",
   symbols=("e_i",), also=("residuals", "variance_of_residuals"))

# --- classification metrics ------------------------------------------------
_M("precision_formula", "Precision",
   plain="Of everything you flagged, what share was right? It is the metric "
         "that cares about false alarms.",
   formula=r"\text{precision} = \frac{TP}{TP + FP}",
   reads_as="true positives over true positives plus false positives",
   worked="You flag 10 customers as likely to leave. 8 really do. "
          "Precision = 8 / 10 = 0.8.",
   why="It is the one that matters when acting on a prediction costs "
       "something: every flag is a phone call somebody has to make.",
   symbols=("TP", "FP"), also=("precision",))

_M("recall_formula", "Recall",
   plain="Of everything that was actually positive, what share did you "
         "catch? It is the metric that cares about misses.",
   formula=r"\text{recall} = \frac{TP}{TP + FN}",
   reads_as="true positives over true positives plus false negatives",
   worked="20 customers actually left. You flagged 8 of them. "
          "Recall = 8 / 20 = 0.4. Note you can reach recall 1.0 instantly by "
          "flagging everybody — which is why it is never read alone.",
   why="When a miss is the expensive mistake — a missed malignancy, a missed "
       "fraud — this is the number the decision turns on.",
   symbols=("TP", "FN"), also=("recall", "true_positive_rate", "sensitivity"))

_M("harmonic_mean", "F1, and the harmonic mean",
   plain="One number combining precision and recall, built so that being bad "
         "at either one drags it down.",
   formula=r"F_1 = 2 \cdot \frac{\text{precision}\cdot\text{recall}}"
           r"{\text{precision}+\text{recall}}",
   reads_as="F-one equals two, times precision times recall, over precision "
            "plus recall",
   worked="Precision 0.8, recall 0.4. The ordinary average would be 0.6. "
          "F1 = 2 x (0.8 x 0.4) / (0.8 + 0.4) = 0.64 / 1.2 = 0.53 — "
          "noticeably lower, because the harmonic mean refuses to let a "
          "strong score hide a weak one.",
   why="It is the standard single number for imbalanced classification, and "
       "the reason is exactly that refusal.",
   symbols=("F_1",), also=("f1", "f1_score"))

_M("false_positive_rate", "False positive rate",
   plain="Of everything that was actually negative, what share did you wrongly "
         "flag?",
   formula=r"\mathrm{FPR} = \frac{FP}{FP + TN}",
   reads_as="false positives over false positives plus true negatives",
   worked="100 customers stayed. You wrongly flagged 2 of them. "
          "FPR = 2 / 100 = 0.02.",
   why="It is the horizontal axis of the ROC curve. Recall is the vertical "
       "one, and the curve is the trade-off between them.",
   symbols=("FP", "TN"))

_M("area_under_curve", "AUC",
   plain="The area under the ROC curve. Read it as: pick one real positive "
         "and one real negative at random; AUC is the chance your model gives "
         "the positive the higher score.",
   formula=r"\mathrm{AUC} = P\big(\hat{p}(\text{positive}) > "
           r"\hat{p}(\text{negative})\big)",
   reads_as="the probability that a random positive scores above a random "
            "negative",
   worked="0.5 is a coin flip — the model is ranking no better than chance. "
          "0.71 is the honest score on the bundled churn data. 0.997 is what "
          "you get with the leaking column left in, which is how you learn to "
          "distrust a suspiciously high AUC.",
   why="It judges the ranking rather than one threshold, so it does not move "
       "when you change where you cut.",
   symbols=(r"\hat{p}",), also=("auc", "roc_auc"))

_M("log_loss_formula", "Log loss (cross-entropy)",
   plain="Scores a predicted probability rather than a yes/no. Being "
         "confident and right costs almost nothing; confident and wrong is "
         "punished savagely.",
   formula=r"\mathcal{L} = -\frac{1}{n}\sum \big[y_i\log(\hat{p}_i) + "
           r"(1-y_i)\log(1-\hat{p}_i)\big]",
   reads_as="minus the average of: y times log p-hat, plus one-minus-y times "
            "log of one-minus-p-hat",
   worked="The truth is 1. Predict 0.9 and the loss is -ln(0.9) = 0.105. "
          "Predict 0.1 on that same row and it is -ln(0.1) = 2.303, more than "
          "twenty times worse. Predict 0.0 and it is infinite.",
   why="It is what logistic regression actually minimises, and the only "
       "common metric that rewards honest uncertainty.",
   symbols=(r"\mathcal{L}", r"\hat{p}"),
   also=("cross_entropy", "log_loss"))

_M("brier_formula", "The Brier score",
   plain="Mean squared error, applied to predicted probabilities.",
   formula=r"\mathrm{BS} = \frac{1}{n}\sum (\hat{p}_i - y_i)^2",
   reads_as="the average squared gap between the predicted probability and "
            "what actually happened",
   worked="You said 0.9 and it happened: (0.9 - 1)^2 = 0.01. You said 0.9 and "
          "it did not: (0.9 - 0)^2 = 0.81. Lower is better, and 0 is perfect.",
   why="It is the gentler sibling of log loss — it does not go to infinity — "
       "and it is the usual way to check whether probabilities are calibrated.",
   also=("brier_score",))

# --- probability -----------------------------------------------------------
_M("conditional_probability", "Conditional probability",
   plain="The chance of A given that you already know B happened. Knowing B "
         "shrinks the world you are counting in.",
   formula=r"P(A \mid B) = \frac{P(A \cap B)}{P(B)}",
   reads_as="P of A given B equals P of A and B, over P of B",
   worked="Of 1000 customers, 200 are on month-to-month contracts, and 90 of "
          "those left. P(left | month-to-month) = 90 / 200 = 0.45 — against a "
          "base rate across everyone of much less.",
   why="Every classifier is estimating a conditional probability: the chance "
       "of the target, given the features.",
   symbols=(r"P(A \mid B)", r"\cap"), also=("conditional", "joint_probability"))

_M("bayes_theorem", "Bayes' theorem",
   plain="Turns P(evidence | cause) into P(cause | evidence) — which is "
         "usually the direction you actually want.",
   formula=r"P(A \mid B) = \frac{P(B \mid A)\,P(A)}{P(B)}",
   reads_as="P of A given B equals P of B given A, times P of A, over P of B",
   worked="A disease affects 1 in 100. The test catches 99% of cases and "
          "wrongly flags 5% of healthy people. You test positive. "
          "P(ill | positive) = (0.99 x 0.01) / (0.99 x 0.01 + 0.05 x 0.99) "
          "= 0.0099 / 0.0594 = 0.167. Only about 17%, because the healthy "
          "group is so much larger.",
   why="It is the whole of naive Bayes, and the correction that stops you "
       "over-reading a positive result on a rare condition.",
   symbols=(r"P(A \mid B)",), also=("bayes_rule", "prior", "posterior"))

_M("base_rate", "The base rate",
   plain="How common the thing is before you look at any evidence.",
   formula=r"P(A)",
   reads_as="P of A, the probability of A on its own",
   worked="46% of the bundled churn customers left. Any model must beat 46% "
          "accuracy to have done anything at all — and predicting 'everybody "
          "leaves' would score exactly that.",
   why="It is the number that makes a 95% accuracy claim meaningless when the "
       "positive class is 3% of the data.",
   also=("class_prior", "prior_probability"))

_M("independence", "Independence",
   plain="Two things are independent when knowing one tells you nothing about "
         "the other.",
   formula=r"P(A \cap B) = P(A)\,P(B)",
   reads_as="P of A and B equals P of A times P of B",
   worked="Two fair coins: P(both heads) = 0.5 x 0.5 = 0.25. But tenure and "
          "total charges are obviously not independent — knowing one "
          "narrows the other a lot — so you may not multiply their "
          "probabilities like that.",
   why="It is the assumption naive Bayes makes and knowingly breaks, and the "
       "assumption a train/test split relies on to be a fair test.",
   symbols=(r"\cap",), also=("conditional_independence", "independence_assumption"))

_M("normal_distribution", "The normal distribution",
   plain="The bell curve: most values near the middle, symmetric tails, and "
         "almost nothing beyond three standard deviations.",
   formula=r"X \sim \mathcal{N}(\mu, \sigma^2)",
   reads_as="X is distributed normally with mean mu and variance sigma squared",
   worked="About 68% of values fall within one standard deviation of the mean "
          "and 95% within two. With a mean of 60 and an SD of 10, roughly 95 "
          "people in 100 sit between 40 and 80.",
   why="Residuals are assumed roughly normal by linear regression, and the "
       "central limit theorem is why averages become normal even when the raw "
       "data is not.",
   symbols=(r"\mu", r"\sigma^2", r"\mathcal{N}"), also=("gaussian",))

_M("clt", "The central limit theorem",
   plain="Take samples, average each one, and those averages form a bell "
         "curve — even if the thing you sampled was nothing like a bell curve.",
   formula=r"\bar{X}_n \xrightarrow{d} \mathcal{N}\left(\mu, "
           r"\frac{\sigma^2}{n}\right)",
   reads_as="the sample mean tends towards a normal distribution with mean mu "
            "and variance sigma squared over n",
   worked="House prices are heavily right-skewed. But take 50 houses at a "
          "time and average them, repeat a thousand times, and those "
          "thousand averages are close to symmetric and bell-shaped.",
   why="It is the reason a confidence interval around a mean score works at "
       "all, and why cross-validation's average is more trustworthy than any "
       "one fold.",
   symbols=(r"\bar{X}_n",), also=("central_limit_theorem",))

_M("standard_error", "The standard error",
   plain="The standard deviation of an estimate rather than of the data: how "
         "much your average would wobble if you collected the data again.",
   formula=r"\mathrm{SE} = \frac{\sigma}{\sqrt{n}}",
   reads_as="sigma over the square root of n",
   worked="An SD of 10 across 100 rows gives SE = 10 / 10 = 1. Note the "
          "square root: to halve the wobble you need four times the data.",
   why="It is what turns five cross-validation scores into a statement about "
       "whether model A really beat model B or just got luckier folds.",
   symbols=(r"\sqrt{n}",))

_M("confidence_interval", "A confidence interval",
   plain="A range around an estimate, wide enough that the procedure would "
         "contain the true value most of the time.",
   formula=r"\bar{x} \pm 1.96 \times \mathrm{SE}",
   reads_as="x-bar plus or minus 1.96 standard errors",
   worked="A mean of 50 with a standard error of 1 gives 50 ± 1.96, so about "
          "48.0 to 52.0. If a rival model scores 51, the two intervals "
          "overlap heavily and you have not shown anything.",
   why="It converts 'my model scored 0.83' into 'my model scored 0.83, give "
       "or take 0.04', which is the only version worth reporting.",
   also=("uncertainty_quantification", "sampling_distribution"))

_M("p_value", "The p-value",
   plain="If nothing real were going on, how often would you see a result at "
         "least this extreme by chance alone?",
   formula=r"p = P(\text{result at least this extreme} \mid H_0 \text{ true})",
   reads_as="p equals the probability of a result this extreme, given the "
            "null hypothesis is true",
   worked="p = 0.03 means: a world with no real effect would produce this "
          "result about 3 times in 100. It does NOT mean there is a 3% chance "
          "you are wrong — that is the single most common misreading of it.",
   why="It is how you check whether the gap between two models survives the "
       "noise in your folds.",
   symbols=("H_0", "p"), also=("significance", "null_hypothesis", "test_statistic"))

_M("bonferroni", "The Bonferroni correction",
   plain="Test enough things and something will look significant by luck. "
         "This divides your threshold by the number of tests.",
   formula=r"\alpha_{\text{adjusted}} = \frac{\alpha}{m}",
   reads_as="alpha divided by m, the number of tests",
   worked="Testing 5 models at the usual 0.05 threshold: use 0.05 / 5 = 0.01 "
          "for each. Without the correction, the chance of at least one false "
          "alarm across five tests is about 23%, not 5%.",
   why="Every hyperparameter grid is dozens of simultaneous comparisons. This "
       "is why the winner of a large grid search often does not repeat.",
   symbols=(r"\alpha", "m"), also=("multiple_testing",))

# --- linear models and calculus --------------------------------------------
_M("linear_equation", "A linear equation",
   plain="A prediction made by multiplying each feature by its own weight and "
         "adding everything up, plus a starting value.",
   formula=r"\hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + "
           r"\beta_p x_p",
   reads_as="y-hat equals beta-zero, plus beta-one times x-one, plus beta-two "
            "times x-two, and so on",
   worked="Price = 40,000 + 120 x area + 8,000 x quality. A 1,500 sq ft house "
          "of quality 6 predicts 40,000 + 180,000 + 48,000 = 268,000. "
          "Beta-zero, the 40,000, is where the line starts.",
   why="Linear regression, logistic regression, SVMs and a single neuron are "
       "all this same line with different things done to the result.",
   symbols=(r"\beta_0", r"\beta_1", r"\hat{y}"),
   also=("coefficient", "intercept", "weighted_sum", "linear_model"))

_M("summation", "Sigma notation",
   plain="A compact way of writing 'add all of these up'. It is a for-loop "
         "wearing a Greek letter.",
   formula=r"\sum_{i=1}^{n} x_i = x_1 + x_2 + \dots + x_n",
   reads_as="the sum, from i equals one to n, of x-sub-i",
   worked="With x = [2, 4, 9], the sum from i=1 to 3 of x_i is 2 + 4 + 9 = 15. "
          "In code that is simply sum(x). The i underneath is the counter; "
          "the n on top is where it stops.",
   why="Once you read sigma as 'for each row, add', two thirds of the "
       "formulas in this course stop looking like formulas.",
   symbols=(r"\sum", "i", "n"))

_M("argmin", "argmin and argmax",
   plain="Not the smallest value — the input that produced it. 'Which one "
         "wins', rather than 'what did it score'.",
   formula=r"\hat{\beta} = \arg\min_{\beta} \; \mathcal{L}(\beta)",
   reads_as="beta-hat is the beta that minimises the loss",
   worked="Scores of [0.7, 0.9, 0.4] for three models. The max is 0.9; the "
          "argmax is 1, the position of the winner. Training a model is "
          "always an argmin: find the weights that make the loss smallest.",
   why="It is how every 'fit' in this course is written down formally.",
   symbols=(r"\arg\min", r"\arg\max"), also=("argmax",))

_M("derivative", "The derivative",
   plain="The slope at a point: if I nudge the input up a little, how much "
         "does the output move, and which way?",
   formula=r"f'(x) = \frac{df}{dx}",
   reads_as="f prime of x, or d-f by d-x",
   worked="For f(x) = x^2, the derivative is 2x. At x = 3 the slope is 6 — "
          "steep and rising. At x = 0 it is 0, which is the bottom of the "
          "curve. That zero is exactly what gradient descent is hunting for.",
   why="It is the whole mechanism of learning: the derivative of the loss "
       "tells the model which way to move each weight.",
   symbols=("f'(x)", r"\frac{df}{dx}"))

_M("partial_derivative", "The partial derivative",
   plain="The same slope, but with several inputs: hold every weight still "
         "except one, and ask how the loss moves as that one changes.",
   formula=r"\frac{\partial \mathcal{L}}{\partial \beta_j}",
   reads_as="partial L by partial beta-j",
   worked="With a loss depending on beta_1 and beta_2, the partial with "
          "respect to beta_1 answers: 'if I raise beta_1 by a hair and leave "
          "beta_2 alone, does the loss go up or down?'",
   why="A model has thousands of weights and must update all of them. One "
       "partial derivative per weight is how.",
   symbols=(r"\partial",))

_M("gradient", "The gradient",
   plain="All the partial derivatives collected into one vector: the compass "
         "needle pointing in the direction of steepest increase.",
   formula=r"\nabla \mathcal{L} = \left[\frac{\partial \mathcal{L}}"
           r"{\partial \beta_1}, \dots, \frac{\partial \mathcal{L}}"
           r"{\partial \beta_p}\right]",
   reads_as="nabla L, or grad L: the vector of every partial derivative",
   worked="If the gradient is [3, -1], raising beta_1 pushes the loss up "
          "three times as fast as raising beta_2 pushes it down. So you step "
          "the opposite way: beta_1 down hard, beta_2 up a little.",
   why="Gradient descent is one line: take a step against the gradient, "
       "repeat. That is training.",
   symbols=(r"\nabla",))

_M("learning_rate", "The learning rate",
   plain="How big a step to take each time you move against the gradient.",
   formula=r"\beta \leftarrow \beta - \eta \, \nabla \mathcal{L}(\beta)",
   reads_as="beta becomes beta minus eta times the gradient of the loss",
   worked="Too small — 0.000001 — and the loss creeps down over thousands of "
          "epochs and you run out of patience. Too large — 10 — and each step "
          "overshoots the valley, the loss climbs, and you get NaN. Something "
          "like 0.01 is a normal starting guess.",
   why="It is the single hyperparameter most likely to be the actual reason a "
       "network 'does not work'.",
   symbols=(r"\eta", r"\leftarrow"), also=("eta", "step_size"))

_M("chain_rule", "The chain rule",
   plain="For a function inside a function, multiply the slopes. It is how "
         "blame travels backwards through layers.",
   formula=r"\frac{d}{dx}f(g(x)) = f'(g(x)) \cdot g'(x)",
   reads_as="the derivative of f of g of x is f-prime of g of x, times "
            "g-prime of x",
   worked="Loss depends on the output, the output depends on the hidden "
          "layer, the hidden layer depends on a weight. Multiply the three "
          "slopes together and you have the loss's slope with respect to that "
          "weight.",
   why="Backpropagation is the chain rule applied layer by layer and nothing "
       "more. Once you see that, the algorithm stops being mysterious.",
   also=("chain_rule_application", "backpropagation"))

_M("vector", "A vector",
   plain="An ordered list of numbers. One row of your data, with its features "
         "in a fixed order, is a vector.",
   formula=r"\mathbf{x} = [x_1, x_2, \dots, x_p]",
   reads_as="bold x, the list of x-one through x-p",
   worked="A house of 1,500 sq ft, 3 bedrooms, quality 6 is the vector "
          "[1500, 3, 6]. Every row of the table becomes a point in a space "
          "with one axis per column.",
   why="'Feature space', 'distance between two rows' and 'a direction' all "
       "mean something precise once rows are vectors.",
   symbols=(r"\mathbf{x}",))

_M("dot_product", "The dot product",
   plain="Multiply two vectors element by element and add up the results. "
         "One number out.",
   formula=r"\mathbf{w}\cdot\mathbf{x} = \sum_{j=1}^{p} w_j x_j",
   reads_as="w dot x equals the sum over j of w-j times x-j",
   worked="[1, 2, 3] dot [4, 5, 6] = 4 + 10 + 18 = 32. A linear model's whole "
          "prediction is one dot product of the weights with the row, plus "
          "the intercept.",
   why="It is also how a kernel measures similarity and how a neuron computes "
       "its input, so it turns up in three separate weeks.",
   symbols=(r"\cdot",))

_M("matrix_multiplication", "Matrix multiplication",
   plain="Doing many dot products at once: every row of the first against "
         "every column of the second.",
   formula=r"(AB)_{ij} = \sum_k A_{ik} B_{kj}",
   reads_as="the i-j entry of A times B is the sum over k of A-i-k times B-k-j",
   worked="A 100x5 data matrix times a 5x1 weight vector gives 100x1: one "
          "prediction per row, in a single operation. The inner numbers must "
          "match — 5 and 5 — and the outer ones give the result's shape.",
   why="It is why numpy is fast: one matrix multiply replaces a hundred "
       "thousand Python loop iterations.",
   also=("matmul",))

_M("normal_equation", "The normal equation",
   plain="The exact, one-shot solution for linear regression's best weights — "
         "no gradient descent needed.",
   formula=r"\hat{\beta} = (X^\top X)^{-1} X^\top y",
   reads_as="beta-hat equals X-transpose-X inverse, times X-transpose y",
   worked="You will not compute this by hand; sklearn's LinearRegression does "
          "it for you. What matters is that it involves an inverse, which "
          "fails when two columns are near-duplicates — and that is exactly "
          "the instability multicollinearity causes.",
   why="It explains why linear regression is instant while a neural network "
       "takes epochs: only the linear case has a closed-form answer.",
   symbols=("X^\\top", "^{-1}"))

_M("l2_norm", "The L2 norm",
   plain="The straight-line length of a vector: square, add, square root.",
   formula=r"\|\mathbf{w}\|_2 = \sqrt{\sum_j w_j^2}",
   reads_as="the L2 norm of w: the square root of the sum of the squared "
            "weights",
   worked="For [3, -4]: 9 + 16 = 25, square root 5. Ridge regression adds "
          "this (squared) to the loss, which shrinks every weight smoothly "
          "towards zero without ever quite reaching it.",
   why="It is the penalty in ridge, and it is also Euclidean distance under "
       "a different name.",
   symbols=(r"\|\cdot\|_2",), also=("l2_penalty", "ridge_penalty"))

_M("l1_norm", "The L1 norm",
   plain="Add up the sizes of the weights, ignoring sign.",
   formula=r"\|\mathbf{w}\|_1 = \sum_j |w_j|",
   reads_as="the L1 norm of w: the sum of the absolute values of the weights",
   worked="For [3, -4]: 3 + 4 = 7. Lasso adds this to the loss, and because "
          "its penalty does not ease off as a weight approaches zero, it "
          "pushes weak weights all the way to exactly 0 — which is why lasso "
          "selects features and ridge does not.",
   why="The difference between L1 and L2 is the whole reason you would choose "
       "lasso over ridge.",
   symbols=(r"\|\cdot\|_1",), also=("l1_penalty", "lasso_penalty"))

_M("euclidean_distance", "Euclidean distance",
   plain="Straight-line distance between two points, in any number of "
         "dimensions. Pythagoras, extended.",
   formula=r"d(\mathbf{a},\mathbf{b}) = \sqrt{\sum_j (a_j - b_j)^2}",
   reads_as="the square root of the sum of the squared differences, feature "
            "by feature",
   worked="From (0,0) to (3,4): 9 + 16 = 25, square root 5. With three "
          "features it is the same sum with one more term.",
   why="k-NN and k-means are built entirely on it — and it is also why "
       "scaling matters: a column measured in thousands dominates the sum "
       "and the distance stops meaning anything.",
   symbols=("d(a,b)",), also=("distance", "euclidean"))

_M("manhattan_distance", "Manhattan distance",
   plain="Distance if you may only travel along the axes, like walking a city "
         "grid.",
   formula=r"d(\mathbf{a},\mathbf{b}) = \sum_j |a_j - b_j|",
   reads_as="the sum of the absolute differences, feature by feature",
   worked="From (0,0) to (3,4) it is 3 + 4 = 7, against the straight-line 5. "
          "It is never shorter than Euclidean.",
   why="It holds up better than straight-line distance when there are many "
       "features, which is one practical answer to the curse of "
       "dimensionality.",
   also=("l1_distance",))

_M("z_score_formula", "The z-score",
   plain="How many standard deviations a value sits from the mean.",
   formula=r"z = \frac{x - \bar{x}}{\sigma}",
   reads_as="z equals x minus the mean, all over sigma",
   worked="x = 80 with a mean of 60 and an SD of 10 gives z = 2: two standard "
          "deviations above average. About 2.5% of a normal distribution sits "
          "beyond z = 2.",
   why="It is both the outlier test and exactly what StandardScaler does to "
       "every column.",
   symbols=("z",), also=("standardisation", "z_score"))

_M("sigmoid_formula", "The sigmoid",
   plain="Squashes any number, however large or negative, into the range 0 "
         "to 1 — so it can be read as a probability.",
   formula=r"\sigma(z) = \frac{1}{1 + e^{-z}}",
   reads_as="sigma of z equals one over one plus e to the minus z",
   worked="At z = 0 it is exactly 0.5. At z = 2 it is 1/(1+0.135) = 0.88. At "
          "z = -2 it is 0.12. Far out in either direction it flattens, which "
          "is where the vanishing gradient problem comes from.",
   why="It is the only thing separating logistic regression from linear "
       "regression, and it was the original activation function.",
   symbols=(r"\sigma(z)",), also=("sigmoid", "logistic_function", "expit"))

_M("odds", "Odds and log-odds",
   plain="Odds compare a probability with its opposite. Log-odds take the "
         "natural log of that, which unlocks the range from minus infinity to "
         "plus infinity.",
   formula=r"\text{odds} = \frac{p}{1-p}, \qquad "
           r"\text{logit}(p) = \ln\!\left(\frac{p}{1-p}\right)",
   reads_as="odds are p over one minus p; the logit is the natural log of "
            "the odds",
   worked="p = 0.8 gives odds of 0.8/0.2 = 4, or '4 to 1 on'. The log-odds "
          "are ln(4) = 1.39. p = 0.5 gives odds of 1 and log-odds of 0 — the "
          "exact centre.",
   why="Logistic regression's coefficients are in log-odds, which is why they "
       "read strangely until you exponentiate them.",
   symbols=(r"\text{logit}",), also=("log_odds", "logit"))

_M("hyperplane", "A hyperplane",
   plain="The flat boundary a linear model draws. A line in 2D, a plane in "
         "3D, and the same idea with more features than you can picture.",
   formula=r"\mathbf{w}\cdot\mathbf{x} + b = 0",
   reads_as="w dot x plus b equals zero",
   worked="On one side of it the expression is positive and the model says "
          "class 1; on the other it is negative and the model says class 0. "
          "Exactly on it the model is at 50/50.",
   why="It is what 'linearly separable' means, and what an SVM is positioning "
       "when it maximises the margin.",
   also=("decision_boundary", "margin"))

# --- trees, clusters, components -------------------------------------------
_M("entropy_formula", "Entropy",
   plain="How mixed a group is, measured in bits. Zero when everything is one "
         "class, largest when it is an even split.",
   formula=r"H = -\sum_{k} p_k \log_2 p_k",
   reads_as="H equals minus the sum over classes of p-k times log-two of p-k",
   worked="A 50/50 split gives H = 1 bit — maximum confusion. A 90/10 split "
          "gives -(0.9 x -0.152 + 0.1 x -3.322) = 0.47 bits. A pure group "
          "gives 0.",
   why="A decision tree picks the split that drops it the most. That drop is "
       "the information gain.",
   symbols=("H", "p_k"), also=("entropy", "information_gain"))

_M("gini_formula", "Gini impurity",
   plain="The chance that two items drawn from the group belong to different "
         "classes. Same job as entropy, cheaper to compute.",
   formula=r"G = 1 - \sum_k p_k^2",
   reads_as="G equals one minus the sum of the squared class proportions",
   worked="A 50/50 split: 1 - (0.25 + 0.25) = 0.5, the worst possible for two "
          "classes. A 90/10 split: 1 - (0.81 + 0.01) = 0.18. A pure group: 0.",
   why="It is sklearn's default split criterion, and in practice it and "
       "entropy almost always choose the same split.",
   symbols=("G",), also=("gini",))

_M("centroid", "A centroid",
   plain="The mean position of a group of points: the average of every "
         "feature, taken over the cluster's members.",
   formula=r"\boldsymbol{\mu}_k = \frac{1}{|C_k|}\sum_{i \in C_k} \mathbf{x}_i",
   reads_as="mu-k equals one over the size of cluster k, times the sum of its "
            "member vectors",
   worked="Three points at (1,1), (3,3) and (5,2) have a centroid at "
          "((1+3+5)/3, (1+3+2)/3) = (3, 2).",
   why="k-means alternates two steps forever: assign each point to its "
       "nearest centroid, then move each centroid to the mean of what it "
       "caught.",
   symbols=(r"\boldsymbol{\mu}_k", r"\in"), also=("mean_vector",))

_M("within_cluster_variance", "Inertia",
   plain="Total squared distance from every point to its own centroid. Lower "
         "means tighter clusters.",
   formula=r"\mathrm{inertia} = \sum_{k}\sum_{i \in C_k} "
           r"\|\mathbf{x}_i - \boldsymbol{\mu}_k\|^2",
   reads_as="the sum, over clusters and over their members, of squared "
            "distance to the centroid",
   worked="It always falls as you add clusters — with one cluster per point "
          "it reaches 0 — which is precisely why you cannot pick k by "
          "minimising it, and why the elbow method looks for the bend instead.",
   why="Understanding that it cannot be minimised naively is the whole "
       "lesson of choosing k.",
   also=("inertia", "elbow_method"))

_M("eigenvector", "Eigenvectors and eigenvalues",
   plain="A direction a transformation does not rotate, and the factor by "
         "which it stretches along that direction.",
   formula=r"A\mathbf{v} = \lambda \mathbf{v}",
   reads_as="A times v equals lambda times v",
   worked="Take the covariance matrix of your features. Its eigenvector with "
          "the largest eigenvalue is the direction the data spreads most. "
          "That direction is the first principal component, and its "
          "eigenvalue is how much variance lies along it.",
   why="PCA is exactly this and nothing else: find the directions of greatest "
       "spread, keep the first few, discard the rest.",
   symbols=(r"\lambda", r"\mathbf{v}"),
   also=("eigenvalue", "principal_component", "pca"))

_M("covariance_matrix", "The covariance matrix",
   plain="A square table holding every pair's covariance, with each feature's "
         "own variance down the diagonal.",
   formula=r"\Sigma_{jk} = \mathrm{cov}(x_j, x_k)",
   reads_as="the j-k entry of Sigma is the covariance of feature j with "
            "feature k",
   worked="With 5 features it is 5x5 and symmetric, because cov(a,b) equals "
          "cov(b,a). The diagonal entries are variances, since a feature's "
          "covariance with itself is its variance.",
   why="It is the input to PCA, and looking at it is the quickest way to see "
       "which features are near-duplicates of each other.",
   symbols=(r"\Sigma",))

_M("orthogonality", "Orthogonality",
   plain="Two directions at right angles, carrying no shared information. "
         "Their dot product is zero.",
   formula=r"\mathbf{a}\cdot\mathbf{b} = 0",
   reads_as="a dot b equals zero",
   worked="[1, 0] and [0, 1] are orthogonal: their dot product is "
          "1x0 + 0x1 = 0. Every pair of principal components is orthogonal by "
          "construction, which is why they never repeat each other.",
   why="It is the guarantee that PCA's components do not overlap, and the "
       "reason they cure multicollinearity.",
   also=("orthogonal",))

_M("kl_divergence", "KL divergence",
   plain="How much one probability distribution differs from another. Zero "
         "when identical, and never negative.",
   formula=r"D_{KL}(P\,\|\,Q) = \sum_i P(i)\log\frac{P(i)}{Q(i)}",
   reads_as="the KL divergence from Q to P: the sum of P times the log of the "
            "ratio P over Q",
   worked="It is not symmetric — swapping P and Q gives a different number — "
          "so it is a discrepancy, not a distance. t-SNE minimises it between "
          "the neighbour distributions in high and low dimensions.",
   why="It is the objective t-SNE optimises, and log loss is a KL divergence "
       "in disguise.",
   symbols=("D_{KL}",))

_M("inverse_document_frequency", "TF-IDF",
   plain="Weigh each word by how often it appears in this document, damped by "
         "how common it is across all documents.",
   formula=r"\mathrm{tfidf}(t,d) = \mathrm{tf}(t,d) \times "
           r"\log\!\frac{N}{\mathrm{df}(t)}",
   reads_as="term frequency in the document, times the log of N over the "
            "document frequency",
   worked="'the' appears in all 1000 documents: idf = ln(1000/1000) = 0, so "
          "it is erased. A word in 100 of them: ln(10) = 2.30, so its counts "
          "are multiplied by 2.3.",
   why="It is why a bag of words can beat a naive word count without a "
       "stopword list: the common words delete themselves.",
   symbols=(r"\mathrm{tf}", r"\mathrm{df}"),
   also=("tfidf", "term_frequency"))

_M("moving_average", "The moving average",
   plain="The mean of the last k values, recomputed at every step, which "
         "smooths a jumpy series.",
   formula=r"\mathrm{MA}_t = \frac{1}{k}\sum_{j=0}^{k-1} x_{t-j}",
   reads_as="the moving average at time t is the mean of the last k values up "
            "to and including t",
   worked="A 7-day average of daily sales flattens the weekend dip. Note it "
          "must only look backwards: including x at time t+1 would be a "
          "lookahead and would leak the future into your features.",
   why="It is the standard time-series feature, and the standard place "
       "beginners accidentally leak.",
   also=("rolling_window", "rolling_mean"))

_M("bootstrap_sampling", "Bootstrapping",
   plain="Resample your own data with replacement, many times, and watch how "
         "much the answer moves.",
   formula=r"\hat{\theta}^{*(b)} = f\!\left(X^{*(b)}\right), \; b=1\dots B",
   reads_as="for each of B resamples, recompute the statistic on that resample",
   worked="Draw 4200 rows from your 4200 rows with replacement — some rows "
          "appear twice, about 37% not at all — score the model, and repeat "
          "1000 times. The middle 95% of those scores is a confidence "
          "interval, with no formula required.",
   why="It is where the bagging in random forests comes from, and the "
       "simplest honest way to put error bars on any metric.",
   symbols=(r"\hat{\theta}",), also=("bootstrap", "resampling", "bagging"))

_M("bias_variance_decomposition", "The bias-variance decomposition",
   plain="Expected error splits into three parts: being systematically wrong, "
         "being unstable, and noise nobody can remove.",
   formula=r"\mathbb{E}\big[(y-\hat{f})^2\big] = "
           r"\underbrace{\mathrm{Bias}^2}_{\text{too simple}} + "
           r"\underbrace{\mathrm{Var}}_{\text{too twitchy}} + "
           r"\underbrace{\sigma^2}_{\text{irreducible}}",
   reads_as="expected squared error equals bias squared, plus variance, plus "
            "irreducible noise",
   worked="A straight line through curved data has high bias and low "
          "variance: wrong in the same way every time. A deep tree has low "
          "bias and high variance: it would draw a different shape from a "
          "different sample. The third term stays whatever you do.",
   why="It is the reason underfitting and overfitting are two ends of one "
       "dial rather than two unrelated bugs.",
   symbols=(r"\mathbb{E}", r"\sigma^2"),
   also=("bias_variance", "irreducible_error", "expectation"))

_M("maximum_likelihood", "Maximum likelihood",
   plain="Choose the parameters that make the data you actually observed as "
         "probable as possible.",
   formula=r"\hat{\theta} = \arg\max_{\theta} \; \prod_{i=1}^{n} "
           r"P(x_i \mid \theta) = \arg\max_{\theta} \sum_{i=1}^{n} "
           r"\log P(x_i \mid \theta)",
   reads_as="theta-hat is the theta maximising the product of the "
            "probabilities, equivalently the sum of their logs",
   worked="A coin lands heads 7 times in 10. The p making that most likely is "
          "0.7 — which is just the sample proportion, derived rather than "
          "assumed. Logs turn the product into a sum, which is both "
          "numerically safer and easier to differentiate.",
   why="It is where loss functions come from. Squared error is maximum "
       "likelihood under normal noise; log loss is maximum likelihood under "
       "a coin flip. They were never arbitrary choices.",
   symbols=(r"\theta", r"\prod"),
   also=("likelihood", "log_likelihood", "likelihood_function", "loss_derivation"))

_M("indicator_variable", "An indicator (dummy) variable",
   plain="A column that is 1 when a condition holds and 0 when it does not.",
   formula=r"\mathbb{1}[\text{condition}] = "
           r"\begin{cases}1 & \text{if true}\\ 0 & \text{otherwise}\end{cases}",
   reads_as="the indicator of a condition: one if true, zero otherwise",
   worked="contract = 'month-to-month' becomes a column of 1s and 0s. Its "
          "coefficient then reads as: the effect of being month-to-month "
          "rather than the reference category.",
   why="It is what one-hot encoding produces, and the dummy trap is what "
       "happens when you keep every level instead of dropping one.",
   symbols=(r"\mathbb{1}",), also=("dummy_variable", "one_hot"))

_M("expected_value", "Expected value",
   plain="The long-run average: each outcome weighted by how likely it is.",
   formula=r"\mathbb{E}[X] = \sum_i x_i \, P(x_i)",
   reads_as="the expectation of X equals the sum of each value times its "
            "probability",
   worked="A flagged customer is worth 200 if the flag is right and costs 20 "
          "if wrong. At a 30% hit rate: 0.3 x 200 - 0.7 x 20 = 60 - 14 = 46 "
          "per flag. Positive, so flagging pays.",
   why="It is how a cost matrix turns a probability into a decision, which is "
       "the week-10 lesson on choosing a threshold by money rather than by F1.",
   symbols=(r"\mathbb{E}",), also=("expectation", "expected_value_calc"))

_M("roc_curve", "The ROC curve",
   plain="Slide the threshold from 1 down to 0 and plot, at each stop, how "
         "many real positives you caught against how many negatives you "
         "wrongly flagged.",
   formula=r"\mathrm{ROC}: \big(\mathrm{FPR}(t),\; \mathrm{TPR}(t)\big)"
           r"\quad \text{for every threshold } t",
   reads_as="for every threshold t, plot the false positive rate against the "
            "true positive rate",
   worked="The diagonal line from corner to corner is random guessing. A "
          "curve bulging towards the top-left is a model that catches "
          "positives without dragging negatives along. The area under it is "
          "the AUC.",
   why="It shows every threshold at once, so you can pick the operating point "
       "from the picture rather than accepting sklearn's default of 0.5.",
   symbols=("TPR", "FPR"), also=("roc",))

_M("threshold", "The classification threshold",
   plain="The cut-off above which a predicted probability is called a yes. "
         "It defaults to 0.5, and 0.5 is very often the wrong number.",
   formula=r"\hat{y} = \begin{cases}1 & \hat{p} \ge t\\ 0 & \hat{p} < t"
           r"\end{cases}",
   reads_as="predict one when the predicted probability is at or above t, "
            "otherwise zero",
   worked="At t = 0.5 you might flag 200 customers and catch 60% of leavers. "
          "Drop to t = 0.3 and you flag 500 and catch 85%. Nothing about the "
          "model changed — only where you cut.",
   why="Precision, recall, accuracy and F1 all move when you move it, while "
       "AUC does not. Knowing which numbers depend on it is most of reading "
       "a metrics table correctly.",
   symbols=("t", r"\hat{p}"), also=("decision_threshold",))

_M("accuracy_definition", "Accuracy",
   plain="The share of predictions that were right. The simplest metric and "
         "the easiest to be fooled by.",
   formula=r"\text{accuracy} = \frac{TP + TN}{TP + TN + FP + FN}",
   reads_as="the correct predictions, over all predictions",
   worked="If 3 customers in 100 leave, predicting 'nobody leaves' scores 97% "
          "accuracy and has caught nobody. That is why the course reaches "
          "precision and recall on the very next day.",
   why="It is the right metric only when the classes are balanced and both "
       "mistakes cost the same, which is rarer than it sounds.",
   also=("accuracy",))

_M("polynomial", "A polynomial",
   plain="A sum of powers of x. Degree 1 is a straight line, degree 2 a "
         "parabola, and higher degrees bend more the further you go.",
   formula=r"f(x) = \beta_0 + \beta_1 x + \beta_2 x^2 + \dots + \beta_d x^d",
   reads_as="beta-zero, plus beta-one x, plus beta-two x squared, and so on "
            "up to degree d",
   worked="Degree 2 on a single feature turns [x] into [x, x^2], and a linear "
          "model fitted on those two columns draws a curve. The model is "
          "still linear in its coefficients — that is the trick.",
   why="It is how a straight-line model fits a curve, and how you generate "
       "overfitting on demand: raise the degree and watch the test score "
       "collapse.",
   symbols=("x^d",), also=("polynomial_degree", "polynomial_terms", "degree",
                           "power", "basis_expansion"))

_M("probability_density", "A probability density",
   plain="For values that vary continuously, probability lives in areas under "
         "a curve, not in the height of the curve at a point.",
   formula=r"P(a \le X \le b) = \int_a^b f(x)\,dx, \qquad "
           r"\int_{-\infty}^{\infty} f(x)\,dx = 1",
   reads_as="the probability X lands between a and b is the area under f "
            "between a and b; the whole area is one",
   worked="The chance a house is exactly 1500.000 sq ft is zero — there are "
          "infinitely many possible values. The chance it is between 1400 and "
          "1600 is a real number, and it is the area under the curve across "
          "that strip.",
   why="It is why a histogram's y-axis can read above 1 without anything "
       "being wrong, and what a KDE plot is drawing.",
   symbols=(r"\int", "f(x)"), also=("density", "pdf", "probability_distribution"))

_M("kernel_function", "A kernel",
   plain="A shortcut that measures similarity between two rows as if you had "
         "expanded them into a far larger feature space, without ever "
         "building it.",
   formula=r"K(\mathbf{x},\mathbf{z}) = \phi(\mathbf{x})\cdot\phi(\mathbf{z})",
   reads_as="K of x and z equals the dot product of the two points after "
            "mapping them with phi",
   worked="The RBF kernel is exp(-gamma times the squared distance between "
          "the two points): 1 when they sit on top of each other, falling "
          "towards 0 as they separate. An SVM using it can draw a curved "
          "boundary while still, underneath, fitting a hyperplane.",
   why="It is what lets an SVM separate classes no straight line can, and the "
       "reason the same algorithm handles both easy and awkward shapes.",
   symbols=("K", r"\phi"), also=("kernel", "rbf", "kernel_trick"))

_M("derivative_of_sigmoid", "The derivative of the sigmoid",
   plain="How fast the sigmoid is rising at a point. It is largest in the "
         "middle and nearly zero at both ends.",
   formula=r"\sigma'(z) = \sigma(z)\big(1 - \sigma(z)\big)",
   reads_as="sigma prime of z equals sigma of z times one minus sigma of z",
   worked="At z = 0, sigma is 0.5, so the slope is 0.5 x 0.5 = 0.25 — the "
          "steepest it ever gets. At z = 6, sigma is 0.9975, so the slope is "
          "about 0.0025. Multiply a handful of those together through several "
          "layers and the gradient has effectively vanished.",
   why="It is the arithmetic behind the vanishing gradient problem, and "
       "therefore the reason ReLU replaced the sigmoid in hidden layers.",
   symbols=(r"\sigma'",), also=("vanishing_gradient",))

_M("random_seed", "The random seed",
   plain="A starting number for the pseudo-random generator. Same seed, same "
         "'random' choices, every run.",
   formula=r"\text{seed} \Rightarrow \text{a fixed sequence } "
           r"r_1, r_2, r_3, \dots",
   reads_as="fixing the seed fixes the whole sequence of random numbers",
   worked="train_test_split(..., random_state=42) gives you the same split "
          "tomorrow. Without it, your score moves every run and you cannot "
          "tell an improvement from noise.",
   why="It is the difference between a result you can show someone and a "
       "number you happened to see once.",
   also=("pseudorandomness", "sampling", "determinism"))

_M("paired_test", "A paired comparison",
   plain="Compare two models on the same folds rather than separately, so the "
         "difficulty of each fold cancels out.",
   formula=r"d_i = a_i - b_i, \qquad t = \frac{\bar{d}}"
           r"{\mathrm{SE}(d)}",
   reads_as="take the per-fold difference, then test whether its mean "
            "differs from zero",
   worked="Model A scores 0.81, 0.74, 0.88; model B scores 0.79, 0.71, 0.86 "
          "on those same folds. The differences are +0.02, +0.03, +0.02 — "
          "small but consistent, which is far more convincing than the gap "
          "between the two averages alone.",
   why="It is why the same folds must be used for every model you compare, "
       "and why a model that 'won' on different splits has won nothing.",
   symbols=("d_i", r"\bar{d}"), also=("same_folds", "paired_comparison"))

_M("marginalisation", "Marginalising",
   plain="Getting the probability of one thing by adding up every way it "
         "could happen across the other thing.",
   formula=r"P(A) = \sum_b P(A \cap B{=}b) = \sum_b P(A \mid B{=}b)\,P(B{=}b)",
   reads_as="P of A equals the sum, over every value b, of P of A given b, "
            "times P of b",
   worked="Overall churn = churn among month-to-month customers times their "
          "share, plus churn among one-year customers times theirs, plus the "
          "same for two-year. Each group weighted by how common it is.",
   why="It is the denominator in Bayes' theorem, and the reason a base rate "
       "can differ so sharply from any single group's rate.",
   symbols=(r"\sum_b",), also=("marginal", "law_of_total_probability"))

_M("psi", "The population stability index",
   plain="How far a feature's distribution today has drifted from what it was "
         "at training time.",
   formula=r"\mathrm{PSI} = \sum_i (a_i - e_i)\ln\!\frac{a_i}{e_i}",
   reads_as="the sum over bins of actual minus expected, times the log of "
            "actual over expected",
   worked="The rough convention: under 0.1 is stable, 0.1 to 0.25 deserves a "
          "look, over 0.25 means the incoming data no longer resembles what "
          "the model was trained on.",
   why="It is the standard drift alarm, and drift is the usual reason a model "
       "that worked in March is quietly wrong by September.",
   also=("population_stability", "distribution_distance"))


# ---------------------------------------------------------------------------
# The symbol glossary
# ---------------------------------------------------------------------------
# Most of the time the blocker is not the idea, it is that nobody said the
# squiggle's name out loud.

@dataclass(frozen=True)
class Symbol:
    glyph: str
    latex: str
    name: str
    says: str            # what it does, in one line


SYMBOLS: tuple[Symbol, ...] = (
    Symbol("Σ", r"\sum", "sigma (capital)", "Add all of these up. A for-loop."),
    Symbol("Π", r"\prod", "pi (capital)", "Multiply all of these together."),
    Symbol("μ", r"\mu", "mu", "The mean — usually of the true population."),
    Symbol("σ", r"\sigma", "sigma", "Standard deviation. Squared, it is variance."),
    Symbol("x̄", r"\bar{x}", "x-bar", "The mean of the sample you actually have."),
    Symbol("ŷ", r"\hat{y}", "y-hat", "A predicted value. The hat means estimated."),
    Symbol("θ", r"\theta", "theta", "Whatever parameters the model is learning."),
    Symbol("β", r"\beta", "beta", "A coefficient: one feature's weight."),
    Symbol("α", r"\alpha", "alpha", "Regularisation strength, or a significance "
                                    "threshold. Context decides."),
    Symbol("λ", r"\lambda", "lambda", "Also regularisation strength; also an "
                                      "eigenvalue."),
    Symbol("η", r"\eta", "eta", "The learning rate: how big a step to take."),
    Symbol("ε", r"\epsilon", "epsilon", "Noise, or a tiny tolerance."),
    Symbol("∂", r"\partial", "partial", "Partial derivative: slope in one "
                                        "variable, others held still."),
    Symbol("∇", r"\nabla", "nabla, or grad", "The vector of every partial "
                                             "derivative at once."),
    Symbol("∈", r"\in", "is in", "Belongs to this set or group."),
    Symbol("∑ᵢ", r"\sum_i", "sum over i", "Add up over every row, indexed by i."),
    Symbol("|x|", r"|x|", "absolute value", "Size, sign discarded."),
    Symbol("‖x‖", r"\|x\|", "norm of x", "The length of a vector."),
    Symbol("ᵀ", r"^\top", "transpose", "Flip a matrix's rows and columns."),
    Symbol("⁻¹", r"^{-1}", "inverse", "The matrix that undoes this one."),
    Symbol("E[X]", r"\mathbb{E}[X]", "expectation of X", "The long-run average."),
    Symbol("P(A|B)", r"P(A \mid B)", "P of A given B",
           "The chance of A once B is known."),
    Symbol("∼", r"\sim", "is distributed as", "Follows this distribution."),
    Symbol("≈", r"\approx", "approximately", "Close enough for the point being made."),
    Symbol("∝", r"\propto", "proportional to", "Rises in step with, up to a constant."),
    Symbol("argmin", r"\arg\min", "arg-min", "The input that minimises, not the "
                                             "minimum itself."),
    Symbol("ln", r"\ln", "natural log", "Log to base e. The default log here."),
    Symbol("e", r"e", "e (Euler's number)",
           "2.718..., the base of the natural log."),
    Symbol("H₀", r"H_0", "H-nought", "The null hypothesis: nothing real is "
                                     "happening."),
    Symbol("ρ", r"\rho", "rho", "A correlation coefficient."),
    Symbol("Σ (matrix)", r"\Sigma", "sigma (capital, bold context)",
           "The covariance matrix — not a sum. Position tells you which."),
)

SYMBOL_INDEX: dict[str, Symbol] = {}
for _s in SYMBOLS:
    SYMBOL_INDEX[_s.glyph] = _s
    SYMBOL_INDEX[_s.latex] = _s
    SYMBOL_INDEX[_s.name.split(" (")[0]] = _s


# ---------------------------------------------------------------------------
# Lookup
# ---------------------------------------------------------------------------

_NORMALISE = re.compile(r"[^a-z0-9]+")
# Stripped before matching, so "what does sigma mean?" finds the sigma entry.
_FILLER = {"what", "whats", "is", "are", "the", "a", "an", "does", "do", "mean",
           "means", "explain", "tell", "me", "about", "how", "works", "work",
           "formula", "for", "of", "in", "ml", "please", "again", "this"}


def _key(term: str) -> str:
    return _NORMALISE.sub("_", term.strip().lower()).strip("_")


def _core_words(term: str) -> str:
    words = [w for w in _key(term).split("_") if w and w not in _FILLER]
    return "_".join(words)


# Words with a plain meaning of their own in this subject. A one-word query
# from this set may not fuzzy-match a longer entry name: "model" is not
# shorthand for "linear model", and letting it match one put a day-22 equation
# into day 1's prompt — and into the Math Helper's answer for "model".
_CORE_VOCABULARY = {
    "model", "models", "data", "feature", "features", "target", "targets",
    "prediction", "predictions", "predict", "fit", "split", "value", "values",
    "error", "errors", "score", "scores", "test", "train", "training",
    "example", "examples", "row", "rows", "column", "columns", "label",
    "labels", "input", "output", "learning", "supervised", "unsupervised",
}


def _names() -> list[tuple[str, Entry]]:
    names: list[tuple[str, Entry]] = []
    for entry in BANK.values():
        names.append((entry.key, entry))
        names.append((_key(entry.title), entry))
        for alias in entry.also:
            names.append((_key(alias), entry))
    return names


def lookup_exact(term: str) -> Entry | None:
    """A written entry for this exact key, alias or title. No guessing.

    Used wherever a wrong answer is worse than no answer — deciding what a
    lesson's prompt says the day's mathematics is, and which terms to offer as
    buttons. The forgiving matcher below is for a person typing a question,
    where a near miss is helpful; here a near miss teaches the wrong thing.
    """
    key = _key(term)
    if key in BANK:
        return BANK[key]
    for name, entry in _names():
        if name and name == key:
            return entry
    return None


def lookup(term: str) -> Entry | None:
    """Find a bank entry by key, alias, or the question a human would type.

    Three passes, narrowing: the exact key, then the question with its filler
    words removed, then a containment check either way. The last one matters
    because people type "r squared formula" and "what is the r2 score", and a
    glossary that only answers its own spelling is not a glossary.
    """
    key = _key(term)
    if key in BANK:
        return BANK[key]

    core = _core_words(term)
    if core and core in BANK:
        return BANK[core]

    names = _names()
    for name, entry in names:
        if name and (name == key or name == core):
            return entry

    # A Greek letter is a symbol question, not a term question. Without this
    # guard "theta" matches the alias "eta" by containment and cheerfully
    # explains the learning rate, which is the wrong answer delivered
    # confidently — the worst kind.
    if symbol_for(term) is not None:
        return None

    # Containment, but only on whole words: "r squared formula" should find
    # "r squared", while "theta" must not find "eta".
    target = [w for w in core.split("_") if w]

    # A single core word never widens into a longer entry. Without this,
    # "model" matched the alias "linear_model" and answered with a linear
    # equation — on day 1, to a reader who has met neither.
    if len(target) == 1 and target[0] in _CORE_VOCABULARY:
        return None

    for name, entry in sorted(names, key=lambda p: -len(p[0])):
        words = [w for w in name.split("_") if w]
        if not words or not target:
            continue
        if _sublist(words, target) or _sublist(target, words):
            return entry
    return None


def _sublist(needle: list[str], haystack: list[str]) -> bool:
    n = len(needle)
    return n > 0 and any(haystack[i:i + n] == needle
                         for i in range(len(haystack) - n + 1))


def symbol_for(text: str) -> Symbol | None:
    """Find a symbol by its glyph, its LaTeX, or its spoken name.

    Matching is done on the normalised form so that "y hat", "y-hat" and
    "yhat" all land on the same entry — someone who cannot type a circumflex
    is exactly the person asking.
    """
    text = text.strip()
    if text in SYMBOL_INDEX:
        return SYMBOL_INDEX[text]
    wanted = _key(text)
    if not wanted:
        return None
    for symbol in SYMBOLS:
        names = {_key(symbol.name), _key(symbol.name.split(" (")[0]),
                 _key(symbol.glyph), _key(symbol.latex)}
        if wanted in {n for n in names if n}:
            return symbol
    return None


def offer_for(topic: dict[str, Any]) -> list[tuple[str, str, bool]]:
    """The clickable list under a lesson: (key, label, is_in_the_bank).

    Every term the day's lesson leans on, offered before it is needed, because
    formulating the question is itself a barrier when the notation is the
    thing you do not know.
    """
    out: list[tuple[str, str, bool]] = []
    seen: set[str] = set()
    for key in list(topic.get("maths", ())) + list(topic.get("concepts", ())):
        entry = lookup_exact(key)
        # Deduplicate on the answer, not the key: "auc" and
        # "area_under_curve" are the same question asked twice.
        marker = entry.key if entry else key
        if marker in seen:
            continue
        seen.add(marker)
        out.append((key, entry.title if entry else concepts.label(key),
                    entry is not None))
    return out


def ladder(term: str, day: int) -> list[dict[str, Any]]:
    """What this term rests on, newest first, marked by whether it was taught.

    If the real gap is two steps down, the answer should say so rather than
    explaining one unfamiliar thing with another.
    """
    out: list[dict[str, Any]] = []
    for idea in concepts.chain_for(term, depth=2)[:6]:
        entry = lookup(idea.key)
        out.append({
            "key": idea.key,
            "label": entry.title if entry else idea.label,
            "day": idea.day,
            "in_bank": entry is not None,
            "already_taught": idea.day <= day,
        })
    return out


# ---------------------------------------------------------------------------
# Falling back to the model
# ---------------------------------------------------------------------------

SYSTEM = (
    "You explain mathematics to an adult who is capable but was badly taught, "
    "and who is learning machine learning. They have said plainly that maths "
    "is their weak point. They are not stupid and they do not want to be "
    "talked down to; they want the thing actually explained.\n\n"
    "Hard rules:\n"
    "1. Never explain one unfamiliar term with another. If you must use a "
    "second technical word, define it in the same breath.\n"
    "2. Name every symbol out loud. Not 'sigma', but 'the Greek letter sigma, "
    "which here means add all of these up'.\n"
    "3. Always give a worked example with small numbers the reader can check "
    "in their head. Never leave the example abstract.\n"
    "4. Say why the course needs it — which lesson it unlocks.\n"
    "5. No apologising, no 'don't worry', no 'this is actually simple'. Just "
    "explain it.\n"
    "6. British spelling. Plain sentences. No emoji."
)


def prompt_for(term: str, topic: dict[str, Any] | None = None) -> str:
    day = int((topic or {}).get("day", 91))
    known = [i.label for i in concepts.known_by(day, "maths")][-40:]
    parts = [
        f"Explain: {term}",
        "",
        f"The reader is on day {day} of a 91-day machine learning course.",
    ]
    if topic:
        parts.append(f"Today's lesson is '{topic.get('title', '')}' — "
                     f"{topic.get('summary', '')}")
    if known:
        parts.append("Mathematics they have already been shown in this course: "
                     + ", ".join(known))
    parts += [
        "",
        "Return JSON with exactly these keys:",
        '  "title"    - what to call it, 2 to 5 words',
        '  "plain"    - what it is, in two or three sentences, using no '
        'symbols at all',
        '  "formula"  - the formula as LaTeX, no dollar signs, or "" if it '
        'has none',
        '  "reads_as" - that formula spoken aloud, the way you would read it '
        'to someone over the phone',
        '  "worked"   - a worked example with small specific numbers, showing '
        'the arithmetic',
        '  "why"      - one or two sentences on what it unlocks in this course',
        '  "symbols"  - a list of any symbols used, each as '
        '{"glyph": "...", "name": "...", "says": "..."}',
    ]
    return "\n".join(parts)


@dataclass
class Explanation:
    term: str
    title: str
    plain: str
    formula: str = ""
    reads_as: str = ""
    worked: str = ""
    why: str = ""
    symbols: list[dict[str, str]] = field(default_factory=list)
    source: str = "bank"          # bank | model | cache
    ladder: list[dict[str, Any]] = field(default_factory=list)
    error: str = ""

    @property
    def ok(self) -> bool:
        return bool(self.plain) and not self.error


def _from_entry(entry: Entry, term: str, day: int) -> Explanation:
    glossary = []
    for latex in entry.symbols:
        symbol = SYMBOL_INDEX.get(latex)
        if symbol:
            glossary.append({"glyph": symbol.glyph, "name": symbol.name,
                             "says": symbol.says})
        else:
            glossary.append({"glyph": latex, "name": "", "says": ""})
    return Explanation(
        term=term, title=entry.title, plain=entry.plain, formula=entry.formula,
        reads_as=entry.reads_as, worked=entry.worked, why=entry.why,
        symbols=glossary, source="bank", ladder=ladder(entry.key, day),
    )


def cache_id(term: str, day: int) -> str:
    digest = hashlib.sha256(f"{_key(term)}|{day}".encode()).hexdigest()[:20]
    return f"math-{digest}"


def explain(term: str, *, topic: dict[str, Any] | None = None,
            settings: dict[str, str] | None = None,
            use_model: bool = True) -> Explanation:
    """Answer a maths question: from the bank, from the cache, or from the model.

    The bank is checked first on purpose. Those answers are written, verified
    and instant, and the ones most likely to be asked are all in it.
    """
    day = int((topic or {}).get("day", 91))

    entry = lookup(term)
    if entry:
        return _from_entry(entry, term, day)

    symbol = symbol_for(term)
    if symbol:
        return Explanation(
            term=term, title=symbol.name, plain=symbol.says,
            formula=symbol.latex, reads_as=symbol.name,
            why="It appears in the formulas for this part of the course.",
            symbols=[{"glyph": symbol.glyph, "name": symbol.name,
                      "says": symbol.says}],
            source="bank",
        )

    from . import db                              # noqa: PLC0415 - optional

    cached_id = cache_id(term, day)
    try:
        cached = db.get_math(cached_id)
    except Exception:                             # the DB is not load-bearing here
        cached = None
    if cached:
        body = cached["body"]
        return Explanation(term=term, source="cache",
                           ladder=ladder(_key(term), day),
                           **{k: body.get(k, "") for k in
                              ("title", "plain", "formula", "reads_as",
                               "worked", "why")},
                           symbols=body.get("symbols", []))

    if not use_model:
        return Explanation(term=term, title=term, plain="", source="none",
                           error="Not in the offline glossary.")

    from . import llm                             # noqa: PLC0415 - heavy

    result = llm.complete_json(SYSTEM, prompt_for(term, topic),
                               settings=settings, tier="strong",
                               max_tokens=1200)
    if not result:
        return Explanation(term=term, title=term, plain="",
                           error=result.error or "The model did not answer.")

    data = llm.extract_json(result.text)
    if isinstance(data, list) and data:
        data = data[0]
    if isinstance(data, dict) and "items" in data:
        items = data["items"]
        data = items[0] if isinstance(items, list) and items else data
    if not isinstance(data, dict):
        return Explanation(term=term, title=term, plain="",
                           error="The model replied, but not in the expected "
                                 "format. Try asking again.")

    body = {
        "title": str(data.get("title") or term),
        "plain": str(data.get("plain") or ""),
        "formula": str(data.get("formula") or ""),
        "reads_as": str(data.get("reads_as") or ""),
        "worked": str(data.get("worked") or ""),
        "why": str(data.get("why") or ""),
        "symbols": [s for s in (data.get("symbols") or [])
                    if isinstance(s, dict)],
    }
    try:
        db.save_math(cached_id, term, day, body,
                     provider=result.provider, model=result.model)
    except Exception:                             # caching is a saving, not a duty
        pass

    return Explanation(term=term, source="model", ladder=ladder(_key(term), day),
                       symbols=body.pop("symbols"), **body)


def bank_size() -> dict[str, int]:
    unique = {id(e) for e in BANK.values()}
    return {"entries": len(unique), "lookups": len(BANK),
            "symbols": len(SYMBOLS)}


def coverage() -> dict[str, Any]:
    """Which of the curriculum's maths terms the written bank already answers.

    The gap is not a bug — the model fills it — but it is worth seeing, since
    every term that moves into the bank is one more question answered with no
    key, no cost and no chance of invention.
    """
    covered, missing = [], []
    for key in concepts.MATHS:
        (covered if lookup(key) else missing).append(key)
    return {"covered": sorted(covered), "missing": sorted(missing),
            "share": round(len(covered) / max(1, len(concepts.MATHS)), 3)}


def to_json(explanation: Explanation) -> str:
    return json.dumps(explanation.__dict__, indent=2)
