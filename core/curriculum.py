"""The 91-day machine learning plan.

Thirteen weeks at roughly two and a half hours a day. It starts at what a model
is and what "learning" means, and ends at neural networks, with every day
carrying three things: the theory, a picture of what the theory means, and code
run against a real dataset with every step explained.

Two rules shaped the ordering.

Nothing is used before it is taught. Cross-validation does not appear in a
week-2 code cell, regularisation does not appear before overfitting has been
seen. core/concepts.py maps every idea to the day it arrives and the generator
is held to it, the same constraint that fixed the Python mentor after it opened
day 1 with a function definition.

Evaluation comes before models. Weeks 1 and 2 cover the train/test split,
overfitting and the metrics before a single algorithm is fitted, because a
model you cannot evaluate is a model you cannot learn from — and because
almost every serious mistake in applied ML is an evaluation mistake, not an
algorithm one.
"""
from __future__ import annotations

from typing import Any

LESSON, PROJECT, MILESTONE, CAPSTONE = "lesson", "project", "milestone", "capstone"
FOUND, SUP, UNSUP, ADV = "foundations", "supervised", "unsupervised", "advanced"

PLAN_WEEKS = 13
TOPICS: list[dict[str, Any]] = []


def _t(slug: str, title: str, track: str, week: int, day: int,
       difficulty: str, minutes: int, summary: str, objectives: list[str],
       real_world: list[str], *, dataset: str = "", concepts: tuple[str, ...] = (),
       maths: tuple[str, ...] = (), prereqs: tuple[str, ...] = (),
       kind: str = LESSON) -> dict[str, Any]:
    topic = {
        "slug": slug, "title": title, "track": track, "week": week, "day": day,
        "difficulty": difficulty, "minutes": minutes, "summary": summary,
        "objectives": objectives, "real_world": real_world,
        "prereqs": list(prereqs), "kind": kind,
        "dataset": dataset,
        # The ideas this day introduces. concepts.py turns these into the
        # horizon that stops a lesson reaching forward.
        "concepts": list(concepts),
        # The mathematics this day leans on, so the Math Helper knows what to
        # offer before the student has to go looking.
        "maths": list(maths),
    }
    TOPICS.append(topic)
    return topic


B, I, A = "beginner", "intermediate", "advanced"

# ---------------------------------------------------------------------------
# Week 1 — What learning from data actually means
# ---------------------------------------------------------------------------
_t("ml-what-is-a-model", "What a Model Actually Is", FOUND, 1, 1, B, 140,
   "Before any algorithm: what it means for a machine to learn from examples, "
   "and what it cannot do.",
   ["Say what a model is in one sentence without using the word 'learn'",
    "Tell a supervised problem from an unsupervised one and from neither",
    "Explain why more data is not always the answer",
    "Name three problems machine learning is the wrong tool for"],
   ["Every model in production started as somebody deciding a task was a "
    "prediction problem — often wrongly",
    "'Can we predict X?' is answerable before any code is written",
    "Knowing when not to use ML is most of what a senior practitioner does"],
   dataset="customer_churn", concepts=("model", "features", "target",
                                       "supervised", "unsupervised", "prediction"))

_t("ml-data-first-look", "Looking at Data Before Touching It", FOUND, 1, 2, B, 150,
   "Loading a dataset and reading what it is telling you, before any model.",
   ["Load a CSV and describe its shape, types and ranges",
    "Spot a column that is the wrong type",
    "Find missing values and say whether they are random",
    "Write down three questions the data cannot answer"],
   ["The first hour of any real project, every time",
    "Most modelling failures are visible in the first look and ignored",
    "A column of 9999s is how a sensor says 'broken'"],
   dataset="customer_churn", prereqs=("ml-what-is-a-model",),
   concepts=("dataframe", "dtypes", "describe", "missing_values", "distribution"),
   maths=("mean", "median", "percentile", "standard_deviation"))

_t("ml-distributions", "Distributions, Spread and Skew", FOUND, 1, 3, B, 150,
   "What a column's shape tells you, and why the mean is so often the wrong "
   "summary.",
   ["Read a histogram and say what it implies for modelling",
    "Explain when the median beats the mean and why",
    "Recognise a skewed target and know what it will do to a model",
    "Compute and interpret a standard deviation by hand"],
   ["House prices, incomes and waiting times are all skewed, and all get "
    "modelled badly by people who did not check",
    "A/B test results are read off distributions, not averages"],
   dataset="house_prices", prereqs=("ml-data-first-look",),
   concepts=("histogram", "skew", "variance", "outlier", "log_transform"),
   maths=("variance", "standard_deviation", "skewness", "logarithm"))

_t("ml-relationships", "How Two Columns Relate", FOUND, 1, 4, B, 150,
   "Correlation, what it measures, and the many things it does not.",
   ["Read a scatter plot before computing anything",
    "Compute Pearson correlation and state its assumptions",
    "Give a real example of correlation without causation",
    "Spot a non-linear relationship that correlation reports as zero"],
   ["Feature selection starts here",
    "Two near-identical columns will wreck a linear model's coefficients",
    "Every spurious-correlation headline is this lesson unlearned"],
   dataset="house_prices", prereqs=("ml-distributions",),
   concepts=("scatter", "correlation", "covariance", "causation", "collinearity"),
   maths=("covariance", "pearson_correlation", "linear_relationship"))

_t("ml-train-test", "The Split That Makes Learning Measurable", FOUND, 1, 5, B, 160,
   "Why a model is never judged on the data it learned from.",
   ["Explain what a test set is protecting against",
    "Split data correctly, including when classes are imbalanced",
    "Say what goes wrong when the split happens after preprocessing",
    "Describe a situation where a random split is the wrong split"],
   ["The single most common serious error in applied ML",
    "A model scored on its training data is a model with no evidence",
    "Time series and grouped data both break the naive split"],
   dataset="customer_churn", prereqs=("ml-relationships",),
   concepts=("train_test_split", "generalisation", "holdout", "stratify",
             "data_leakage"),
   maths=("sampling", "random_seed"))

_t("ml-first-model", "Your First Model, End to End", FOUND, 1, 6, B, 170,
   "A complete pipeline — load, split, fit, predict, score — with every line "
   "explained.",
   ["Run a full pipeline from CSV to a score you can defend",
    "Explain what .fit() and .predict() actually do",
    "Say what the score means and what it does not",
    "Beat a baseline, and know why beating it matters"],
   ["This shape — load, split, fit, evaluate — never changes, however "
    "complicated the model gets",
    "A baseline is the first thing a reviewer asks for"],
   dataset="iris", prereqs=("ml-train-test",),
   concepts=("fit", "predict", "accuracy", "baseline", "estimator"),
   maths=("accuracy_definition",))

_t("proj-w1", "Project: A Defensible First Prediction", FOUND, 1, 7, B, 180,
   "Build and defend a complete first model on the churn data.",
   ["Produce a model, a score, and a baseline it beats",
    "Write down what you looked at before modelling",
    "State one thing the model cannot be trusted with"],
   ["This is what a first week on a real project produces"],
   dataset="customer_churn", prereqs=("ml-first-model",), kind=PROJECT)

# ---------------------------------------------------------------------------
# Week 2 — Evaluation, before any more models
# ---------------------------------------------------------------------------
_t("ml-overfitting", "Overfitting, Seen Rather Than Defined", FOUND, 2, 8, B, 160,
   "Watching a model memorise, and learning to recognise it from the numbers "
   "alone.",
   ["Draw the training-versus-test curve from memory",
    "Diagnose overfitting from two numbers",
    "Explain why a perfect training score is bad news",
    "Name three ways to reduce it"],
   ["The reason your model does worse in production than in the notebook",
    "Every regularisation technique exists because of this lesson"],
   dataset="house_prices", prereqs=("ml-first-model",),
   concepts=("overfitting", "underfitting", "model_complexity", "capacity"),
   maths=("polynomial_degree",))

_t("ml-bias-variance", "Bias and Variance", FOUND, 2, 9, I, 160,
   "The decomposition that explains why every model choice is a trade.",
   ["State the decomposition and what each term means",
    "Place a given model on the bias-variance axis",
    "Explain why reducing one usually raises the other",
    "Say what irreducible error is and why no model beats it"],
   ["Why the answer to 'which model?' is always 'it depends'",
    "Ensembles are a direct attack on the variance term"],
   dataset="house_prices", prereqs=("ml-overfitting",),
   concepts=("bias", "model_variance", "irreducible_error",
                                     "learning_curve"),
   maths=("expectation", "mean_squared_error", "bias_variance_decomposition"))

_t("ml-cross-validation", "Cross-Validation", FOUND, 2, 10, I, 160,
   "Getting a trustworthy estimate from data you do not have enough of.",
   ["Explain k-fold in one paragraph and implement it",
    "Choose k and defend the choice",
    "Say why a single split can mislead badly on small data",
    "Recognise when cross-validation is itself invalid"],
   ["Every published model comparison rests on this",
    "Small medical and scientific datasets have nothing else"],
   dataset="wine", prereqs=("ml-bias-variance",),
   concepts=("k_fold", "cross_validation", "stratified_kfold", "cv_score"),
   maths=("mean_of_scores", "standard_error"))

_t("ml-classification-metrics", "When Accuracy Lies", FOUND, 2, 11, I, 170,
   "Precision, recall and the confusion matrix, on a problem where the two "
   "mistakes are not equal.",
   ["Build a confusion matrix by hand and read it",
    "Define precision and recall without looking them up",
    "Choose the right metric for a stated cost of error",
    "Explain why 99% accuracy can be worthless"],
   ["Fraud, disease screening and spam are all imbalanced, and accuracy is "
    "useless on all three",
    "A missed malignancy and a false alarm are not the same mistake"],
   dataset="breast_cancer", prereqs=("ml-cross-validation",),
   concepts=("confusion_matrix", "precision", "recall", "f1", "class_imbalance"),
   maths=("precision_formula", "recall_formula", "harmonic_mean"))

_t("ml-roc-thresholds", "Thresholds, ROC and What to Optimise", FOUND, 2, 12, I, 160,
   "A classifier outputs a probability. Choosing where to cut it is a "
   "decision, not a default.",
   ["Explain what moving the threshold trades away",
    "Read an ROC curve and say what AUC measures",
    "Pick a threshold from a stated business cost",
    "Say when AUC is the wrong summary"],
   ["The 0.5 default is almost never the right cut",
    "Credit and medical models are tuned on this curve, not on accuracy"],
   dataset="breast_cancer", prereqs=("ml-classification-metrics",),
   concepts=("threshold", "roc_curve", "auc", "precision_recall_curve"),
   maths=("true_positive_rate", "false_positive_rate", "area_under_curve"))

_t("ml-regression-metrics", "Measuring a Number, Not a Class", FOUND, 2, 13, I, 150,
   "MAE, RMSE and R-squared — what each one punishes, and which to report.",
   ["Compute all three by hand on five points",
    "Say when RMSE beats MAE and why",
    "Explain what R-squared is relative to",
    "Recognise an R-squared that is high for the wrong reason"],
   ["Forecasting error is money, and which error metric you chose decides "
    "which model wins",
    "A high R-squared on a time trend usually means nothing"],
   dataset="house_prices", prereqs=("ml-roc-thresholds",),
   concepts=("mae", "rmse", "r_squared", "residual"),
   maths=("absolute_error", "squared_error", "r_squared_formula"))

_t("proj-w2", "Project: An Evaluation Report", FOUND, 2, 14, I, 180,
   "Take last week's model and produce an honest evaluation of it.",
   ["Report cross-validated scores with their spread",
    "Choose and justify a metric for the churn problem",
    "Pick a threshold and state the cost assumption behind it"],
   ["This is the document that decides whether a model ships"],
   dataset="customer_churn", prereqs=("ml-regression-metrics",), kind=PROJECT)

# ---------------------------------------------------------------------------
# Week 3 — Cleaning and preparing real data
# ---------------------------------------------------------------------------
_t("ml-missing-data", "Missing Values and What They Mean", FOUND, 3, 15, I, 160,
   "Why a value is missing decides what you are allowed to do about it.",
   ["Tell missing-at-random from missing-for-a-reason",
    "Choose between dropping, filling and flagging, and defend it",
    "Show how a naive fill biases a result"],
   ["Dropping rows with missing data is how a sample quietly stops "
    "representing the population",
    "Medical and survey data are missing for reasons that matter"],
   dataset="customer_churn", prereqs=("proj-w2",),
   concepts=("imputation", "mcar", "mnar", "missing_indicator", "dropna"),
   maths=("conditional_probability",))

_t("ml-outliers", "Outliers, Errors and Genuine Extremes", FOUND, 3, 16, I, 150,
   "Telling a broken sensor from a real event, and why deleting both is wrong.",
   ["Find outliers by IQR and by z-score, and say when each fails",
    "Distinguish an impossible value from a rare true one",
    "Show the effect of one extreme point on a fitted line"],
   ["Fraud and equipment failure are outliers you must keep",
    "A negative age is an error; a very large house is not"],
   dataset="house_prices", prereqs=("ml-missing-data",),
   concepts=("iqr", "z_score", "winsorise", "robust_statistics"),
   maths=("quartile", "z_score_formula", "median_absolute_deviation"))

_t("ml-categorical", "Turning Categories into Numbers", FOUND, 3, 17, I, 160,
   "One-hot, ordinal and target encoding — and the trap in each.",
   ["Encode a categorical column three ways and compare",
    "Explain the dummy variable trap",
    "Handle a category that appears in test but not in training"],
   ["Every real dataset has text columns a model cannot read",
    "Target encoding leaks unless it is done inside the fold"],
   dataset="customer_churn", prereqs=("ml-outliers",),
   concepts=("one_hot", "ordinal_encoding", "dummy_trap", "unseen_category"),
   maths=("indicator_variable",))

_t("ml-scaling", "Scaling, and Which Models Care", FOUND, 3, 18, I, 150,
   "Standardisation and normalisation, and why some models collapse without "
   "them.",
   ["Standardise and normalise, and state the difference",
    "Name which model families need scaling and which do not",
    "Show a distance-based model failing on unscaled data"],
   ["A feature in rupees and one in years are not comparable to a distance",
    "Gradient descent crawls on unscaled features"],
   dataset="wine", prereqs=("ml-categorical",),
   concepts=("standardisation", "min_max", "robust_scaler", "distance_metric"),
   maths=("z_score_formula", "euclidean_distance"))

_t("ml-pipelines", "Pipelines, and Why Preprocessing Must Be Fitted", FOUND, 3, 19, I, 160,
   "The leak that hides in every hand-rolled preprocessing step.",
   ["Explain why a scaler fitted before the split leaks",
    "Build a pipeline that cannot leak",
    "Cross-validate a whole pipeline rather than a bare model"],
   ["This is the leak that makes a notebook score far higher than production",
    "Reviewers look for this first"],
   dataset="customer_churn", prereqs=("ml-scaling",),
   concepts=("pipeline", "column_transformer", "fit_transform", "preprocessing_leak"),
   maths=())

_t("ml-feature-engineering", "Making Features That Help", FOUND, 3, 20, I, 160,
   "Ratios, interactions and domain knowledge — where most real gains come "
   "from.",
   ["Build three features from domain reasoning and test whether they help",
    "Explain why a ratio can beat both its parts",
    "Show a feature that helps one model family and not another"],
   ["Feature work beats model choice on most tabular problems",
    "Price per square foot is a better feature than either column alone"],
   dataset="house_prices", prereqs=("ml-pipelines",),
   concepts=("feature_engineering", "interaction", "binning", "domain_knowledge"),
   maths=("ratio", "polynomial_terms"))

_t("proj-w3", "Milestone: A Clean, Leak-Free Pipeline", FOUND, 3, 21, I, 210,
   "Take the raw churn CSV to a model-ready pipeline that provably does not "
   "leak.",
   ["Handle every missing value with a stated reason",
    "Encode and scale inside a pipeline",
    "Show the score before and after, and explain the difference"],
   ["This artefact is the backbone of every project that follows"],
   dataset="customer_churn", prereqs=("ml-feature-engineering",), kind=MILESTONE)

# ---------------------------------------------------------------------------
# Week 4 — Linear regression, properly
# ---------------------------------------------------------------------------
_t("ml-linear-intuition", "The Line of Best Fit", SUP, 4, 22, I, 160,
   "What 'best' means, and why squared error rather than absolute.",
   ["Write the model equation and name every symbol",
    "Explain what the loss function is measuring",
    "Fit a line by hand on four points"],
   ["The foundation under most of statistics and half of ML",
    "Still the model that gets shipped when interpretability matters"],
   dataset="house_prices", prereqs=("proj-w3",),
   concepts=("linear_model", "coefficient", "intercept", "loss_function",
             "least_squares"),
   maths=("linear_equation", "squared_error", "summation", "argmin"))

_t("ml-gradient-descent", "Gradient Descent", SUP, 4, 23, I, 180,
   "How a model actually finds its parameters, one step at a time.",
   ["Explain what a gradient is, geometrically",
    "Implement gradient descent from scratch in ten lines",
    "Show what too large and too small a learning rate each do"],
   ["Every neural network is trained this way",
    "The learning rate is the first thing anyone tunes"],
   dataset="house_prices", prereqs=("ml-linear-intuition",),
   concepts=("gradient_descent", "learning_rate", "convergence", "epoch",
             "cost_surface"),
   maths=("derivative", "partial_derivative", "gradient", "chain_rule"))

_t("ml-multiple-regression", "More Than One Feature", SUP, 4, 24, I, 160,
   "What a coefficient means when other columns are present.",
   ["Interpret a coefficient as 'holding the others constant'",
    "Show how correlated features make coefficients unstable",
    "Explain why adding a feature always raises training R-squared"],
   ["Nearly every misreading of a regression result is this",
    "Economics and medicine argue about exactly this point"],
   dataset="house_prices", prereqs=("ml-gradient-descent",),
   concepts=("multiple_regression", "partial_effect", "multicollinearity",
             "adjusted_r_squared"),
   maths=("matrix_multiplication", "vector", "normal_equation"))

_t("ml-assumptions", "Checking the Assumptions", SUP, 4, 25, I, 160,
   "Residual plots, and what each shape is telling you to fix.",
   ["Read a residual plot and name the violation",
    "Detect heteroscedasticity and respond to it",
    "Show a fitted model that is wrong despite a good R-squared"],
   ["A model that passes on numbers and fails on residuals is broken",
    "Fanning residuals are why house prices get logged"],
   dataset="house_prices", prereqs=("ml-multiple-regression",),
   concepts=("residual_plot", "heteroscedasticity", "linearity_assumption",
             "normality_of_residuals"),
   maths=("residual", "variance_of_residuals"))

_t("ml-regularisation", "Ridge and Lasso", SUP, 4, 26, A, 170,
   "Deliberately making the fit worse so the model generalises better.",
   ["Explain the penalty term and what it does to coefficients",
    "Say why lasso can zero a coefficient and ridge cannot",
    "Tune the penalty by cross-validation"],
   ["The standard answer to wide data with many correlated columns",
    "Lasso is feature selection that happens during fitting"],
   dataset="breast_cancer", prereqs=("ml-assumptions",),
   concepts=("ridge", "lasso", "l1_penalty", "l2_penalty", "alpha",
             "shrinkage"),
   maths=("l1_norm", "l2_norm", "constrained_optimisation"))

_t("ml-polynomial", "Curves, and Where They Go Wrong", SUP, 4, 27, I, 150,
   "Fitting a non-linear relationship with a linear model, and overfitting on "
   "purpose to see it happen.",
   ["Fit polynomials of rising degree and plot the result",
    "Show the exact degree where test error turns upward",
    "Explain why extrapolating a polynomial is dangerous"],
   ["The clearest picture of overfitting anyone has drawn",
    "Physical and economic curves are fitted this way daily"],
   dataset="diabetes", prereqs=("ml-regularisation",),
   concepts=("polynomial_features", "degree", "extrapolation", "basis_expansion"),
   maths=("polynomial", "power"))

_t("proj-w4", "Project: Predicting House Prices", SUP, 4, 28, I, 200,
   "A complete regression study, with assumptions checked and coefficients "
   "interpreted.",
   ["Produce a model with justified preprocessing and a tuned penalty",
    "Check and report the residuals",
    "Compare your recovered coefficients against the known truth"],
   ["The one dataset where you can check whether the model found reality"],
   dataset="house_prices", prereqs=("ml-polynomial",), kind=PROJECT)

# ---------------------------------------------------------------------------
# Week 5 — Classification
# ---------------------------------------------------------------------------
_t("ml-logistic", "Logistic Regression", SUP, 5, 29, I, 170,
   "Predicting a probability, and why a straight line will not do it.",
   ["Explain the sigmoid and what it is for",
    "Interpret a coefficient as a change in log-odds",
    "Say why squared error is the wrong loss here"],
   ["The default classifier in medicine, credit and marketing",
    "The last layer of most neural classifiers is exactly this"],
   dataset="customer_churn", prereqs=("proj-w4",),
   concepts=("logistic_regression", "sigmoid", "log_odds", "probability_output",
             "cross_entropy"),
   maths=("exponential", "logarithm", "odds", "sigmoid_formula",
          "log_loss_formula"))

_t("ml-decision-boundary", "Decision Boundaries", SUP, 5, 30, I, 160,
   "What a classifier is actually drawing in feature space.",
   ["Plot a decision boundary in two dimensions",
    "Say which models can draw curved boundaries and which cannot",
    "Show what a boundary looks like when classes overlap"],
   ["The clearest way to see what a model has and has not learned",
    "Explains at a glance why one model beats another here"],
   dataset="iris", prereqs=("ml-logistic",),
   concepts=("decision_boundary", "linear_separability",
                                     "feature_space"),
   maths=("hyperplane", "dot_product"))

_t("ml-knn", "K Nearest Neighbours", SUP, 5, 31, I, 150,
   "The model that does no learning at all, and what that costs.",
   ["Implement k-NN from scratch",
    "Show what k controls, at both extremes",
    "Explain why it needs scaling and why it is slow to predict"],
   ["A strong baseline that is often forgotten",
    "Recommendation systems start here"],
   dataset="iris", prereqs=("ml-decision-boundary",),
   concepts=("knn", "lazy_learning", "distance_weighting", "curse_of_dimensionality"),
   maths=("euclidean_distance", "manhattan_distance"))

_t("ml-naive-bayes", "Naive Bayes", SUP, 5, 32, I, 160,
   "A classifier built directly from probability, and the assumption in its "
   "name.",
   ["State Bayes' theorem and apply it to a two-class problem",
    "Explain what 'naive' refers to and when it is harmless",
    "Show why it works well on text despite the assumption being false"],
   ["Spam filtering was built on this",
    "Still competitive on text with very little data"],
   dataset="breast_cancer", prereqs=("ml-knn",),
   concepts=("naive_bayes", "conditional_independence", "prior", "posterior",
             "likelihood"),
   maths=("bayes_theorem", "conditional_probability", "product_rule"))

_t("ml-svm", "Support Vector Machines", SUP, 5, 33, A, 180,
   "Maximising the margin, and the kernel trick that makes it non-linear.",
   ["Explain what the margin is and why maximising it helps",
    "Say what a support vector is",
    "Show a linear and an RBF kernel on the same data"],
   ["The strongest classical classifier on small, clean, wide data",
    "Dominant in bioinformatics before deep learning"],
   dataset="breast_cancer", prereqs=("ml-naive-bayes",),
   concepts=("svm", "margin", "support_vector", "kernel", "rbf", "c_parameter"),
   maths=("dot_product", "lagrange_multiplier", "kernel_function"))

_t("ml-imbalance", "When One Class Is Rare", SUP, 5, 34, A, 160,
   "Resampling, class weights, and why the fix is usually not resampling.",
   ["Compare class weighting against over- and under-sampling",
    "Explain why SMOTE must happen inside the fold",
    "Show that the threshold often matters more than the resampling"],
   ["Fraud is one in a thousand and every naive model predicts 'no fraud'",
    "Rare disease screening has the same shape"],
   dataset="customer_churn", prereqs=("ml-svm",),
   concepts=("class_weight", "oversampling", "undersampling", "smote",
             "resampling_leak"),
   maths=("class_prior", "weighted_loss"))

_t("proj-w5", "Project: A Churn Classifier That Ships", SUP, 5, 35, A, 200,
   "Compare four classifiers honestly and pick one with a defended threshold.",
   ["Cross-validate four model families on the same pipeline",
    "Choose a metric and threshold from a stated cost",
    "Show the leaking column being correctly excluded"],
   ["This is the comparison a reviewer will ask you to justify"],
   dataset="customer_churn", prereqs=("ml-imbalance",), kind=PROJECT)

# ---------------------------------------------------------------------------
# Week 6 — Trees and ensembles
# ---------------------------------------------------------------------------
_t("ml-decision-tree", "Decision Trees", SUP, 6, 36, I, 170,
   "A model you can read, and the greedy rule that builds it.",
   ["Explain how a split is chosen, with the impurity measure",
    "Read a fitted tree and convert it to rules",
    "Show a tree overfitting and prune it"],
   ["The only model a non-technical stakeholder can audit directly",
    "Medical triage protocols are decision trees"],
   dataset="customer_churn", prereqs=("proj-w5",),
   concepts=("decision_tree", "split", "gini", "entropy", "pruning", "max_depth"),
   maths=("entropy_formula", "gini_formula", "information_gain", "logarithm"))

_t("ml-random-forest", "Random Forests", SUP, 6, 37, I, 170,
   "Many weak trees, disagreeing usefully.",
   ["Explain bagging and why randomness helps",
    "Say why a forest overfits far less than one deep tree",
    "Read feature importances and state their limits"],
   ["The default first serious model on tabular data",
    "Wins more Kaggle tabular baselines than anything except boosting"],
   dataset="customer_churn", prereqs=("ml-decision-tree",),
   concepts=("bagging", "random_forest", "bootstrap", "feature_subsampling",
             "oob_score", "feature_importance"),
   maths=("bootstrap_sampling", "variance_reduction", "averaging"))

_t("ml-boosting", "Gradient Boosting", SUP, 6, 38, A, 180,
   "Building a model out of its own mistakes.",
   ["Explain how boosting differs from bagging, mechanically",
    "Say what each successive tree is fitted to",
    "Tune learning rate against number of trees"],
   ["The strongest tabular model family there is",
    "XGBoost and LightGBM are this idea, optimised"],
   dataset="customer_churn", prereqs=("ml-random-forest",),
   concepts=("boosting", "gradient_boosting",
                                 "residual_fitting", "n_estimators",
                                 "boosting_shrinkage", "early_stopping"),
   maths=("gradient", "additive_model", "weighted_sum"))

_t("ml-model-comparison", "Comparing Models Honestly", SUP, 6, 39, A, 160,
   "Is that difference real, or is it noise?",
   ["Report a score with its spread across folds",
    "Say when a 1% difference is meaningless",
    "Compare models on the same folds and explain why that matters"],
   ["Most published model comparisons do not survive this",
    "The difference between a result and a fluctuation"],
   dataset="wine", prereqs=("ml-boosting",),
   concepts=("paired_comparison", "score_variance", "statistical_significance",
             "same_folds"),
   maths=("standard_error", "confidence_interval", "paired_test"))

_t("ml-hyperparameters", "Tuning Without Fooling Yourself", SUP, 6, 40, A, 170,
   "Grid search, random search, and the nested split that keeps it honest.",
   ["Run a grid search inside cross-validation",
    "Explain why random search often beats grid search",
    "Say what a nested split protects against"],
   ["Tuning on the test set is the most common way to publish a lie",
    "Compute budget is why random search exists"],
   dataset="breast_cancer", prereqs=("ml-model-comparison",),
   concepts=("grid_search", "random_search", "nested_cv", "hyperparameter",
             "validation_set"),
   maths=("search_space", "combinatorics"))

_t("ml-interpretability", "Explaining a Model's Decision", SUP, 6, 41, A, 170,
   "Permutation importance and partial dependence — what the model is really "
   "using.",
   ["Compute permutation importance and say why it beats tree importance",
    "Read a partial dependence plot",
    "Show a model relying on a feature it should not"],
   ["Regulated industries require an explanation, not just a score",
    "This is how leakage is caught after the fact"],
   dataset="customer_churn", prereqs=("ml-hyperparameters",),
   concepts=("permutation_importance", "partial_dependence", "shap",
             "global_vs_local"),
   maths=("expectation", "marginal_effect"))

_t("proj-w6", "Milestone: The Model Report", SUP, 6, 42, A, 220,
   "A full written comparison with tuning, interpretation and a recommendation.",
   ["Tune three families under nested cross-validation",
    "Interpret the winner and state what drives it",
    "Recommend one, with the cost assumption written down"],
   ["The document that gets a model approved or rejected"],
   dataset="customer_churn", prereqs=("ml-interpretability",), kind=MILESTONE)

# ---------------------------------------------------------------------------
# Week 7 — Unsupervised learning
# ---------------------------------------------------------------------------
_t("ml-clustering-intro", "Finding Groups Without Labels", UNSUP, 7, 43, I, 160,
   "What clustering is for, and why it has no right answer.",
   ["Say what makes a clustering good in the absence of labels",
    "Explain why the same data supports several valid clusterings",
    "Name three real uses that are not customer segmentation"],
   ["Customer segments, document topics, gene groups",
    "The results are a hypothesis, never a finding"],
   dataset="digits", prereqs=("proj-w6",),
   concepts=("clustering", "similarity", "cluster_validity"),
   maths=("distance", "centroid"))

_t("ml-kmeans", "K-Means", UNSUP, 7, 44, I, 170,
   "The algorithm in four lines, and everything it assumes.",
   ["Implement k-means from scratch and watch it converge",
    "Choose k by elbow and by silhouette, and distrust both",
    "Show a shape k-means cannot find"],
   ["Image compression, segmentation, vector quantisation",
    "Assumes round, equal-sized clusters, which real data rarely has"],
   dataset="digits", prereqs=("ml-clustering-intro",),
   concepts=("kmeans", "centroid", "inertia", "elbow_method", "silhouette",
             "initialisation"),
   maths=("euclidean_distance", "mean_vector", "within_cluster_variance"))

_t("ml-hierarchical-dbscan", "Hierarchical Clustering and DBSCAN", UNSUP, 7, 45, A, 170,
   "Clustering without choosing k, and clustering by density.",
   ["Read a dendrogram and cut it sensibly",
    "Explain how DBSCAN finds shapes k-means cannot",
    "Say what DBSCAN does with points that belong nowhere"],
   ["Anomaly detection falls out of density clustering for free",
    "Geographic and sensor data are rarely spherical"],
   dataset="digits", prereqs=("ml-kmeans",),
   concepts=("hierarchical", "dendrogram", "linkage", "dbscan", "epsilon",
             "min_samples", "noise_points"),
   maths=("distance_matrix", "density"))

_t("ml-pca", "Principal Component Analysis", UNSUP, 7, 46, A, 190,
   "Rotating the data to find where the variance actually lives.",
   ["Explain what a principal component is, geometrically",
    "Read an explained-variance plot and choose a cut",
    "Say exactly what is lost when dimensions are dropped"],
   ["Compressing 64 pixel columns to 10 without losing the digits",
    "The standard first move on wide, correlated data"],
   dataset="digits", prereqs=("ml-hierarchical-dbscan",),
   concepts=("pca", "principal_component", "explained_variance", "projection",
             "dimensionality_reduction"),
   maths=("eigenvector", "eigenvalue", "covariance_matrix", "orthogonality",
          "matrix_multiplication"))

_t("ml-tsne-umap", "Seeing High-Dimensional Data", UNSUP, 7, 47, A, 160,
   "t-SNE and UMAP, and how to read them without being misled.",
   ["Say what these preserve and what they distort",
    "Explain why cluster sizes and distances in a t-SNE plot mean nothing",
    "Show the same data under two settings giving two stories"],
   ["Every single-cell biology paper has one of these plots",
    "Also the most over-interpreted plot in the field"],
   dataset="digits", prereqs=("ml-pca",),
   concepts=("tsne", "umap", "perplexity", "neighbourhood_preservation",
             "manifold"),
   maths=("kl_divergence", "probability_distribution"))

_t("ml-anomaly", "Finding What Does Not Belong", UNSUP, 7, 48, A, 160,
   "Anomaly detection when you have almost no examples of the thing you want.",
   ["Explain isolation forest in one paragraph",
    "Say why anomaly detection is not just classification with few positives",
    "Set a contamination rate from a stated cost"],
   ["Fraud, intrusion, machine failure",
    "The positives are too rare and too varied to learn from directly"],
   dataset="customer_churn", prereqs=("ml-tsne-umap",),
   concepts=("anomaly_detection", "isolation_forest", "novelty", "contamination"),
   maths=("path_length", "probability_tail"))

_t("proj-w7", "Project: Segmenting Without Labels", UNSUP, 7, 49, A, 200,
   "Cluster the customers, then argue the segments are real.",
   ["Produce a clustering with a defended k",
    "Reduce dimensions first and justify how many were kept",
    "Describe each segment in words a manager could act on"],
   ["The deliverable is the interpretation, not the algorithm"],
   dataset="customer_churn", prereqs=("ml-anomaly",), kind=PROJECT)

# ---------------------------------------------------------------------------
# Week 8 — Text, time and other awkward data
# ---------------------------------------------------------------------------
_t("ml-text-features", "Turning Text into Numbers", ADV, 8, 50, I, 170,
   "Bag of words and TF-IDF, and what both throw away.",
   ["Build a TF-IDF matrix and explain each part of the name",
    "Say what is lost when word order is discarded",
    "Show why rare words carry more signal"],
   ["Search, spam filtering and document classification all start here",
    "Still the right tool when you have 500 documents, not 500,000"],
   dataset="customer_churn", prereqs=("proj-w7",),
   concepts=("bag_of_words", "tfidf", "tokenisation", "stopwords", "ngram",
             "sparse_matrix"),
   maths=("term_frequency", "inverse_document_frequency", "logarithm"))

_t("ml-text-classification", "Classifying Text", ADV, 8, 51, I, 160,
   "A full text pipeline, and why naive Bayes refuses to die.",
   ["Build a text classification pipeline end to end",
    "Compare naive Bayes and logistic regression on the same features",
    "Read the most informative features and sanity-check them"],
   ["Support ticket routing, sentiment, moderation",
    "Inspecting the top features catches label errors fast"],
   dataset="customer_churn", prereqs=("ml-text-features",),
   concepts=("text_pipeline", "vectoriser", "informative_features"),
   maths=("log_odds",))

_t("ml-time-series", "Data That Has an Order", ADV, 8, 52, A, 180,
   "Why everything you have learned about splitting is wrong here.",
   ["Explain why a random split leaks in time series",
    "Build lag features without leaking the future",
    "Validate with a rolling origin"],
   ["Demand, prices and traffic are all ordered",
    "The most common leak in industry forecasting"],
   dataset="house_prices", prereqs=("ml-text-classification",),
   concepts=("time_series", "lag_features", "rolling_window", "temporal_split",
             "lookahead_bias", "stationarity"),
   maths=("autocorrelation", "moving_average", "differencing"))

_t("ml-leakage-deep", "Leakage, Hunted Properly", ADV, 8, 53, A, 170,
   "Every way the answer sneaks into the features, and how to find them.",
   ["Name five distinct kinds of leakage",
    "Find the leaking column in the churn data from its behaviour alone",
    "Explain why a suspiciously good score is bad news"],
   ["The single most expensive mistake in applied ML",
    "A model that scores 0.99 is usually broken, not brilliant"],
   dataset="customer_churn", prereqs=("ml-time-series",),
   concepts=("target_leakage", "train_test_contamination", "temporal_leakage",
             "group_leakage", "too_good_to_be_true"),
   maths=())

_t("ml-groups", "When Rows Are Not Independent", ADV, 8, 54, A, 160,
   "Grouped data, repeated measures, and the split that respects them.",
   ["Explain why rows from one patient must not straddle the split",
    "Use a grouped split correctly",
    "Show the inflated score a naive split produces"],
   ["Medical, sensor and user-level data are all grouped",
    "Ignoring it inflates every reported score"],
   dataset="customer_churn", prereqs=("ml-leakage-deep",),
   concepts=("group_kfold", "clustered_data", "repeated_measures"),
   maths=("independence_assumption",))

_t("ml-calibration", "Are the Probabilities Real?", ADV, 8, 55, A, 160,
   "A model that says 0.8 should be right 80% of the time. Most are not.",
   ["Plot a calibration curve and read it",
    "Explain why tree ensembles are usually badly calibrated",
    "Calibrate a model and show the improvement"],
   ["Any decision made on a probability threshold needs this",
    "Risk scores that are not calibrated are not risk scores"],
   dataset="breast_cancer", prereqs=("ml-groups",),
   concepts=("calibration", "reliability_curve", "platt_scaling",
             "isotonic_regression", "brier_score"),
   maths=("expected_probability", "brier_formula"))

_t("proj-w8", "Capstone: An End-to-End Study", ADV, 8, 56, A, 260,
   "A complete, honest, reproducible piece of applied ML.",
   ["Take raw data to a calibrated, interpreted, validated model",
    "Document every decision and its alternative",
    "State plainly what the model must not be used for"],
   ["This is a portfolio piece, and the shape of a real deliverable"],
   dataset="customer_churn", prereqs=("ml-calibration",), kind=CAPSTONE)

# ---------------------------------------------------------------------------
# Week 9 — Neural networks, from the maths up
# ---------------------------------------------------------------------------
_t("ml-perceptron", "The Perceptron", ADV, 9, 57, A, 170,
   "One neuron, built by hand, and exactly what it can and cannot do.",
   ["Implement a perceptron from scratch",
    "Show it solving AND and failing on XOR",
    "Explain why that failure stopped the field for a decade"],
   ["Every neural network is this unit, repeated",
    "The XOR failure is the reason hidden layers exist"],
   dataset="iris", prereqs=("proj-w8",),
   concepts=("perceptron", "weights", "bias_term",
                                 "activation", "step_function"),
   maths=("dot_product", "weighted_sum", "inequality"))

_t("ml-mlp-forward", "Layers, and the Forward Pass", ADV, 9, 58, A, 180,
   "Stacking neurons, and computing a prediction by hand.",
   ["Compute a forward pass through a 2-3-1 network with a pen",
    "Explain why a network without non-linearity is just a linear model",
    "Compare three activation functions and their failure modes"],
   ["This computation, repeated billions of times, is a language model",
    "Why ReLU replaced sigmoid nearly everywhere"],
   dataset="digits", prereqs=("ml-perceptron",),
   concepts=("hidden_layer", "forward_pass", "relu", "sigmoid_activation",
             "tanh", "universal_approximation"),
   maths=("matrix_multiplication", "composition_of_functions", "non_linearity"))

_t("ml-backprop", "Backpropagation", ADV, 9, 59, A, 200,
   "The chain rule, applied carefully, and nothing more mysterious than that.",
   ["Derive the gradient for one weight by hand",
    "Explain backpropagation as the chain rule applied backwards",
    "Show a vanishing gradient and say what causes it"],
   ["Every deep learning framework is an efficient implementation of this",
    "Understanding it is what separates using a library from knowing why it "
    "fails"],
   dataset="digits", prereqs=("ml-mlp-forward",),
   concepts=("backpropagation", "chain_rule_application", "vanishing_gradient",
             "weight_update", "computational_graph"),
   maths=("chain_rule", "partial_derivative", "gradient", "derivative_of_sigmoid"))

_t("ml-training-nn", "Training, and Why It Goes Wrong", ADV, 9, 60, A, 180,
   "Initialisation, batch size, learning rate schedules and the failures each "
   "causes.",
   ["Explain why initialising all weights to zero fails",
    "Say what batch size trades against what",
    "Diagnose three training curves from their shape alone"],
   ["Most time spent on neural networks is spent on this",
    "A loss curve is the first thing anyone looks at"],
   dataset="digits", prereqs=("ml-backprop",),
   concepts=("weight_initialisation", "batch_size",
                                 "mini_batch", "sgd", "momentum", "adam",
                                 "learning_rate_schedule", "loss_curve"),
   maths=("stochastic_approximation", "moving_average", "variance_of_gradient"))

_t("ml-regularising-nn", "Keeping a Network Honest", ADV, 9, 61, A, 170,
   "Dropout, early stopping and weight decay — overfitting, revisited with "
   "far more capacity.",
   ["Explain dropout and why it works",
    "Set up early stopping correctly",
    "Show a network memorising the training set"],
   ["A large network will memorise anything you give it",
    "These three are in essentially every trained model"],
   dataset="digits", prereqs=("ml-training-nn",),
   concepts=("dropout", "weight_decay", "data_augmentation",
                                 "capacity_control"),
   maths=("expectation_under_dropout", "l2_penalty"))

_t("ml-nn-vs-classical", "When a Network Is the Wrong Answer", ADV, 9, 62, A, 160,
   "On tabular data, gradient boosting usually wins. Knowing why matters.",
   ["Compare a network and a boosted forest on the same tabular problem",
    "Say what data shapes favour networks",
    "State the cost of a network beyond its score"],
   ["Most business problems are tabular, and most should not use deep learning",
    "Knowing this saves months"],
   dataset="customer_churn", prereqs=("ml-regularising-nn",),
   concepts=("inductive_bias", "tabular_vs_perceptual", "sample_efficiency",
             "compute_cost"),
   maths=())

_t("proj-w9", "Milestone: A Network From Scratch", ADV, 9, 63, A, 240,
   "Build and train a small network using only numpy, then check it against a "
   "library.",
   ["Implement forward and backward passes yourself",
    "Train it to a sensible accuracy on digits",
    "Match your gradients against a numerical check"],
   ["After this, no framework is a black box"],
   dataset="digits", prereqs=("ml-nn-vs-classical",), kind=MILESTONE)

# ---------------------------------------------------------------------------
# Week 10 — Getting models into the world
# ---------------------------------------------------------------------------
_t("ml-reproducibility", "Making a Result Reproducible", ADV, 10, 64, I, 160,
   "Seeds, versions and the discipline that lets someone else get your number.",
   ["Make a pipeline produce identical results twice",
    "Name three sources of non-determinism",
    "Record everything needed to rerun a result in a year"],
   ["An unreproducible result is not a result",
    "The replication crisis reached ML some time ago"],
   dataset="customer_churn", prereqs=("proj-w9",),
   concepts=("random_state", "determinism", "environment_pinning", "experiment_log"),
   maths=("pseudorandomness",))

_t("ml-saving-models", "Saving and Loading a Model", ADV, 10, 65, I, 150,
   "Persisting a fitted pipeline, and what breaks when you do it badly.",
   ["Save and reload a full pipeline and get identical predictions",
    "Explain why saving the model without the preprocessing is useless",
    "Say what a version mismatch does on load"],
   ["A model that only exists in a notebook does not exist",
    "The preprocessing is part of the model"],
   dataset="customer_churn", prereqs=("ml-reproducibility",),
   concepts=("serialisation", "joblib", "pickle_risk", "version_mismatch",
             "model_artifact"),
   maths=())

_t("ml-drift", "Models Decay", ADV, 10, 66, A, 170,
   "The world moves and the model does not. Detecting it before it costs "
   "something.",
   ["Distinguish data drift from concept drift",
    "Detect a distribution shift between two samples",
    "Design a monitoring plan for a deployed model"],
   ["Every deployed model degrades; the only question is whether you notice",
    "COVID broke essentially every demand forecast overnight"],
   dataset="customer_churn", prereqs=("ml-saving-models",),
   concepts=("data_drift", "concept_drift", "population_stability",
             "monitoring", "retraining_trigger"),
   maths=("distribution_distance", "psi", "ks_statistic"))

_t("ml-fairness", "Who Does the Model Fail?", ADV, 10, 67, A, 180,
   "Measuring whether errors fall evenly, and the impossibility result "
   "underneath.",
   ["Compute error rates per group and compare",
    "State two fairness definitions that cannot both hold",
    "Show a model that is accurate overall and unfair in a subgroup"],
   ["Hiring, lending and sentencing models have all failed this way",
    "An aggregate score hides who is being harmed"],
   dataset="customer_churn", prereqs=("ml-drift",),
   concepts=("group_fairness", "demographic_parity", "equalised_odds",
             "subgroup_error", "proxy_variable"),
   maths=("conditional_probability", "rate_comparison"))

_t("ml-serving", "Serving a Prediction", ADV, 10, 68, A, 170,
   "Wrapping a model in something that answers a request.",
   ["Turn a saved pipeline into a callable prediction function",
    "Validate an incoming row before predicting on it",
    "Explain what to do when a feature arrives missing"],
   ["This is where the model meets reality",
    "Input validation prevents the majority of production incidents"],
   dataset="customer_churn", prereqs=("ml-fairness",),
   concepts=("inference", "input_validation", "schema", "batch_vs_online",
             "latency"),
   maths=())

_t("ml-cost-of-errors", "Turning a Model into a Decision", ADV, 10, 69, A, 160,
   "Expected value, and why the best model is not always the best decision.",
   ["Build a cost matrix and find the threshold that minimises expected cost",
    "Show a lower-AUC model making better decisions",
    "Explain why the threshold belongs to the business, not the model"],
   ["This is the conversation that gets a model deployed",
    "Retention offers cost money; sending them to everyone loses money"],
   dataset="customer_churn", prereqs=("ml-serving",),
   concepts=("cost_matrix", "expected_value", "decision_threshold", "utility"),
   maths=("expectation", "weighted_sum", "optimisation"))

_t("proj-w10", "Project: A Model You Could Hand Over", ADV, 10, 70, A, 220,
   "Package the churn model with everything another person would need.",
   ["Save the pipeline and prove it reloads identically",
    "Write the input schema and validation",
    "Produce the cost-based threshold recommendation"],
   ["The difference between a notebook and a deliverable"],
   dataset="customer_churn", prereqs=("ml-cost-of-errors",), kind=PROJECT)

# ---------------------------------------------------------------------------
# Week 11 — The statistics underneath
# ---------------------------------------------------------------------------
_t("ml-probability-refresher", "The Probability You Actually Need", ADV, 11, 71, I, 170,
   "Conditional probability, independence and Bayes, rebuilt carefully.",
   ["Compute a conditional probability from a contingency table",
    "Explain independence and show two variables that are not",
    "Apply Bayes to a screening-test problem and get the surprising answer"],
   ["The base-rate answer surprises doctors, so it will surprise you",
    "Every probabilistic model rests on this"],
   dataset="breast_cancer", prereqs=("proj-w10",),
   concepts=("probability", "conditional", "independence", "base_rate",
             "bayes_rule"),
   maths=("conditional_probability", "bayes_theorem", "joint_probability",
          "marginalisation"))

_t("ml-distributions-formal", "The Distributions That Keep Appearing", ADV, 11, 72, I, 170,
   "Normal, binomial, Poisson and the long-tailed ones, and where each shows "
   "up in ML.",
   ["Match four real phenomena to their distributions",
    "Explain the central limit theorem and what it does not say",
    "Show why assuming normality on a long tail fails"],
   ["Why errors are modelled as normal, and when that is wrong",
    "Counts are Poisson, waiting times are exponential"],
   dataset="house_prices", prereqs=("ml-probability-refresher",),
   concepts=("normal_distribution", "binomial", "poisson", "heavy_tail",
             "central_limit_theorem"),
   maths=("probability_density", "expectation", "variance", "clt"))

_t("ml-estimation", "Where Loss Functions Come From", ADV, 11, 73, A, 180,
   "Maximum likelihood, and why squared error and log loss are not arbitrary "
   "choices.",
   ["Derive squared error from a normal likelihood",
    "Derive log loss from a Bernoulli likelihood",
    "Explain what assumption each loss encodes"],
   ["Suddenly every loss function has a reason",
    "Choosing a loss is choosing a noise model"],
   dataset="house_prices", prereqs=("ml-distributions-formal",),
   concepts=("maximum_likelihood", "likelihood_function", "log_likelihood",
             "loss_derivation"),
   maths=("likelihood", "logarithm", "derivative", "argmax", "product_rule"))

_t("ml-uncertainty", "Saying How Sure You Are", ADV, 11, 74, A, 170,
   "Confidence intervals and the bootstrap, applied to model scores.",
   ["Bootstrap a confidence interval around an accuracy",
    "Explain what a 95% interval does and does not mean",
    "Show two models whose intervals overlap and draw the right conclusion"],
   ["A score without an interval is half a result",
    "The bootstrap works when the formula does not exist"],
   dataset="wine", prereqs=("ml-estimation",),
   concepts=("confidence_interval", "sampling_distribution",
                                 "uncertainty_quantification"),
   maths=("resampling", "percentile", "standard_error"))

_t("ml-hypothesis-testing", "Is This Difference Real?", ADV, 11, 75, A, 170,
   "P-values, what they mean, and the many ways they are misused.",
   ["State precisely what a p-value is",
    "Run a paired test on two models' fold scores",
    "Explain p-hacking and multiple comparisons"],
   ["A/B tests are decided on this and read wrongly constantly",
    "Testing twenty features finds one 'significant' by chance"],
   dataset="wine", prereqs=("ml-uncertainty",),
   concepts=("null_hypothesis", "p_value", "significance", "multiple_testing",
             "effect_size"),
   maths=("test_statistic", "null_distribution", "bonferroni"))

_t("ml-causal-intro", "Prediction Is Not Causation", ADV, 11, 76, A, 180,
   "Why a good predictor tells you nothing about what to do.",
   ["Explain confounding with a concrete example",
    "Show a feature that predicts well and would be useless to intervene on",
    "Say what a randomised experiment buys that observation cannot"],
   ["Every 'our model shows X causes Y' claim from a predictive model is wrong",
    "The gap between a dashboard and a decision"],
   dataset="customer_churn", prereqs=("ml-hypothesis-testing",),
   concepts=("causal_effect", "confounding", "intervention", "randomisation",
             "selection_bias"),
   maths=("conditional_vs_interventional",))

_t("proj-w11", "Project: A Result With Error Bars", ADV, 11, 77, A, 200,
   "Re-report an earlier comparison with intervals and an honest verdict.",
   ["Bootstrap intervals for every reported score",
    "Test whether the best model is really better",
    "Rewrite one earlier claim that the statistics do not support"],
   ["Most model comparisons do not survive this treatment"],
   dataset="wine", prereqs=("ml-causal-intro",), kind=PROJECT)

# ---------------------------------------------------------------------------
# Week 12 — Interview and consolidation
# ---------------------------------------------------------------------------
_t("ml-interview-theory", "The Theory Questions", ADV, 12, 78, A, 170,
   "Bias-variance, regularisation, the assumptions — asked the way interviews "
   "ask them.",
   ["Answer ten standard theory questions out loud, unaided",
    "Explain regularisation to a non-technical listener",
    "Catch the false premise in a leading question"],
   ["The first screen at almost every company",
    "Explaining simply is the real test"],
   dataset="", prereqs=("proj-w11",),
   concepts=("interview_theory",), maths=())

_t("ml-interview-coding", "The Coding Round", ADV, 12, 79, A, 180,
   "Implementing metrics, k-means and gradient descent from scratch, on the "
   "clock.",
   ["Implement four metrics from their definitions",
    "Write k-means and logistic regression without a library",
    "Talk through complexity while writing"],
   ["Whiteboard ML rounds test exactly these",
    "Knowing the formula and coding it are different skills"],
   dataset="iris", prereqs=("ml-interview-theory",),
   concepts=("from_scratch_implementation",),
   maths=("metric_formulas", "gradient"))

_t("ml-case-study", "The Case Study Round", ADV, 12, 80, A, 180,
   "'How would you build X?' — structured, with the assumptions stated.",
   ["Structure an open problem in five minutes",
    "State assumptions and the metric before proposing a model",
    "Name the failure modes before being asked"],
   ["Senior interviews are almost entirely this",
    "Framing beats algorithm knowledge here"],
   dataset="", prereqs=("ml-interview-coding",),
   concepts=("problem_framing", "metric_selection", "failure_modes"), maths=())

_t("ml-debugging", "When the Model Is Bad", ADV, 12, 81, A, 170,
   "A systematic route from 'it does not work' to the cause.",
   ["Follow a fixed checklist from symptom to cause",
    "Diagnose four broken pipelines from their symptoms",
    "Say what to check first and why"],
   ["The most useful skill and the least taught",
    "Usually the data, rarely the algorithm"],
   dataset="customer_churn", prereqs=("ml-case-study",),
   concepts=("debugging_checklist", "sanity_check", "ablation"), maths=())

_t("ml-reading-papers", "Reading a Paper Critically", ADV, 12, 82, A, 160,
   "Getting the idea out of a paper, and finding the weakness in its "
   "evaluation.",
   ["Extract the contribution of a paper in three sentences",
    "Find the baseline the authors chose and ask whether it was fair",
    "Name what the ablation does not tell you"],
   ["Essential for the research internships you are applying to",
    "Most reported gains shrink under scrutiny"],
   dataset="", prereqs=("ml-debugging",),
   concepts=("paper_structure", "baseline_fairness", "ablation_reading"),
   maths=())

_t("ml-portfolio", "A Project Someone Will Actually Read", ADV, 12, 83, A, 170,
   "Turning your work into evidence of skill.",
   ["Write a README that states the problem, the result and the limits",
    "Choose what to show and what to cut",
    "Make the repository runnable by a stranger"],
   ["The portfolio is what gets the interview",
    "A clear limitations section reads as competence, not weakness"],
   dataset="", prereqs=("ml-reading-papers",),
   concepts=("project_writeup", "reproducible_repo"), maths=())

_t("proj-w12", "Milestone: A Mock Interview Loop", ADV, 12, 84, A, 220,
   "Theory, coding and case study, timed, then scored against a rubric.",
   ["Complete all three rounds under time",
    "Score yourself against the rubric honestly",
    "Write down your three weakest answers"],
   ["A rehearsal is worth more than another topic"],
   dataset="iris", prereqs=("ml-portfolio",), kind=MILESTONE)

# ---------------------------------------------------------------------------
# Week 13 — The capstone
# ---------------------------------------------------------------------------
_t("ml-problem-framing", "Choosing and Framing the Problem", ADV, 13, 85, A, 170,
   "Turning a vague goal into a specific, measurable prediction task.",
   ["Write a problem statement with a metric and a success bar",
    "State what data you would need and what you actually have",
    "Name the decision the model is meant to support"],
   ["A badly framed problem cannot be rescued by modelling"],
   dataset="customer_churn", prereqs=("proj-w12",),
   concepts=("problem_statement", "success_criteria", "scope"), maths=())

_t("ml-capstone-eda", "Capstone: Understanding the Data", ADV, 13, 86, A, 200,
   "The exploration, done properly and written down.",
   ["Produce an EDA that answers the questions framing raised",
    "Document every quality problem found",
    "Decide the preprocessing before touching a model"],
   ["The half of the work that gets skipped and shows"],
   dataset="customer_churn", prereqs=("ml-problem-framing",), kind=CAPSTONE)

_t("ml-capstone-baseline", "Capstone: Baseline and Iteration", ADV, 13, 87, A, 200,
   "The simplest thing that works, then earned improvements.",
   ["Establish a baseline before any modelling",
    "Improve it three times, each with a measured reason",
    "Keep a log of what failed as well as what worked"],
   ["'We tried and it did not help' is a finding worth recording"],
   dataset="customer_churn", prereqs=("ml-capstone-eda",), kind=CAPSTONE)

_t("ml-capstone-validation", "Capstone: Validation and Interpretation", ADV, 13, 88, A, 200,
   "Proving the result and explaining what drives it.",
   ["Validate with the right scheme for this data",
    "Report intervals, not point estimates",
    "Interpret the model and check it against domain sense"],
   ["Where the work becomes defensible"],
   dataset="customer_churn", prereqs=("ml-capstone-baseline",), kind=CAPSTONE)

_t("ml-capstone-writeup", "Capstone: The Write-Up", ADV, 13, 89, A, 200,
   "The document that makes the work useful to someone else.",
   ["Write the full report: problem, data, method, result, limits",
    "Include the decision recommendation and its cost basis",
    "State what you would do with another month"],
   ["This is the portfolio piece"],
   dataset="customer_churn", prereqs=("ml-capstone-validation",), kind=CAPSTONE)

_t("ml-whats-next", "What You Do Not Know Yet", ADV, 13, 90, A, 150,
   "An honest map of the field beyond this course, and where to go next.",
   ["Name five areas this course deliberately did not cover",
    "Choose a direction and the next three resources for it",
    "Say what you would need to learn for a research role"],
   ["Deep learning, NLP, reinforcement learning, causal inference, MLOps",
    "Knowing the shape of your ignorance is a skill"],
   dataset="", prereqs=("ml-capstone-writeup",),
   concepts=("field_map", "next_steps"), maths=())

_t("ml-final-review", "Final Review", ADV, 13, 91, A, 180,
   "Everything, tested at once, against the record of what you struggled with.",
   ["Pass a comprehensive assessment drawn from your weakest topics",
    "Re-derive three results without notes",
    "Write your own summary of the thirteen weeks"],
   ["The recall test that makes the knowledge stick"],
   dataset="customer_churn", prereqs=("ml-whats-next",), kind=MILESTONE)


DAYS_PER_WEEK = 7


def by_week() -> dict[int, list[dict[str, Any]]]:
    out: dict[int, list[dict[str, Any]]] = {}
    for topic in TOPICS:
        out.setdefault(topic["week"], []).append(topic)
    for group in out.values():
        group.sort(key=lambda t: t["day"])
    return out


def by_slug() -> dict[str, dict[str, Any]]:
    return {t["slug"]: t for t in TOPICS}


def get(slug: str) -> dict[str, Any] | None:
    return by_slug().get(slug)


def on_day(day: int) -> dict[str, Any] | None:
    return next((t for t in TOPICS if t["day"] == day), None)


def total_minutes() -> int:
    return sum(t["minutes"] for t in TOPICS)


def validate() -> list[str]:
    """Structural checks, run at seed time and by the tests.

    A typo in a prerequisite slug should fail loudly here rather than silently
    breaking the gating three weeks later, when the only symptom is that a
    topic never unlocks and nobody can say why.
    """
    problems: list[str] = []
    slugs = [t["slug"] for t in TOPICS]

    seen: set[str] = set()
    for slug in slugs:
        if slug in seen:
            problems.append(f"duplicate slug: {slug}")
        seen.add(slug)

    expected = PLAN_WEEKS * DAYS_PER_WEEK
    if len(TOPICS) != expected:
        problems.append(f"expected {expected} topics, found {len(TOPICS)}")

    days = sorted(t["day"] for t in TOPICS)
    if days != list(range(1, len(TOPICS) + 1)):
        problems.append(f"days are not 1..{len(TOPICS)} without gaps")

    day_of = {t["slug"]: t["day"] for t in TOPICS}
    for topic in TOPICS:
        for prereq in topic["prereqs"]:
            if prereq not in seen:
                problems.append(f"{topic['slug']}: unknown prereq '{prereq}'")
            elif day_of[prereq] >= topic["day"]:
                problems.append(
                    f"{topic['slug']} (day {topic['day']}) requires {prereq} "
                    f"(day {day_of[prereq]}), which comes later")

    for week, group in by_week().items():
        if len(group) != DAYS_PER_WEEK:
            problems.append(f"week {week} has {len(group)} days, expected "
                            f"{DAYS_PER_WEEK}")
        if group[-1]["kind"] == LESSON:
            problems.append(f"week {week} does not end with something built")

    # An idea is introduced once. Two topics both claiming to introduce
    # cross-validation means one of them teaches it a second time as though it
    # were new, and the horizon would then place it on whichever came first.
    introduced: dict[str, int] = {}
    for topic in TOPICS:
        for key in topic["concepts"]:
            if key in introduced:
                problems.append(f"concept '{key}' is introduced twice: days "
                                f"{introduced[key]} and {topic['day']}")
            else:
                introduced[key] = topic["day"]

    # Every day that names a dataset must name one that exists.
    from . import datasets                             # noqa: PLC0415
    for topic in TOPICS:
        name = topic.get("dataset")
        if name and name not in datasets.CATALOGUE:
            problems.append(f"{topic['slug']}: unknown dataset '{name}'")

    if not any(t["kind"] == CAPSTONE for t in TOPICS):
        problems.append("no capstone in the plan")

    # The course's own stated ordering principle: evaluation before models.
    first_model = min((t["day"] for t in TOPICS
                       if "linear_model" in t["concepts"]
                       or "logistic_regression" in t["concepts"]), default=99)
    for idea in ("train_test_split", "overfitting", "cross_validation"):
        arrives = next((t["day"] for t in TOPICS if idea in t["concepts"]), 0)
        if not arrives:
            problems.append(f"'{idea}' is never taught")
        elif arrives >= first_model:
            problems.append(f"'{idea}' (day {arrives}) must come before the "
                            f"first model (day {first_model})")

    return problems
