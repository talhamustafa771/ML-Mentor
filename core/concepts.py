"""What the student is allowed to have met by a given day.

The Python mentor learned this lesson expensively. The curriculum fixed the
order, the generator ignored it, and day 1 opened with a function definition
taught on day 43. Nothing in the prompt had ever said *what the reader already
knows* — only what the topic was.

Machine learning makes the same failure both likelier and harder to see. A
model asked to teach "the train/test split" will reach for a pipeline, scale
the features, cross-validate and print a classification report, because that is
what the code it was trained on looks like. Every one of those is correct
practice and useless on day 5: four ideas the reader has never met, smuggled in
as though they were punctuation.

So this module supplies the missing constraint and makes it checkable.

  introduced_on(concept)  - the day an idea first arrives
  allowed_api(day)        - the library tools available by then
  check_code(code, day)   - parses a cell and names anything from the future
  horizon(topic)          - the prompt fragment that states the boundary
  chain_for(concept)      - what a term rests on, for the Math Helper

Two decisions are worth stating, because they are the opposite of the Python
version's.

The day each idea arrives is *derived from the curriculum*, never written down
twice. Move a topic and the horizon moves with it. The Python mentor kept a
parallel table and needed a test to stop the two drifting; here they cannot.

The check flags only what it recognises. The Python horizon worked from an
allow-list and rejected anything unknown, which is right when the reader knows
nothing. This reader already writes Python, so an unknown symbol is far more
likely to be ordinary code than a leak from week 9. A false alarm sends a
perfectly good lesson back to be rewritten, and that costs more than the rare
miss. The list of recognised symbols is therefore long and deliberately
maintained: it is the whole enforcement surface.
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import Any

from . import curriculum

PLAN_DAYS = 91


# ---------------------------------------------------------------------------
# Naming
# ---------------------------------------------------------------------------
# Most slugs read fine once the underscores go. These are the ones that do not:
# acronyms that must not be lowercased, and terms whose plain expansion would
# be wrong or vague in a prompt.
_LABELS: dict[str, str] = {
    "auc": "AUC", "roc_curve": "the ROC curve", "pca": "PCA", "tsne": "t-SNE",
    "umap": "UMAP", "svm": "support vector machines", "knn": "k-nearest neighbours",
    "mae": "MAE (mean absolute error)", "rmse": "RMSE", "r_squared": "R squared",
    "f1": "the F1 score", "iqr": "the interquartile range",
    "mcar": "missing completely at random", "mnar": "missing not at random",
    "smote": "SMOTE", "oob_score": "the out-of-bag score",
    "tfidf": "TF-IDF", "sgd": "stochastic gradient descent",
    "relu": "ReLU", "l1_penalty": "the L1 penalty", "l2_penalty": "the L2 penalty",
    "k_fold": "k-fold cross-validation", "cv_score": "cross-validated scores",
    "one_hot": "one-hot encoding", "min_max": "min-max scaling",
    "psi": "the population stability index", "ks_statistic": "the KS statistic",
    "shap": "SHAP values", "adam": "the Adam optimiser",
    "kmeans": "k-means", "dbscan": "DBSCAN", "ngram": "n-grams",
    "clt": "the central limit theorem",
    "central_limit_theorem": "the central limit theorem",
    "p_value": "p-values", "mse": "mean squared error",
    "rbf": "the RBF kernel", "c_parameter": "the C parameter in an SVM",
    "gini": "Gini impurity", "entropy": "entropy as a split criterion",
    "kl_divergence": "KL divergence", "r_squared_formula": "the R squared formula",
    "log_odds": "log-odds", "odds": "odds",
    "train_test_split": "the train/test split",
    "data_leakage": "data leakage", "target_leakage": "target leakage",
    "dummy_trap": "the dummy variable trap",
    "curse_of_dimensionality": "the curse of dimensionality",
    "bias": "bias, in the bias-variance sense",
    "bias_term": "the bias term of a neuron",
    "variance": "variance", "split": "a split in a decision tree",
    "threshold": "the classification threshold",
    "alpha": "the regularisation strength alpha",
    "stratify": "stratified sampling", "fit": "calling .fit()",
    "predict": "calling .predict()", "fit_transform": "calling .fit_transform()",
    "estimator": "the estimator interface",
    "dropna": "dropping rows with missing values",
    "pipeline": "sklearn Pipelines", "column_transformer": "ColumnTransformer",
    "grid_search": "GridSearchCV", "random_search": "RandomizedSearchCV",
    "nested_cv": "nested cross-validation",
    "group_kfold": "grouped cross-validation",
    "early_stopping": "early stopping", "n_estimators": "the number of estimators",
    "max_depth": "limiting tree depth",
    "class_weight": "class weighting",
    "predict_proba": "predicted probabilities",
    "mini_batch": "mini-batches", "batch_size": "the batch size",
    "sgd_optimiser": "the SGD optimiser",
}

_ACRONYM = re.compile(r"\b(auc|roc|pca|svm|knn|mae|rmse|iqr|sgd|shap|psi|ks|clt|"
                      r"mcar|mnar|smote|oob|tfidf|l1|l2|relu|kl|cv)\b")


def label(slug: str) -> str:
    """How to name a concept to the model and to the reader."""
    if slug in _LABELS:
        return _LABELS[slug]
    text = slug.replace("_", " ")
    return _ACRONYM.sub(lambda m: m.group(1).upper(), text)


# ---------------------------------------------------------------------------
# The timeline, derived from the curriculum
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Idea:
    key: str
    day: int
    topic_slug: str
    topic_title: str
    week: int
    kind: str            # "concept" or "maths"

    @property
    def label(self) -> str:
        return label(self.key)


def _build() -> tuple[dict[str, Idea], dict[str, Idea]]:
    concepts: dict[str, Idea] = {}
    maths: dict[str, Idea] = {}
    for topic in sorted(curriculum.TOPICS, key=lambda t: int(t["day"])):
        day, week = int(topic["day"]), int(topic["week"])
        for key in topic.get("concepts", ()):
            concepts.setdefault(key, Idea(key, day, topic["slug"],
                                          topic["title"], week, "concept"))
        for key in topic.get("maths", ()):
            maths.setdefault(key, Idea(key, day, topic["slug"],
                                       topic["title"], week, "maths"))
    return concepts, maths


CONCEPTS, MATHS = _build()


def _earliest() -> dict[str, Idea]:
    """One timeline for lookups, taking whichever list mentions a term first.

    A few terms appear as mathematics before they appear as a concept —
    confidence intervals are used to compare two models on day 39, long before
    the day-74 lesson on quantifying uncertainty. The horizon must answer
    "when could the reader first have met this", so the earlier of the two
    wins. Merging the dictionaries naively would have let the later one
    silently overwrite the earlier, which is a wrong answer in the direction
    that matters: it would ban something already taught.
    """
    out: dict[str, Idea] = dict(MATHS)
    for key, idea in CONCEPTS.items():
        existing = out.get(key)
        if existing is None or idea.day < existing.day:
            out[key] = idea
    return out


EVERYTHING: dict[str, Idea] = _earliest()


def introduced_on(key: str) -> int:
    """The day an idea arrives, or 0 if the plan never teaches it."""
    idea = EVERYTHING.get(key)
    return idea.day if idea else 0


def known_by(day: int, kind: str = "") -> list[Idea]:
    pool = {"concept": CONCEPTS, "maths": MATHS}.get(kind, EVERYTHING)
    return sorted((i for i in pool.values() if i.day <= day),
                  key=lambda i: (i.day, i.key))


def introduced_after(day: int, kind: str = "") -> list[Idea]:
    pool = {"concept": CONCEPTS, "maths": MATHS}.get(kind, EVERYTHING)
    return sorted((i for i in pool.values() if i.day > day),
                  key=lambda i: (i.day, i.key))


def introduced_by_topic(slug: str) -> list[Idea]:
    return [i for i in EVERYTHING.values() if i.topic_slug == slug]


def chain_for(key: str, depth: int = 3) -> list[Idea]:
    """What an idea rests on: the ideas taught by its topic's prerequisites.

    The Math Helper uses this. A student who asks what AUC means and does not
    know what a false positive rate is needs the chain, not the definition —
    and the curriculum's own prerequisite links already describe it, so nothing
    needs to be asserted a second time here.
    """
    idea = EVERYTHING.get(key)
    if idea is None:
        return []
    by_slug = {t["slug"]: t for t in curriculum.TOPICS}
    out: list[Idea] = []
    seen: set[str] = {key}
    frontier = [idea.topic_slug]
    for _ in range(max(0, depth)):
        nxt: list[str] = []
        for slug in frontier:
            topic = by_slug.get(slug)
            if not topic:
                continue
            for prereq in topic.get("prereqs", ()):
                nxt.append(prereq)
                parent = by_slug.get(prereq)
                if not parent:
                    continue
                for other in parent.get("concepts", ()):
                    if other not in seen and other in EVERYTHING:
                        seen.add(other)
                        out.append(EVERYTHING[other])
        frontier = nxt
        if not frontier:
            break
    return sorted(out, key=lambda i: -i.day)


# ---------------------------------------------------------------------------
# Library symbols
# ---------------------------------------------------------------------------
# The actual enforcement surface. Concept names are fuzzy in prose; an import
# is not. `StandardScaler` in a day-12 cell is unambiguous evidence that the
# lesson reached six days forward, and no amount of rephrasing hides it.
#
# Each symbol points at the concept that makes it legitimate, so the day comes
# from the curriculum rather than from a number typed here.

_API: dict[str, str] = {
    # week 1 - looking at data
    "describe": "describe", "info": "dtypes", "dtypes": "dtypes",
    "isna": "missing_values", "isnull": "missing_values",
    "hist": "histogram", "histplot": "histogram", "distplot": "histogram",
    "boxplot": "outlier", "skew": "skew", "log1p": "log_transform",
    "scatter": "scatter", "scatterplot": "scatter", "regplot": "scatter",
    "corr": "correlation", "heatmap": "correlation", "pairplot": "correlation",
    "cov": "covariance",
    # week 1-2 - the split, first fit, first score
    "train_test_split": "train_test_split",
    "fit": "fit", "predict": "predict",
    "accuracy_score": "accuracy",
    "DummyClassifier": "baseline", "DummyRegressor": "baseline",
    "learning_curve": "learning_curve", "LearningCurveDisplay": "learning_curve",
    "validation_curve": "model_complexity",
    "cross_val_score": "cross_validation", "cross_validate": "cross_validation",
    "cross_val_predict": "cross_validation",
    "KFold": "k_fold", "StratifiedKFold": "stratified_kfold",
    "ShuffleSplit": "k_fold",
    "confusion_matrix": "confusion_matrix",
    "ConfusionMatrixDisplay": "confusion_matrix",
    "precision_score": "precision", "recall_score": "recall",
    "f1_score": "f1", "classification_report": "f1",
    "balanced_accuracy_score": "class_imbalance",
    "predict_proba": "threshold", "decision_function": "threshold",
    "roc_curve": "roc_curve", "RocCurveDisplay": "roc_curve",
    "roc_auc_score": "auc",
    "precision_recall_curve": "precision_recall_curve",
    "average_precision_score": "precision_recall_curve",
    "mean_absolute_error": "mae", "mean_squared_error": "rmse",
    "root_mean_squared_error": "rmse", "r2_score": "r_squared",
    # week 3 - cleaning and pipelines
    "SimpleImputer": "imputation", "KNNImputer": "imputation",
    "IterativeImputer": "imputation", "fillna": "imputation",
    "MissingIndicator": "missing_indicator",
    "dropna": "dropna",
    "quantile": "iqr", "zscore": "z_score", "winsorize": "winsorise",
    "OneHotEncoder": "one_hot", "get_dummies": "one_hot",
    "OrdinalEncoder": "ordinal_encoding", "LabelEncoder": "ordinal_encoding",
    "TargetEncoder": "ordinal_encoding",
    "StandardScaler": "standardisation", "MinMaxScaler": "min_max",
    "RobustScaler": "robust_scaler", "Normalizer": "standardisation",
    "Pipeline": "pipeline", "make_pipeline": "pipeline",
    "ColumnTransformer": "column_transformer",
    "make_column_transformer": "column_transformer",
    "make_column_selector": "column_transformer",
    "fit_transform": "fit_transform", "transform": "fit_transform",
    "FunctionTransformer": "feature_engineering",
    "cut": "binning", "qcut": "binning", "KBinsDiscretizer": "binning",
    # week 4 - regression
    "LinearRegression": "linear_model",
    "SGDRegressor": "gradient_descent", "SGDClassifier": "gradient_descent",
    "Ridge": "ridge", "RidgeCV": "ridge",
    "Lasso": "lasso", "LassoCV": "lasso", "ElasticNet": "lasso",
    "PolynomialFeatures": "polynomial_features",
    "variance_inflation_factor": "multicollinearity",
    # week 5 - classification
    "LogisticRegression": "logistic_regression",
    "LogisticRegressionCV": "logistic_regression",
    "expit": "sigmoid", "log_loss": "cross_entropy",
    "DecisionBoundaryDisplay": "decision_boundary",
    "KNeighborsClassifier": "knn", "KNeighborsRegressor": "knn",
    "NearestNeighbors": "knn",
    "GaussianNB": "naive_bayes", "MultinomialNB": "naive_bayes",
    "BernoulliNB": "naive_bayes", "ComplementNB": "naive_bayes",
    "SVC": "svm", "SVR": "svm", "LinearSVC": "svm", "LinearSVR": "svm",
    "SMOTE": "smote", "RandomOverSampler": "oversampling",
    "RandomUnderSampler": "undersampling",
    "resample": "resampling_leak",
    # week 6 - trees and ensembles
    "DecisionTreeClassifier": "decision_tree",
    "DecisionTreeRegressor": "decision_tree",
    "plot_tree": "decision_tree", "export_text": "decision_tree",
    "BaggingClassifier": "bagging", "BaggingRegressor": "bagging",
    "RandomForestClassifier": "random_forest",
    "RandomForestRegressor": "random_forest",
    "ExtraTreesClassifier": "random_forest",
    "ExtraTreesRegressor": "random_forest",
    "feature_importances_": "feature_importance",
    "GradientBoostingClassifier": "gradient_boosting",
    "GradientBoostingRegressor": "gradient_boosting",
    "HistGradientBoostingClassifier": "gradient_boosting",
    "HistGradientBoostingRegressor": "gradient_boosting",
    "AdaBoostClassifier": "boosting", "AdaBoostRegressor": "boosting",
    "XGBClassifier": "gradient_boosting", "LGBMClassifier": "gradient_boosting",
    "GridSearchCV": "grid_search", "RandomizedSearchCV": "random_search",
    "HalvingGridSearchCV": "grid_search", "ParameterGrid": "grid_search",
    "permutation_importance": "permutation_importance",
    "partial_dependence": "partial_dependence",
    "PartialDependenceDisplay": "partial_dependence",
    "shap": "shap", "TreeExplainer": "shap",
    # week 7 - unsupervised
    "KMeans": "kmeans", "MiniBatchKMeans": "kmeans",
    "silhouette_score": "silhouette", "silhouette_samples": "silhouette",
    "inertia_": "inertia",
    "AgglomerativeClustering": "hierarchical", "dendrogram": "dendrogram",
    "linkage": "linkage", "DBSCAN": "dbscan", "HDBSCAN": "dbscan",
    "PCA": "pca", "IncrementalPCA": "pca", "TruncatedSVD": "pca",
    "explained_variance_ratio_": "explained_variance",
    "TSNE": "tsne", "UMAP": "umap", "MDS": "manifold",
    "IsolationForest": "isolation_forest",
    "LocalOutlierFactor": "anomaly_detection",
    "OneClassSVM": "novelty", "EllipticEnvelope": "anomaly_detection",
    # week 8 - text, time, leakage, calibration
    "CountVectorizer": "bag_of_words", "TfidfVectorizer": "tfidf",
    "TfidfTransformer": "tfidf", "HashingVectorizer": "bag_of_words",
    "word_tokenize": "tokenisation",
    "shift": "lag_features", "rolling": "rolling_window",
    "TimeSeriesSplit": "temporal_split",
    "adfuller": "stationarity", "autocorrelation_plot": "stationarity",
    "GroupKFold": "group_kfold", "GroupShuffleSplit": "group_kfold",
    "StratifiedGroupKFold": "group_kfold", "LeaveOneGroupOut": "group_kfold",
    "CalibratedClassifierCV": "calibration",
    "calibration_curve": "reliability_curve",
    "CalibrationDisplay": "reliability_curve",
    "brier_score_loss": "brier_score",
    "IsotonicRegression": "isotonic_regression",
    # week 9 - neural networks
    "Perceptron": "perceptron",
    "MLPClassifier": "hidden_layer", "MLPRegressor": "hidden_layer",
    "relu": "relu", "tanh": "tanh", "softmax": "activation",
    # week 10 - shipping
    "joblib": "joblib", "dump": "serialisation", "load_model": "serialisation",
    "pickle": "pickle_risk",
    "ks_2samp": "ks_statistic",
    # week 11 - statistics
    "bootstrap": "resampling", "ttest_ind": "null_hypothesis",
    "ttest_rel": "paired_test", "wilcoxon": "paired_test",
    "chi2_contingency": "null_hypothesis", "norm": "normal_distribution",
    "binom": "binomial", "poisson": "poisson",
}

# The handful where the concept's day is not the day the tool becomes usable.
# Scalers arrive on day 18 and are called with .fit_transform(), which the
# curriculum names on day 19 when pipelines explain why the two halves are
# separate. Flagging the call one day early would be pedantry, not teaching.
_API_DAY: dict[str, int] = {
    "fit_transform": 18,
    "transform": 18,
    # A random forest is built on day 37 and needs to be told how many trees.
    # The curriculum discusses what the number does on day 38, with boosting,
    # where it actually matters — so the concept is a day later than the
    # argument's first honest use.
    "n_estimators": 37,
    "random_state": 5,     # the split introduces it, before seeds are discussed
    "stratify": 5,
    "quantile": 2,         # percentiles are week-1 descriptive statistics
}

# Keyword arguments that give a technique away. Deliberately short: `alpha=`
# means transparency to matplotlib and regularisation to Ridge, and a horizon
# that rejects a shaded plot has stopped being useful.
_KWARGS: dict[str, str] = {
    "stratify": "stratify",
    "random_state": "random_state",
    "class_weight": "class_weight",
    "max_depth": "max_depth",
    "n_estimators": "n_estimators",
    "n_neighbors": "knn",
    "n_clusters": "clustering",
    "n_components": "dimensionality_reduction",
    "early_stopping": "early_stopping",
    "cv": "cross_validation",
    "scoring": "cv_score",
    "param_grid": "grid_search",
    "hidden_layer_sizes": "hidden_layer",
}

# Attribute names common enough elsewhere that matching them would misfire.
_NEVER_FLAG = {"fit", "predict", "transform", "describe", "info", "corr",
               "hist", "quantile", "shift", "rolling", "cut", "load_model"}

# The tools a model reaches for whatever it has been told. Named explicitly in
# every horizon, however far ahead they sit, because the truncated list is what
# let `def` through on day 1 of the Python mentor.
ATTRACTORS = ("train_test_split", "StandardScaler", "Pipeline",
              "cross_val_score", "GridSearchCV", "RandomForestClassifier",
              "classification_report", "roc_auc_score", "OneHotEncoder",
              "LogisticRegression")

# Never allowed on any day. A lesson that downloads its data cannot be verified
# and will break the first time the container has no network, which on the
# hosted plan is every restart.
FORBIDDEN_ALWAYS: dict[str, str] = {
    "read_csv": "read the bundled dataset with core.datasets.load(name)",
    "read_excel": "read the bundled dataset with core.datasets.load(name)",
    "fetch_openml": "use a bundled dataset; nothing may be downloaded",
    "fetch_california_housing": "use a bundled dataset; nothing may be downloaded",
    "fetch_20newsgroups": "use a bundled dataset; nothing may be downloaded",
    "load_dataset": "use core.datasets.load(name)",
    "urlopen": "nothing may be downloaded",
    "requests": "nothing may be downloaded",
    "make_classification": "use a bundled dataset so the numbers are real",
    "make_regression": "use a bundled dataset so the numbers are real",
    "make_blobs": "use a bundled dataset so the numbers are real",
    "torch": "this course is classical ML; torch is not installed",
    "tensorflow": "this course is classical ML; tensorflow is not installed",
    "keras": "this course is classical ML; keras is not installed",
}


def day_for_symbol(symbol: str) -> int:
    """The first day a library symbol or keyword may legitimately appear."""
    if symbol in _API_DAY:
        return _API_DAY[symbol]
    concept = _API.get(symbol) or _KWARGS.get(symbol)
    return introduced_on(concept) if concept else 0


def earliest_day_for(code: str) -> int:
    """The first day on which this code would be legal.

    The inverse of check_code, and the reason the challenge bank can place
    itself. A task that fits a logistic regression cannot be offered in week 2
    however well it illustrates cross-validation, and asking an author to
    remember that is how it goes wrong. Reading it off the code cannot.

    Returns 0 for code that is never legal — a download, or torch.
    """
    hits, error = scan(code or "")
    if error:
        return 1
    day = 1
    for name, _ in hits:
        if name in FORBIDDEN_ALWAYS:
            return 0
        arrives = day_for_symbol(name)
        if arrives:
            day = max(day, arrives)
    return day


def allowed_api(day: int) -> list[str]:
    return sorted(s for s in _API if 0 < day_for_symbol(s) <= day)


def forbidden_api(day: int) -> list[tuple[str, int]]:
    out = [(s, day_for_symbol(s)) for s in _API]
    return sorted(((s, d) for s, d in out if d > day), key=lambda p: (p[1], p[0]))


# ---------------------------------------------------------------------------
# Scanning
# ---------------------------------------------------------------------------

@dataclass
class Violation:
    symbol: str
    label: str
    taught_on_day: int
    line: int
    reason: str = ""

    def __str__(self) -> str:
        if self.reason:
            return f"line {self.line}: {self.symbol} - {self.reason}"
        return (f"line {self.line}: {self.symbol} "
                f"({self.label}, not taught until day {self.taught_on_day})")


class _Scanner(ast.NodeVisitor):
    def __init__(self) -> None:
        self.hits: list[tuple[str, int]] = []

    def _note(self, name: str, node: ast.AST) -> None:
        self.hits.append((name, getattr(node, "lineno", 0)))

    def visit_Name(self, node):
        self._note(node.id, node)
        self.generic_visit(node)

    def visit_Attribute(self, node):
        if node.attr not in _NEVER_FLAG or isinstance(node.ctx, ast.Load):
            self._note(node.attr, node)
        self.generic_visit(node)

    def visit_keyword(self, node):
        if node.arg:
            self._note(node.arg, node)
        self.generic_visit(node)

    def visit_Import(self, node):
        for alias in node.names:
            for part in alias.name.split("."):
                self._note(part, node)

    def visit_ImportFrom(self, node):
        for part in (node.module or "").split("."):
            self._note(part, node)
        for alias in node.names:
            self._note(alias.name, node)


def scan(code: str) -> tuple[list[tuple[str, int]], str | None]:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return [], f"SyntaxError on line {exc.lineno}: {exc.msg}"
    scanner = _Scanner()
    scanner.visit(tree)
    return scanner.hits, None


def check_code(code: str, day: int,
               extra_allowed: tuple[str, ...] = ()) -> list[Violation]:
    """Everything in `code` the reader has not been taught by `day`.

    Unrecognised names pass. See the module docstring: this reader already
    writes Python, so the cost of a false alarm is higher than the cost of a
    miss, and the recognised set is where the effort goes instead.
    """
    hits, error = scan(code)
    if error:
        return []
    allowed = set(extra_allowed)
    seen: set[str] = set()
    out: list[Violation] = []

    for name, line in hits:
        if name in seen or name in allowed:
            continue
        if name in FORBIDDEN_ALWAYS:
            seen.add(name)
            out.append(Violation(name, name, 0, line, FORBIDDEN_ALWAYS[name]))
            continue
        concept = _API.get(name) or _KWARGS.get(name)
        if concept is None:
            continue
        arrives = day_for_symbol(name)
        if arrives == 0 or arrives <= day:
            continue
        seen.add(name)
        out.append(Violation(name, label(concept), arrives, line))

    out.sort(key=lambda v: (v.taught_on_day == 0 and -1 or v.taught_on_day))
    return out


# ---------------------------------------------------------------------------
# Reaching ahead in prose, by hand
# ---------------------------------------------------------------------------
# The symbol scan above catches `from sklearn.model_selection import
# train_test_split` on day 1. It does not catch a day-1 lesson that shuffles
# the rows itself, slices off a quarter, scores the result and calls it "test
# accuracy" — no forbidden symbol appears anywhere, and the reader has still
# been taught four ideas from weeks 1 to 2.
#
# That is exactly what happened, so the phrases are checked too. The map is
# deliberately short and made only of terms that are diagnostic on their own:
# "pipeline" is absent because a data pipeline at work is an innocent
# sentence, and "variance" is absent because it means two different things in
# this course. A phrase that could be said in passing is not worth the false
# alarms.

_PHRASES: dict[str, str] = {
    r"train[-/ ]?test split": "train_test_split",
    # The real lesson wrote "the training subset" and "the test rows", which
    # "training set" alone did not catch. The noun varies; the qualifier does
    # not. "split" on its own stays out — a decision tree splits, and so does
    # a string.
    r"train(ing)? (set|rows|subset|half|portion|part|data)": "train_test_split",
    r"test (set|rows|subset|half|portion|part)": "train_test_split",
    r"random split": "train_test_split",
    r"held[- ]out": "train_test_split",
    r"holdout": "train_test_split",
    r"\baccuracy\b": "accuracy",
    r"\bbaseline\b": "baseline",
    r"overfit(ting|s|ted)?\b": "overfitting",
    r"underfit(ting|s|ted)?\b": "underfitting",
    r"cross[- ]validat(ion|e|ed|ing)": "cross_validation",
    r"confusion matrix": "confusion_matrix",
    r"\bprecision\b": "precision",
    r"\brecall\b": "recall",
    r"\bf1\b": "f1",
    r"roc curve": "roc_curve",
    r"\bauc\b": "auc",
    r"\bthreshold\b": "threshold",
    r"\bimput(e|ed|ing|ation)\b": "imputation",
    r"one[- ]hot": "one_hot",
    r"standardis|standardiz": "standardisation",
    r"linear regression": "linear_model",
    r"\bintercept\b": "intercept",
    r"\bcoefficients?\b": "coefficient",
    r"gradient descent": "gradient_descent",
    r"learning rate": "learning_rate",
    r"regularis|regulariz": "ridge",
    r"logistic regression": "logistic_regression",
    r"decision tree": "decision_tree",
    r"random forest": "random_forest",
    r"\bleakage\b": "data_leakage",
    r"principal component": "pca",
    r"k[- ]means": "kmeans",
    r"\bresiduals?\b": "residual",
    r"r[- ]squared": "r_squared",
    r"\bp[- ]values?\b": "p_value",
}

_PHRASE_PATTERNS = [(re.compile(pattern, re.I), key)
                    for pattern, key in _PHRASES.items()]


def prose_reaches_ahead(text: str, day: int, min_hits: int = 1
                        ) -> list[tuple[str, int, int]]:
    """Ideas from later in the plan that this prose is teaching.

    Returns (concept, the day it arrives, how many times it was said).

    `min_hits` is the whole design. A lesson may fairly say "in three weeks you
    will cross-validate this" — one mention is a signpost, not a lesson. Saying
    it five times is teaching it early. Callers pass a high threshold for the
    main theory, where forward references are legitimate, and 1 for the
    commentary attached to a code cell, where a mention means the code in front
    of the reader is doing the thing.
    """
    # Several phrases map to one idea — "training set" and "held-out" are both
    # the train/test split — so hits are summed per idea rather than reported
    # once per spelling.
    counts: dict[str, int] = {}
    for pattern, key in _PHRASE_PATTERNS:
        arrives = introduced_on(key)
        if arrives == 0 or arrives <= day:
            continue
        hits = len(pattern.findall(text or ""))
        if hits:
            counts[key] = counts.get(key, 0) + hits

    found = [(key, introduced_on(key), hits)
             for key, hits in counts.items() if hits >= min_hits]
    return sorted(found, key=lambda f: f[1])


CODE_FIELDS = ("code", "example", "starter_code", "snippet", "solution",
               "reference_solution", "cell", "setup")
PROSE_FIELDS = ("question", "explanation", "front", "back", "statement",
                "task", "scenario", "theory", "walkthrough")
_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)


def _blocks(item: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for field in CODE_FIELDS:
        value = item.get(field)
        if isinstance(value, str) and value.strip():
            out.append(value)
        elif isinstance(value, list):
            out.extend(v for v in value if isinstance(v, str))
    for field in PROSE_FIELDS:
        value = item.get(field)
        if isinstance(value, str) and "```" in value:
            out.extend(_FENCE.findall(value))
    for step in item.get("steps", []) or []:
        if isinstance(step, dict):
            for key in ("code", "snippet"):
                value = step.get(key)
                if isinstance(value, str) and value.strip():
                    out.append(value)
    return out


def reaches_ahead(item: dict[str, Any], day: int,
                  extra_allowed: tuple[str, ...] = ()) -> str:
    """Name the first thing in a generated item that the reader cannot have met.

    Only code is examined. The prose is left alone on purpose: a day-5 lesson
    may fairly say "in three weeks you will cross-validate this", and scanning
    English for the word "pipeline" would reject a sentence about a data
    pipeline in a factory.
    """
    for block in _blocks(item):
        violations = check_code(block, day, extra_allowed)
        if violations:
            return "; ".join(str(v) for v in violations[:3])
    return ""


# ---------------------------------------------------------------------------
# The prompt fragment
# ---------------------------------------------------------------------------

def horizon(topic: dict[str, Any]) -> str:
    day = int(topic.get("day", 1))
    week = int(topic.get("week", 1))
    taught = known_by(day, "concept")
    upcoming = forbidden_api(day)

    lines = [
        "KNOWLEDGE HORIZON - a hard constraint, not a style note.",
        f"This is day {day} of {PLAN_DAYS}, week {week}. The reader has worked "
        f"through days 1 to {day - 1} and nothing else. They can already "
        "write Python, pandas and numpy; what they cannot do is anything this "
        "plan has not yet taught.",
        "",
    ]

    # Which side to spell out depends on which is shorter. Early on the list of
    # what is known fits easily and is the more useful half; by week 9 the ban
    # list is the short one and the other would run to three hundred entries.
    if len(taught) <= 110:
        lines += [
            "TAUGHT SO FAR, and therefore available:",
            "  " + ", ".join(i.label for i in taught) or "  nothing yet",
            "",
        ]
    else:
        recent = [i for i in taught if i.day > day - 14]
        lines += [
            f"TAUGHT SO FAR: everything in days 1 to {day - 1} of this plan. "
            "Most recently:",
            "  " + ", ".join(i.label for i in recent),
            "",
        ]

    banned = list(upcoming[:12])
    for symbol in ATTRACTORS:
        arrives = day_for_symbol(symbol)
        if arrives > day and all(symbol != s for s, _ in banned):
            banned.append((symbol, arrives))
    banned.sort(key=lambda p: p[1])

    if banned:
        lines += [
            "NOT YET TAUGHT. Using one of these makes the lesson useless to "
            "the reader, so it must not appear anywhere - not in the code, a "
            "comment, an aside, or a self-check question:",
        ]
        for symbol, arrives in banned:
            concept = _API.get(symbol, "")
            lines.append(f"  - {symbol}"
                         + (f" ({label(concept)})" if concept else "")
                         + f" - day {arrives}")
        if len(banned) < len(upcoming):
            lines.append("  - ...and everything else later in the plan.")
        lines.append("")

    lines += [
        "Do not route around this by writing a future technique out by hand. "
        "Splitting the rows yourself, scoring how many predictions were right, "
        "counting right and wrong answers into a table, or picking a cut-off "
        "for a probability are not neutral moves: each is an idea with its own "
        "day, and doing it without naming it teaches it just as much as "
        "importing it would. If today's material cannot carry the lesson on "
        "its own, the lesson is about today's material anyway.",
        "",
        "NEVER, on any day: read_csv, any download, any fetch_*, "
        "make_classification (load data only with  from core.datasets import "
        "load); torch, tensorflow, keras (not installed).",
        "",
        "If the topic seems thin with only what is available, go deeper rather "
        "than forward. A full day on what a train/test split protects against "
        "beats a preview of cross-validation the reader cannot follow.",
    ]
    return "\n".join(lines)


def maths_offered(topic: dict[str, Any]) -> list[str]:
    """The mathematics this day leans on, for the Math Helper's opening offer.

    The student said plainly that maths is where they struggle. Rather than
    waiting to be asked, every lesson surfaces the terms it is about to use, so
    the question can be clicked instead of formulated.
    """
    return list(topic.get("maths", ()))


def summary_for(topic: dict[str, Any]) -> dict[str, Any]:
    day = int(topic.get("day", 1))
    return {
        "day": day,
        "available_api": allowed_api(day),
        "forbidden_api": [{"symbol": s, "day": d} for s, d in forbidden_api(day)],
        "concepts_known": [i.key for i in known_by(day, "concept")],
        "maths_known": [i.key for i in known_by(day, "maths")],
    }


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------

def validate() -> list[str]:
    """Problems that would make the horizon quietly wrong.

    Run by the test suite. The Python mentor needed this because its table of
    days was written by hand; here the days are derived, so what can still go
    wrong is a symbol pointing at a concept the curriculum does not teach, or
    a prerequisite naming a topic that no longer exists.
    """
    problems: list[str] = []

    for symbol, concept in sorted(_API.items()):
        if concept not in EVERYTHING:
            problems.append(f"API '{symbol}' maps to unknown concept "
                            f"'{concept}'")
    for kwarg, concept in sorted(_KWARGS.items()):
        if concept not in EVERYTHING:
            problems.append(f"keyword '{kwarg}' maps to unknown concept "
                            f"'{concept}'")
    for symbol in _API_DAY:
        if symbol not in _API and symbol not in _KWARGS:
            problems.append(f"day override for '{symbol}', which is not a "
                            "recognised symbol")

    slugs = {t["slug"] for t in curriculum.TOPICS}
    for topic in curriculum.TOPICS:
        for prereq in topic.get("prereqs", ()):
            if prereq not in slugs:
                problems.append(f"{topic['slug']}: unknown prerequisite "
                                f"'{prereq}'")
            elif any(t["slug"] == prereq and t["day"] >= topic["day"]
                     for t in curriculum.TOPICS):
                problems.append(f"{topic['slug']} (day {topic['day']}): "
                                f"prerequisite '{prereq}' is not earlier")

    # A symbol whose concept arrives after the symbol is unusable is a
    # curriculum ordering bug, not a mapping one, and worth naming as such.
    for symbol, concept in sorted(_API.items()):
        if concept in EVERYTHING and symbol in _API_DAY:
            if _API_DAY[symbol] > EVERYTHING[concept].day:
                problems.append(f"override for '{symbol}' is later than the "
                                f"concept '{concept}' it belongs to")

    return problems
