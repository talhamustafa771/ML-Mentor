"""The datasets every lesson runs against.

Bundled as CSV rather than downloaded. Three reasons, all learned the hard way
on the Python mentor: a hosted container wipes its disk on every restart, so a
download would repeat forever; a dataset that moves or changes silently breaks
the lesson that was written against it; and the verifier can only check a code
cell by running it, which it cannot do if the data is not there.

Lessons call `load("customer_churn")`. The sandbox makes this module importable
inside generated code, so a snippet reads the same in a lesson, in the notebook
and in the student's own experiments.

Each dataset carries a description of what it is *for*: which problems it
contains, and therefore which lesson it belongs to. The generator is given that
description so it picks a dataset that can actually demonstrate the point.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).resolve().parent.parent / "datasets"


@dataclass(frozen=True)
class Dataset:
    name: str
    task: str                    # classification | regression | unsupervised
    target: str
    rows: int
    summary: str
    teaches: tuple[str, ...]     # the problems it was chosen to contain
    truth: str = ""              # the real relationship, where it is known
    notes: str = ""

    @property
    def path(self) -> Path:
        return DATA_DIR / f"{self.name}.csv"


CATALOGUE: dict[str, Dataset] = {
    "customer_churn": Dataset(
        name="customer_churn", task="classification", target="churned",
        rows=4200,
        summary="Subscription customers, and whether each one left.",
        teaches=("missing values that are not missing at random",
                 "categorical encoding, including a rare level",
                 "target leakage", "correlated near-duplicate features",
                 "impossible values that survive a naive clean",
                 "class balance and why accuracy misleads"),
        truth="Churn genuinely depends on tenure, monthly charges, support "
              "calls and contract type. Age has no real effect at all, so a "
              "model that finds one has fitted noise. last_call_outcome is "
              "recorded after the customer decided and leaks the answer.",
        notes="Synthetic, so the true effects are known and a model can be "
              "checked against them rather than merely scored.",
    ),
    "house_prices": Dataset(
        name="house_prices", task="regression", target="price", rows=2600,
        summary="Houses and what they sold for.",
        teaches=("a skewed target and why logging it helps",
                 "genuine outliers that must not simply be deleted",
                 "a feature made redundant by another",
                 "residual plots that fan out",
                 "interpreting coefficients on a log scale"),
        truth="Price rises multiplicatively with area and quality and falls "
              "with age, with a fixed premium per neighbourhood. Bedrooms is "
              "derived from area and adds nothing once area is known — with "
              "one instructive exception: because it is capped at eight, it "
              "quietly flags the eleven enormous houses, so it appears to "
              "help until those are handled. Note also that quality "
              "correlates with raw price more strongly than area does, "
              "because area's effect is multiplicative and a straight-line "
              "correlation understates it.",
        notes="Synthetic, with the generating equation recorded in "
              "datasets/make_synthetic.py.",
    ),
    "iris": Dataset(
        name="iris", task="classification", target="species", rows=150,
        summary="Three iris species, four flower measurements.",
        teaches=("a first classifier on data small enough to read",
                 "decision boundaries in two dimensions",
                 "one class that separates cleanly and two that do not"),
        notes="The standard first dataset. Small enough to print whole.",
    ),
    "breast_cancer": Dataset(
        name="breast_cancer", task="classification", target="diagnosis",
        rows=569,
        summary="Cell nucleus measurements from breast masses, benign or "
                "malignant.",
        teaches=("thirty correlated features", "why scaling matters",
                 "precision against recall when one error is worse",
                 "regularisation on wide data"),
        notes="Real measurements. The asymmetry of the two mistakes is the "
              "point: a missed malignancy is not a false alarm.",
    ),
    "wine": Dataset(
        name="wine", task="classification", target="target", rows=178,
        summary="Chemical analysis of wines from three cultivars.",
        teaches=("features on wildly different scales",
                 "three-class problems", "feature importance"),
    ),
    "diabetes": Dataset(
        name="diabetes", task="regression", target="target", rows=442,
        summary="Ten baseline measurements and disease progression a year on.",
        teaches=("regression with a weak signal",
                 "an honest R-squared that is not close to 1",
                 "why more features stop helping"),
        notes="Useful precisely because the achievable fit is poor. A model "
              "that looks bad here is often correct.",
    ),
    "digits": Dataset(
        name="digits", task="classification", target="target", rows=1797,
        summary="Handwritten digits as 8x8 grids of pixel values.",
        teaches=("images as vectors", "PCA and what it discards",
                 "clustering without labels", "confusion between similar classes"),
    ),
}


def load(name: str):
    """The dataset as a pandas DataFrame.

    This is what generated code calls. It is deliberately the only way in, so
    a lesson can never depend on a file path that exists on one machine.
    """
    import pandas as pd                    # noqa: PLC0415 - heavy, load lazily

    if name not in CATALOGUE:
        raise KeyError(
            f"No dataset called {name!r}. Available: "
            + ", ".join(sorted(CATALOGUE))
        )
    path = CATALOGUE[name].path
    if not path.exists():
        raise FileNotFoundError(
            f"{path.name} is missing from the datasets folder. Run "
            "`python datasets/make_synthetic.py` to rebuild the generated ones."
        )
    return pd.read_csv(path)


def describe(name: str) -> str:
    """What this dataset is for, as a prompt fragment for the generator."""
    d = CATALOGUE[name]
    parts = [
        f"DATASET: {d.name} ({d.rows} rows, {d.task}, target column "
        f"'{d.target}')",
        d.summary,
        "It was chosen because it contains: " + "; ".join(d.teaches) + ".",
    ]
    if d.truth:
        parts.append("Ground truth: " + d.truth)
    if d.notes:
        parts.append("Note: " + d.notes)
    parts.append(
        "Load it with:  from core.datasets import load\\n"
        f"               df = load('{d.name}')\\n"
        "Never read a CSV by path and never download anything."
    )
    return "\n".join(parts)


def for_task(task: str) -> list[Dataset]:
    return [d for d in CATALOGUE.values() if d.task == task]


def available() -> list[str]:
    return sorted(d.name for d in CATALOGUE.values() if d.path.exists())


def missing() -> list[str]:
    return sorted(d.name for d in CATALOGUE.values() if not d.path.exists())


def columns(name: str) -> list[str]:
    """Column names without loading the whole file, for prompt context."""
    path = CATALOGUE[name].path
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        header = handle.readline().strip()
    return [c.strip() for c in header.split(",")]
