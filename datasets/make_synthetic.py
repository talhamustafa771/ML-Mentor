"""Builds the two synthetic datasets, reproducibly.

Why synthetic rather than a famous one: with generated data the true
relationship is known. When a lesson fits a regression and reads off a
coefficient, it can say whether the model recovered the real effect or not —
which is the whole question in applied ML and one that Titanic or Boston
housing can never answer, because nobody knows their true data-generating
process either.

They are built to contain the problems a beginner must learn to handle, each
one deliberate and documented below: missing values that are not missing at
random, a categorical with a rare level, a skewed target, outliers, a
near-duplicate pair of columns, and one feature that leaks the answer.

Run from the project root to regenerate:  python datasets/make_synthetic.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
SEED = 20260920


# ---------------------------------------------------------------------------
# Classification: would this customer leave?
# ---------------------------------------------------------------------------
# True model, for the answer key:
#   churn is driven by tenure (strongly negative), monthly_charges (positive),
#   support_calls (positive) and contract type. age has NO real effect, so a
#   model that finds one has found noise. total_charges is tenure ×
#   monthly_charges plus noise — a near-duplicate that wrecks coefficient
#   interpretation. last_call_outcome is recorded after the decision to leave,
#   so it leaks: it predicts almost perfectly and is useless in production.

CHURN_TRUTH = {
    "tenure_months": -0.055,
    "monthly_charges": 0.021,
    "support_calls": 0.38,
    "contract=month-to-month": 0.95,
    "contract=one-year": 0.10,
    "contract=two-year": 0.0,
    "age": 0.0,
    "intercept": -1.35,
}


def make_churn(n: int = 4200) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)

    tenure = rng.gamma(2.0, 12.0, n).clip(0, 72).round().astype(int)
    age = rng.normal(46, 15, n).clip(18, 88).round().astype(int)
    monthly = (rng.normal(68, 26, n) + 0.12 * tenure).clip(18, 160).round(2)
    calls = rng.poisson(0.8 + 0.035 * (72 - tenure) / 10, n)

    contract = rng.choice(["month-to-month", "one-year", "two-year"],
                          n, p=[0.55, 0.27, 0.18])
    # A rare level, so the student meets a category with almost no data in it.
    internet = rng.choice(["fiber", "dsl", "none", "satellite"],
                          n, p=[0.46, 0.38, 0.14, 0.02])

    logit = (CHURN_TRUTH["intercept"]
             + CHURN_TRUTH["tenure_months"] * tenure
             + CHURN_TRUTH["monthly_charges"] * monthly
             + CHURN_TRUTH["support_calls"] * calls
             + np.select([contract == "month-to-month", contract == "one-year"],
                         [CHURN_TRUTH["contract=month-to-month"],
                          CHURN_TRUTH["contract=one-year"]], 0.0))
    churn = rng.binomial(1, 1 / (1 + np.exp(-logit)))

    frame = pd.DataFrame({
        "customer_id": [f"C{i:05d}" for i in range(n)],
        "age": age,
        "tenure_months": tenure,
        "monthly_charges": monthly,
        # Near-duplicate of tenure × monthly. Correlated enough to destabilise
        # a linear model's coefficients without being an exact copy.
        "total_charges": (tenure * monthly + rng.normal(0, 45, n)).round(2),
        "support_calls": calls,
        "contract": contract,
        "internet_service": internet,
        "churned": churn,
    })

    # Leakage: recorded only after the customer decided. Predicts churn almost
    # perfectly and would be unavailable when a real prediction is needed.
    frame["last_call_outcome"] = np.where(
        churn == 1,
        rng.choice(["cancelled", "escalated", "unresolved"], n, p=[.7, .2, .1]),
        rng.choice(["resolved", "no_contact", "escalated"], n, p=[.55, .4, .05]))

    # Missing values, and deliberately NOT at random: short-tenure customers
    # are likelier to have no billing total yet. Dropping those rows biases the
    # sample, which is the lesson.
    missing_total = rng.random(n) < (0.16 * np.exp(-tenure / 20))
    frame.loc[missing_total, "total_charges"] = np.nan
    frame.loc[rng.random(n) < 0.04, "monthly_charges"] = np.nan
    frame.loc[rng.random(n) < 0.02, "contract"] = None

    # A handful of impossible values, so cleaning has something to catch.
    frame.loc[rng.choice(n, 14, replace=False), "age"] = -1
    frame.loc[rng.choice(n, 9, replace=False), "monthly_charges"] = 9999.0

    return frame


# ---------------------------------------------------------------------------
# Regression: what should this house sell for?
# ---------------------------------------------------------------------------
# True model: price rises with area and quality, falls with age, and the
# neighbourhood effect is a fixed premium. The relationship is multiplicative,
# so the residuals fan out badly until the target is logged — which is the
# point of the lesson on transforming a skewed target.

HOUSE_TRUTH = {
    "form": "price = exp(11.1 + 0.00042*area_sqft + 0.14*quality "
            "- 0.006*age_years + neighbourhood_premium) * noise",
    "neighbourhood_premium": {"riverside": 0.34, "central": 0.22,
                              "suburb": 0.0, "industrial": -0.19},
    "note": "bedrooms has no independent effect once area is known",
}


def make_houses(n: int = 2600) -> pd.DataFrame:
    rng = np.random.default_rng(SEED + 1)

    area = rng.lognormal(7.35, 0.38, n).clip(380, 9000).round().astype(int)
    quality = rng.integers(1, 11, n)
    age = rng.gamma(2.2, 11, n).clip(0, 120).round().astype(int)
    hood = rng.choice(["suburb", "central", "riverside", "industrial"],
                      n, p=[0.44, 0.28, 0.16, 0.12])
    # Determined by area, so it adds nothing once area is in the model.
    bedrooms = np.clip((area / 620 + rng.normal(0, 0.7, n)).round(), 1, 8).astype(int)

    premium = np.select(
        [hood == "riverside", hood == "central", hood == "industrial"],
        [HOUSE_TRUTH["neighbourhood_premium"]["riverside"],
         HOUSE_TRUTH["neighbourhood_premium"]["central"],
         HOUSE_TRUTH["neighbourhood_premium"]["industrial"]], 0.0)

    log_price = (11.1 + 0.00042 * area + 0.14 * quality
                 - 0.006 * age + premium + rng.normal(0, 0.19, n))
    price = np.exp(log_price).round(-2).astype(int)

    frame = pd.DataFrame({
        "house_id": [f"H{i:05d}" for i in range(n)],
        "area_sqft": area,
        "bedrooms": bedrooms,
        "quality_score": quality,
        "age_years": age,
        "neighbourhood": hood,
        "has_garage": rng.binomial(1, 0.62, n),
        "price": price,
    })

    frame.loc[rng.random(n) < 0.07, "quality_score"] = np.nan
    frame.loc[rng.random(n) < 0.03, "age_years"] = np.nan
    # A few genuine outliers: large, expensive, and real.
    big = rng.choice(n, 11, replace=False)
    frame.loc[big, "area_sqft"] = rng.integers(11000, 22000, 11)
    frame.loc[big, "price"] = (frame.loc[big, "price"] * 3.1).astype(int)

    return frame


def main() -> None:
    for name, frame in (("customer_churn", make_churn()),
                        ("house_prices", make_houses())):
        path = OUT / f"{name}.csv"
        frame.to_csv(path, index=False)
        print(f"{name:18s} {frame.shape[0]:5d} x {frame.shape[1]:2d}  "
              f"{path.stat().st_size/1024:7.1f} KB  "
              f"missing={int(frame.isna().sum().sum())}")


if __name__ == "__main__":
    main()
