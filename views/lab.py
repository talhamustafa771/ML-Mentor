"""The lab.

A notebook with the bundled datasets already available and nothing else to set
up. Cells run in order and share a namespace, exactly as a lesson's do, so
anything you learn here transfers directly.

It is here because the fastest way to understand a result is to change one
thing and look again, and a lesson page is the wrong place to be destructive.
"""
from __future__ import annotations

import streamlit as st

from core import datasets, db, gamify, sandbox, scheduler, ui
from core.config import LEARNING

settings = ui.page("Lab")

ui.hero("The lab",
        "Every bundled dataset, no setup, nothing to break. Cells run in "
        "order and share their variables.",
        eyebrow="Scratch space")
st.write("")

STARTERS: dict[str, list[str]] = {
    "Look at a dataset": [
        "from core.datasets import load\n"
        "df = load('customer_churn')\n"
        "print(df.shape)\n"
        "print(df.dtypes)",
        "print(df.describe().round(2))\n"
        "print()\n"
        "print(df.isna().sum())",
    ],
    "Draw a distribution": [
        "import matplotlib.pyplot as plt\n"
        "from core.datasets import load\n"
        "df = load('house_prices')",
        "import numpy as np\n"
        "fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))\n"
        "axes[0].hist(df['price'], bins=45)\n"
        "axes[0].set_title('price, as it comes')\n"
        "axes[1].hist(np.log(df['price']), bins=45)\n"
        "axes[1].set_title('price, logged')\n"
        "print('skew before:', round(df['price'].skew(), 2))\n"
        "print('skew after: ', round(np.log(df['price']).skew(), 2))",
    ],
    "Fit something and score it": [
        "from core.datasets import load\n"
        "from sklearn.model_selection import train_test_split\n"
        "df = load('customer_churn')\n"
        "X = df[['tenure_months', 'monthly_charges', 'support_calls']].fillna(0)\n"
        "y = df['churned']\n"
        "X_tr, X_te, y_tr, y_te = train_test_split(\n"
        "    X, y, test_size=0.25, stratify=y, random_state=0)\n"
        "print(len(X_tr), 'train rows,', len(X_te), 'test rows')",
        "from sklearn.linear_model import LogisticRegression\n"
        "from sklearn.metrics import roc_auc_score\n"
        "model = LogisticRegression(max_iter=2000).fit(X_tr, y_tr)\n"
        "auc = roc_auc_score(y_te, model.predict_proba(X_te)[:, 1])\n"
        "print('test AUC:', round(auc, 3))\n"
        "for name, weight in zip(X.columns, model.coef_[0]):\n"
        "    print(f'{name:>18}: {weight:+.4f}')",
    ],
    "Empty": ["from core.datasets import load\n\n"],
}

left, right = st.columns([2, 1])
with left:
    starter = st.selectbox("Start from", list(STARTERS))
with right:
    st.write("")
    if st.button("Load it", use_container_width=True):
        st.session_state["lab_cells"] = list(STARTERS[starter])
        st.session_state.pop("lab_result", None)
        st.rerun()

cells = st.session_state.setdefault("lab_cells", list(STARTERS["Empty"]))
result = st.session_state.get("lab_result")

st.write("")
for index, code in enumerate(cells):
    st.markdown(f'<span class="cell-number">{index + 1}</span>',
                unsafe_allow_html=True)
    cells[index] = st.text_area(
        f"Cell {index + 1}", value=code, height=max(110, 23 * (code.count("\n") + 3)),
        key=f"lab-cell-{index}", label_visibility="collapsed")

    if result is not None and index < len(result.cells):
        output = result.cells[index]
        if output.skipped:
            st.caption("Not run — an earlier cell failed.")
        else:
            if output.stdout.strip():
                st.code(output.stdout.rstrip(), language="text")
            for figure in output.figures:
                ui.figure(figure)
            if output.error:
                st.error(output.error)
    st.write("")

add, remove, run, clear = st.columns([1, 1, 2, 1])
with add:
    if st.button("Add a cell", use_container_width=True):
        cells.append("")
        st.session_state["lab_cells"] = cells
        st.rerun()
with remove:
    if st.button("Remove last", use_container_width=True, disabled=len(cells) < 2):
        cells.pop()
        st.session_state["lab_cells"] = cells
        st.rerun()
with run:
    if st.button("Run everything", type="primary", use_container_width=True):
        st.session_state["lab_cells"] = cells
        with st.spinner("Running…"):
            outcome = sandbox.run_notebook(
                cells, timeout=LEARNING.sandbox_timeout_seconds)
        st.session_state["lab_result"] = outcome
        topic = scheduler.next_topic()
        if topic:
            db.log_attempt(topic_slug=topic["slug"], kind="cell",
                           correct=outcome.ok, score=1.0 if outcome.ok else 0.0)
            gamify.award("experiment", reason="Ran an experiment in the lab",
                         topic_slug=topic["slug"],
                         dedupe_key=f"lab:{db.today()}:{len(cells)}")
        st.rerun()
with clear:
    if st.button("Clear output", use_container_width=True):
        st.session_state.pop("lab_result", None)
        st.rerun()

if result is not None:
    if result.ok:
        st.success(f"Ran {len(result.cells)} cell(s) in {result.seconds:.1f}s.")
    else:
        st.error(f"Stopped at cell {result.failed_index + 1}.")


# ---------------------------------------------------------------------------
st.write("")
st.markdown("---")
ui.section("What is loaded and ready")

for name in datasets.available():
    spec = datasets.CATALOGUE[name]
    with st.expander(f"{name}  ·  {spec.rows} rows  ·  {spec.task}"):
        st.markdown(spec.summary)
        st.code(f"from core.datasets import load\ndf = load('{name}')",
                language="python")
        st.markdown("**Columns**")
        st.caption(", ".join(datasets.columns(name)))
        st.markdown("**It was chosen because it contains**")
        for teaches in spec.teaches:
            st.markdown(f"- {teaches}")
        if spec.truth:
            st.markdown("**What is actually true of it**")
            st.caption(spec.truth)

missing = datasets.missing()
if missing:
    st.warning("Missing from the datasets folder: " + ", ".join(missing)
               + ". Run `python datasets/make_synthetic.py` to rebuild the "
                 "generated ones.")

st.caption("There is no internet here and no file access: read_csv, requests "
           "and downloads are blocked, deliberately. Everything you need is "
           "already on disk.")
