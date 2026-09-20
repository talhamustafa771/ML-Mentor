"""Settings: API keys, models, sandbox limits, and data management."""
from __future__ import annotations

import json

import streamlit as st

from core import challenges, concepts, curriculum, datasets, db
from core import llm, mathpad, sandbox, ui
from core import config
from core.config import PROVIDER_ORDER, PROVIDERS, resolve_api_key

settings = ui.page("Settings")
st.title("Settings")

st.caption(f"{len(curriculum.TOPICS)} days · {len(challenges.CHALLENGES)} "
           f"challenges · {mathpad.bank_size()['entries']} maths terms written "
           f"in · {len(datasets.available())} bundled datasets.")

# Which copy of the app is this? On a phone over Wi-Fi and on the hosted
# version the pages look identical, so the one thing worth saying on sight
# is where the study record is being written.
st.caption(f"Your progress is stored in {config.database_location()}.")

tabs = st.tabs(["AI providers", "Models", "Sandbox", "Your data"])

# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------
with tabs[0]:
    st.markdown(
        "This app works with **Groq** and **NVIDIA Build**. Both speak the same "
        "API protocol and both have free tiers, so one key is enough and two "
        "means the app fails over automatically when one hits its rate limit. "
        "Keys are stored with your data and never sent "
        "anywhere except to the provider you configured."
    )

    for provider_id in PROVIDER_ORDER:
        provider = PROVIDERS[provider_id]
        with st.container(border=True):
            head, state = st.columns([3, 1])
            head.markdown(f"**{provider.label}**")
            head.caption(f"`{provider.base_url}` · [Get a free key]"
                         f"({provider.signup_url})")
            current = resolve_api_key(provider_id, settings)
            state.markdown(
                ui.pill("configured", "good") if current
                else ui.pill("not set"),
                unsafe_allow_html=True,
            )

            entered = st.text_input(
                f"{provider.label} API key", value="", type="password",
                key=f"key_{provider_id}",
                placeholder="paste the key here" if not current
                            else "configured — paste a new key to replace it",
            )
            row = st.columns([1, 1, 3])
            if row[0].button("Save", key=f"save_{provider_id}",
                             use_container_width=True, disabled=not entered):
                db.set_setting(provider.key_name, entered.strip())
                llm.forget_bad_keys()
                st.success(f"{provider.label} key saved.")
                st.rerun()
            if row[1].button("Test", key=f"test_{provider_id}",
                             use_container_width=True, disabled=not current):
                with st.spinner("Connecting…"):
                    ok, payload = llm.list_models(provider_id, settings)
                if ok:
                    st.success(f"Connected. {len(payload)} model(s) available.")
                    st.session_state[f"models_{provider_id}"] = payload
                else:
                    st.error(f"Failed: {payload}")
            if current and row[2].button("Remove key", key=f"del_{provider_id}"):
                db.execute("DELETE FROM settings WHERE key=?", (provider.key_name,))
                llm.forget_bad_keys()
                st.rerun()

    st.markdown("#### Failover order")
    order = st.multiselect(
        "Providers are tried in this order", PROVIDER_ORDER,
        default=[p for p in (settings.get("PROVIDER_ORDER") or
                             ",".join(PROVIDER_ORDER)).split(",")
                 if p in PROVIDERS],
        format_func=lambda p: PROVIDERS[p].label,
    )
    if st.button("Save the order") and order:
        db.set_setting("PROVIDER_ORDER", ",".join(order))
        st.success("Saved.")

    st.markdown("#### Status")
    if st.button("Check both providers"):
        with st.spinner("Checking…"):
            for label, state in llm.health_check(settings).items():
                (st.success if "connected" in state else
                 st.warning if state == "no key" else st.error)(f"{label}: {state}")

    st.divider()
    st.markdown("#### Diagnose a generation failure")
    st.caption(
        "Runs a real request at each stage and reports the provider's own "
        "answer: can it be reached, is the key accepted, does the configured "
        "model still exist, does a plain call work, does a structured (JSON) "
        "call work. Four small requests. Your key is never displayed."
    )

    target = st.selectbox(
        "Provider to test", [p for p in PROVIDER_ORDER
                             if resolve_api_key(p, settings)] or PROVIDER_ORDER,
        format_func=lambda p: PROVIDERS[p].label, key="diag_provider",
    )
    if st.button("Run the diagnosis", type="primary"):
        for tier in ("fast", "strong"):
            with st.spinner(f"Testing the {tier} model…"):
                report = llm.smoke_test(target, tier, settings)
            with st.container(border=True):
                st.markdown(f"**{tier.title()} model**")
                if report.get("stage") == "key":
                    st.error(report["error"])
                elif report.get("stage") == "models":
                    st.error(f"Could not list models: {report['error']}")
                    st.caption(
                        "That usually means the key is wrong, or a firewall, "
                        "VPN or corporate proxy is blocking the provider."
                    )
                else:
                    cols = st.columns(3)
                    cols[0].metric("Models offered", report["models_available"])
                    cols[1].metric("Configured model exists",
                                   "yes" if report["model_in_live_list"] else "NO")
                    cols[2].metric("Overall",
                                   "working" if report["ok"] else "failing")
                    st.markdown(f"- Configured: `{report['configured_model']}`")
                    if not report["model_in_live_list"]:
                        st.warning(
                            "The configured model is not in the provider's live "
                            "list — this is almost certainly your problem. "
                            + (f"Closest live matches: "
                               f"{', '.join(report['closest_matches'])}"
                               if report["closest_matches"] else
                               "Pick one from the Models tab.")
                        )
                    st.markdown(f"- Plain text call: {report['plain_text_call']}")
                    st.markdown(f"- JSON call: {report['json_call']}")
                    if report.get("model_actually_used"):
                        st.markdown(f"- Model actually used: "
                                    f"`{report['model_actually_used']}`")
                    for repair in report.get("repairs_applied", []):
                        st.info(f"Repaired automatically: {repair}")

    learned = llm.diagnostics()
    if any(learned.values()):
        with st.expander("What this session has learned about your providers"):
            st.caption(
                "The app repairs recognised failures rather than reporting "
                "them. Anything it changed on your behalf is listed here so a "
                "silent substitution is never invisible."
            )
            st.json(learned)

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
with tabs[1]:
    st.markdown(
        "Two tiers are used. **Fast** runs the high-volume work — multiple-choice "
        "questions, flashcards, hints and tutor chat. **Strong** runs code review, "
        "problem generation and exams, where a weak model would teach you "
        "something false about complexity.\n\n"
        "Providers retire model names periodically. Press **Refresh** to read the "
        "live list rather than editing any code."
    )

    for provider_id in PROVIDER_ORDER:
        provider = PROVIDERS[provider_id]
        with st.expander(provider.label,
                         expanded=bool(resolve_api_key(provider_id, settings))):
            live = st.session_state.get(f"models_{provider_id}")
            if st.button(f"Refresh {provider.label} model list",
                         key=f"refresh_{provider_id}"):
                ok, payload = llm.list_models(provider_id, settings)
                if ok:
                    st.session_state[f"models_{provider_id}"] = payload
                    live = payload
                    st.success(f"{len(payload)} model(s).")
                else:
                    st.error(payload)

            for tier, default in (("fast", provider.default_fast_model),
                                  ("strong", provider.default_strong_model)):
                key_name = f"MODEL_{provider_id.upper()}_{tier.upper()}"
                current = settings.get(key_name, default)
                if live:
                    options = sorted(set(live) | {current})
                    picked = st.selectbox(
                        f"{tier.title()} model", options,
                        index=options.index(current), key=f"sel_{key_name}",
                    )
                else:
                    picked = st.text_input(f"{tier.title()} model", value=current,
                                           key=f"txt_{key_name}")
                if picked != current:
                    db.set_setting(key_name, picked)
                    st.toast(f"{tier} model set to {picked}")

            st.caption(f"Defaults: fast `{provider.default_fast_model}`, "
                       f"strong `{provider.default_strong_model}`")

# ---------------------------------------------------------------------------
# Sandbox
# ---------------------------------------------------------------------------
with tabs[2]:
    st.markdown(
        "Your code runs in a separate Python process with a wall-clock timeout, "
        "so an infinite loop cannot hang the app. A static scan also rejects a "
        "short list of genuinely dangerous calls before anything executes."
    )
    notes = sandbox.environment_notes()
    st.json(notes)

    if not notes["cpu_and_memory_limits"]:
        st.warning(
            "On Windows there is no resource-limit API, so only the wall-clock "
            "timeout applies — a runaway allocation is stopped by the operating "
            "system rather than by a memory cap. This is stated plainly rather "
            "than glossed over: for code you wrote yourself it is not a "
            "practical risk, but do not point this sandbox at code from "
            "elsewhere."
        )

    st.markdown("#### Blocked in generated and submitted code")
    st.caption("These are blocked to prevent accidents, not attackers.")
    st.code(
        "modules: " + ", ".join(sorted(sandbox.BLOCKED_MODULES)) + "\n"
        "calls:   " + ", ".join(f"{m}.{f}" for m, f in sorted(sandbox.BLOCKED_CALLS))
        + "\nbuiltins: " + ", ".join(sorted(sandbox.BLOCKED_NAMES)),
        language="text",
    )

    st.markdown("#### Try it")
    st.caption("This is the same runner every lesson's code goes through. If "
               "it works here, a lesson will run.")
    test_code = st.text_area(
        "Run a snippet",
        "from core.datasets import load\n"
        "import matplotlib.pyplot as plt\n"
        "df = load('customer_churn')\n"
        "print(df.shape, 'churn rate', round(df['churned'].mean(), 3))\n"
        "fig, ax = plt.subplots()\n"
        "ax.hist(df['tenure_months'], bins=30)\n"
        "ax.set_title('tenure')\n", height=190,
    )
    if st.button("Run it"):
        with st.spinner("Running…"):
            result = sandbox.run_notebook([test_code], timeout=90)
        if result.ok:
            cell = result.cells[0]
            if cell.stdout.strip():
                st.code(cell.stdout.rstrip(), language="text")
            for figure in cell.figures:
                ui.figure(figure)
            st.success(f"Sandbox working — ran in {result.seconds:.1f}s, "
                       f"{result.figure_count()} figure(s) captured.")
        else:
            st.error(result.first_error or result.message)

    st.markdown("#### Bundled datasets")
    rows = [{"Dataset": name,
             "Rows": datasets.CATALOGUE[name].rows,
             "Task": datasets.CATALOGUE[name].task,
             "Target": datasets.CATALOGUE[name].target}
            for name in datasets.available()]
    ui.table(rows, hide_index=True)
    if datasets.missing():
        st.warning("Missing: " + ", ".join(datasets.missing())
                   + ". Run `python datasets/make_synthetic.py`.")

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
with tabs[3]:
    counts = {
        "topics": len(db.all_topics()),
        "attempts": db.query_one("SELECT COUNT(*) n FROM attempts")["n"],
        "submissions": db.query_one("SELECT COUNT(*) n FROM submissions")["n"],
        "generated items": db.query_one("SELECT COUNT(*) n FROM content")["n"],
        "flashcards in review": db.query_one("SELECT COUNT(*) n FROM reviews")["n"],
        "tutor messages": db.query_one("SELECT COUNT(*) n FROM messages")["n"],
        "maths answers cached": db.query_one(
            "SELECT COUNT(*) n FROM math_cache")["n"],
    }
    ui.table([{"Table": k, "Rows": v} for k, v in counts.items()],
             hide_index=True)
    st.caption(f"Everything lives in `{db.DB_PATH}` on this machine. "
               "Nothing is uploaded.")

    st.download_button(
        "Export my progress as JSON",
        json.dumps({
            "exported_at": db.now(),
            "mastery_by_topic": [
                {"slug": t["slug"], "title": t["title"], "week": t["week"],
                 "score": t.get("score", 0.0), "attempts": t.get("attempts", 0)}
                for t in db.topics_with_mastery()
            ],
            "attempts": db.recent_attempts(100000),
            "submissions": [dict(r) for r in db.query("SELECT * FROM submissions")],
            "reviews": [dict(r) for r in db.query("SELECT * FROM reviews")],
            "mastery": [dict(r) for r in db.query("SELECT * FROM mastery")],
        }, indent=2, default=str),
        file_name="ml-mentor-progress.json", mime="application/json",
    )

    st.markdown("#### Regenerate the curriculum")
    st.caption("Re-reads core/curriculum.py and the challenge bank. Your mastery and history are kept.")
    if st.button("Re-seed topics"):
        problems = curriculum.validate() + concepts.validate()
        if problems:
            st.error("Validation failed:\n" + "\n".join(f"- {p}" for p in problems))
        else:
            written = db.seed_topics(curriculum.TOPICS)
            banked = db.seed_challenge_bank(challenges.CHALLENGES)
            retired = db.retire_unteachable_content()
            st.success(f"{written} days and {banked} challenges re-seeded."
                       + (f" {retired} stale item(s) retired." if retired else ""))

    st.markdown("#### Clear the generated content cache")
    st.caption(
        "Deletes cached lessons, questions and cards so fresh ones are "
        "generated. Your attempts, submissions and mastery are untouched."
    )
    if st.button("Clear the content cache"):
        db.execute("DELETE FROM content")
        db.execute("DELETE FROM reviews")
        st.success("Cache cleared.")

    st.divider()
    st.markdown("#### Reset everything")
    st.error(
        "This permanently deletes every attempt, submission, flashcard "
        "schedule, conversation and mastery score. It cannot be undone. Export "
        "your progress first if you want a copy."
    )
    confirmation = st.text_input('Type RESET to confirm', key="reset_confirm")
    if st.button("Delete all my learning data", type="primary",
                 disabled=confirmation != "RESET"):
        db.reset_all(keep_settings=True)
        for key in list(st.session_state):
            if key != "_db_ready":
                del st.session_state[key]
        st.success("All learning data deleted. Your API keys were kept.")
        st.rerun()
