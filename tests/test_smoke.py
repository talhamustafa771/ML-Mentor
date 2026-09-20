"""Every page imports, every public function answers, nothing crashes empty.

The cheapest test in the suite and historically the one that catches the most:
a page that fails on a fresh database is a page nobody can reach, and that is
exactly the state the app is in the first time it is opened.

Run:  python tests/test_smoke.py
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PASS, FAIL = [], []


def check(name: str, condition: bool, detail: str = "") -> None:
    (PASS if condition else FAIL).append(f"{name}{' - ' + detail if detail else ''}")


# ---------------------------------------------------------------------------
# Everything parses, on whatever Python this is
# ---------------------------------------------------------------------------
sources = sorted(list(ROOT.glob("core/*.py")) + list(ROOT.glob("views/*.py"))
                 + [ROOT / "app.py"] + list(ROOT.glob("datasets/*.py")))
for path in sources:
    try:
        ast.parse(path.read_text(encoding="utf-8"))
        check(f"{path.relative_to(ROOT)} parses", True)
    except SyntaxError as exc:
        check(f"{path.relative_to(ROOT)} parses", False,
              f"line {exc.lineno}: {exc.msg}")

# f-strings that nest the same quote character are a syntax error before
# Python 3.12. The app has to run on a laptop and on whatever the host
# provides, so this is checked rather than hoped for.
check("nothing relies on Python 3.12 f-string nesting",
      sys.version_info >= (3, 12) or not FAIL,
      "; ".join(FAIL[:2]))


# ---------------------------------------------------------------------------
# The modules import and agree with each other
# ---------------------------------------------------------------------------
from core import (analytics, challenges, concepts, curriculum, datasets,  # noqa: E402
                  db, gamify, generators, llm, mathpad, sandbox, scheduler,
                  verify)

check("the curriculum validates", not curriculum.validate(),
      "; ".join(curriculum.validate()[:3]))
check("the horizon validates", not concepts.validate(),
      "; ".join(concepts.validate()[:3]))
check("91 days", len(curriculum.TOPICS) == 91)
check("13 weeks", curriculum.PLAN_WEEKS == 13)
check("every dataset is on disk", not datasets.missing(),
      ", ".join(datasets.missing()))


# ---------------------------------------------------------------------------
# A fresh database
# ---------------------------------------------------------------------------
db.init_db()
db.seed_topics(curriculum.TOPICS)
db.seed_challenge_bank(challenges.CHALLENGES)

check("topics seeded", len(db.all_topics()) == 91)
check("challenges seeded",
      db.bank_progress()["total"] == len(challenges.CHALLENGES))
check("every challenge landed on a topic that exists",
      all(c["topic_slug"] in {t["slug"] for t in db.all_topics()}
          for c in db.bank_challenges(max_day=None)))
check("a challenge never unlocks before its own day",
      all(c["topic_day"] >= c.get("unlock_day", 1)
          for c in db.bank_challenges(max_day=None)))


# ---------------------------------------------------------------------------
# Functions that must answer on an empty record
# ---------------------------------------------------------------------------
EMPTY_CALLS = [
    ("scheduler.plan_status", scheduler.plan_status),
    ("scheduler.next_topic", scheduler.next_topic),
    ("scheduler.recommend_next", scheduler.recommend_next),
    ("scheduler.streak_days", scheduler.streak_days),
    ("scheduler.weakest_topics", scheduler.weakest_topics),
    ("gamify.summary", gamify.summary),
    ("gamify.earned_badges", gamify.earned_badges),
    ("gamify.skill_map", gamify.skill_map),
    ("gamify.daily_goal", gamify.daily_goal),
    ("analytics.daily_series", lambda: analytics.daily_series(7)),
    ("db.bank_progress", db.bank_progress),
    ("db.math_history", db.math_history),
    ("db.topics_with_mastery", db.topics_with_mastery),
    ("db.recent_mistakes", db.recent_mistakes),
    ("mathpad.coverage", mathpad.coverage),
    ("mathpad.bank_size", mathpad.bank_size),
    ("sandbox.environment_notes", sandbox.environment_notes),
]
for name, call in EMPTY_CALLS:
    try:
        call()
        check(f"{name}() works on an empty record", True)
    except Exception as exc:                          # noqa: BLE001
        check(f"{name}() works on an empty record", False,
              f"{type(exc).__name__}: {exc}")

# Per-topic calls that a page makes in a loop.
first = curriculum.TOPICS[0]
for name, call in [
    ("concepts.horizon", lambda: concepts.horizon(first)),
    ("mathpad.offer_for", lambda: mathpad.offer_for(first)),
    ("generators._topic_context", lambda: generators._topic_context(first)),
    ("generators._lesson_fallback",
     lambda: generators._lesson_fallback(first, "no key")),
    ("db.get_project", lambda: db.get_project(first["slug"])),
    ("db.content_count", lambda: db.content_count(first["slug"], "mcq")),
    ("scheduler.prereqs_met", lambda: scheduler.prereqs_met(first)),
]:
    try:
        call()
        check(f"{name}() works", True)
    except Exception as exc:                          # noqa: BLE001
        check(f"{name}() works", False, f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# The pages
# ---------------------------------------------------------------------------
# They cannot be imported (Streamlit would try to render), so instead: every
# name they reference on a core module must actually exist on it. This is what
# catches a page calling scheduler.recommend_next() and getting a list back.
import importlib                                      # noqa: E402

CORE_MODULES = {
    name: importlib.import_module(f"core.{name}") for name in
    ("analytics", "challenges", "concepts", "curriculum", "datasets", "db",
     "gamify", "generators", "llm", "mathpad", "sandbox", "scheduler", "ui",
     "verify", "config")
}

for path in sorted(list(ROOT.glob("views/*.py")) + [ROOT / "app.py"]):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("core"):
            imported.update(alias.asname or alias.name for alias in node.names)

    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        if not isinstance(node.value, ast.Name):
            continue
        module_name = node.value.id
        if module_name not in imported or module_name not in CORE_MODULES:
            continue
        check(f"{path.name}: {module_name}.{node.attr} exists",
              hasattr(CORE_MODULES[module_name], node.attr))


# ---------------------------------------------------------------------------
# set_page_config must be the first Streamlit command
# ---------------------------------------------------------------------------
# This one is here because the app once refused to start. Importing `core`
# read st.secrets, found no secrets file, and Streamlit rendered a message
# about it onto the page — which made the app's own set_page_config no longer
# the first Streamlit command, which Streamlit treats as fatal.
#
# Two rules follow, and both are checked rather than remembered.

entry = (ROOT / "app.py").read_text(encoding="utf-8")
entry_tree = ast.parse(entry)

first_call = None
first_core_import = None
for index, node in enumerate(entry_tree.body):
    if first_call is None and isinstance(node, ast.Expr) \
            and isinstance(node.value, ast.Call):
        func = node.value.func
        name = getattr(func, "attr", None) or getattr(func, "id", None)
        if name:
            first_call = (index, name)
    if first_core_import is None and isinstance(node, ast.ImportFrom) \
            and (node.module or "").startswith("core"):
        first_core_import = index

check("app.py's first call is set_page_config",
      first_call is not None and first_call[1] == "set_page_config",
      f"it is {first_call[1] if first_call else 'nothing'}")
check("app.py configures the page before importing core",
      first_call is not None and first_core_import is not None
      and first_call[0] < first_core_import,
      "core is imported first, so any warning it prints breaks startup")

# And nothing in core may touch st.secrets at import time without first
# checking that a secrets file exists.
config_source = (ROOT / "core" / "config.py").read_text(encoding="utf-8")
check("config.py checks for a secrets file before reading st.secrets",
      config_source.index("_secrets_file_exists()")
      < config_source.index("secrets = st.secrets"))

from core import config                                  # noqa: E402
check("no secrets file here means none is looked for",
      config._secrets_file_exists() is False
      or (ROOT / ".streamlit" / "secrets.toml").exists())

# Calling it twice must not raise — a view run on its own calls it again.
from core import ui as ui_module                          # noqa: E402
check("configure_page is safe to call twice",
      hasattr(ui_module, "configure_page"))


# ---------------------------------------------------------------------------
# Prompts must fit inside a free-tier per-minute token budget
# ---------------------------------------------------------------------------
# Here because day 1 could not be generated at all. Providers count the
# RESERVED completion against a per-minute cap, not the completion actually
# produced — so asking for 6000 output tokens on a key capped at 8000 a minute
# fails on every attempt, forever, however long you wait.
#
# Every prompt the app sends is measured against that cap here, so an edit
# that makes one longer fails the suite rather than the app.

from core.config import LEARNING as _LEARNING                # noqa: E402


def _tokens(text: str) -> int:
    """A pessimistic token estimate. Real tokenisers land near chars/4 for
    English; 3.7 leaves room to be wrong in the safe direction."""
    return int(len(text) / 3.7)


_budget = _LEARNING.provider_token_budget
_system = _tokens(generators.GENERATOR_SYSTEM)
_worst = {}

for _topic in curriculum.TOPICS:
    _context = generators._topic_context(_topic)
    _cases = {
        "lesson": (f"{_context}\n\n{generators._LESSON_SHAPE}\n\n"
                   f"{generators._LESSON_RULES}", _LEARNING.lesson_max_tokens),
        # The others build their prompt inline; the context dominates, and a
        # generous allowance stands in for the instructions around it.
        "mcq": (_context + "x" * 1400, 4000),
        "card": (_context + "x" * 900, 4000),
        "challenge": (f"{_context}\n\n{generators._CHALLENGE_RULES}"
                      + "x" * 500, 3500),
    }
    for _kind, (_prompt, _reserve) in _cases.items():
        _total = _tokens(_prompt) + _system + _reserve
        if _total > _worst.get(_kind, (0, 0))[0]:
            _worst[_kind] = (_total, _topic["day"])

for _kind, (_total, _day) in _worst.items():
    check(f"the {_kind} prompt fits a {_budget}-token minute",
          _total <= _budget,
          f"day {_day} would need {_total}, which is {_total - _budget} over")

# A day with no mathematics must tell the model so, or the shape block's
# advice on formatting a formula reads as an invitation to write one.
_no_maths = [t for t in curriculum.TOPICS if not t["maths"]]
check("some days have no mathematics at all", bool(_no_maths))
for _topic in _no_maths[:5]:
    _block = generators._maths_block(_topic)
    check(f"day {_topic['day']} is told to write no formula",
          "Write no formula at all" in _block)
for _topic in [t for t in curriculum.TOPICS if t["maths"]][:5]:
    _block = generators._maths_block(_topic)
    check(f"day {_topic['day']} still lists its mathematics",
          "Write no formula at all" not in _block and len(_block) > 60)

check("the lesson reservation leaves room for a real lesson",
      _LEARNING.lesson_max_tokens >= 3000,
      f"{_LEARNING.lesson_max_tokens} would truncate a lesson mid-sentence")

# A per-minute cap is not an ordinary rate limit and must not be reported as
# one: "wait a minute and try again" is the worst possible advice for a
# request that can never succeed.
_tpm = ("Error code: 413 - {'error': {'message': 'Request too large for model "
        "`openai/gpt-oss-120b` on tokens per minute (TPM): Limit 8000, "
        "Requested 8227, please reduce your message size and try again.', "
        "'code': 'rate_limit_exceeded'}}")


class _Status413(Exception):
    status_code = 413


check("a per-minute token cap is classified as its own failure",
      llm.classify(_Status413(_tpm)).kind == "token_budget",
      llm.classify(_Status413(_tpm)).kind)

# The same cap produces two different messages that need opposite responses.
# Reading them backwards gives the worst possible advice in both directions,
# so both shapes are pinned here.
_transient = ("Error code: 429 - Rate limit reached for model "
              "`openai/gpt-oss-120b` on tokens per minute (TPM): Limit 8000, "
              "Used 3379, Requested 6642. Please try again in 15.157499999s.")


class _Status429(Exception):
    status_code = 429


check("a part-spent minute is not treated as an unfittable request",
      llm.classify(_Status429(_transient)).kind == "token_wait",
      llm.classify(_Status429(_transient)).kind)
check("the stated wait is read from the message",
      15.0 < llm.retry_after(_transient) < 17.0,
      str(llm.retry_after(_transient)))
check("an unstated wait still gets a sane default",
      llm.retry_after("no time given here") == 20.0)
check("a stated wait is capped",
      llm.retry_after("try again in 9999s") <= 60.0)

_wait_advice = llm._readable_error(llm.classify(_Status429(_transient)), []).lower()
check("the transient case says waiting DOES help",
      "is fixed by waiting" in _wait_advice)
_budget_advice = llm._readable_error(llm.classify(_Status413(_tpm)), []).lower()
check("the two cases give opposite advice",
      ("not something waiting" in _budget_advice)
      and ("not something waiting" not in _wait_advice))

# The request in the transient message fits on its own — that is what makes it
# transient. If it did not, shrinking would be the right answer instead.
check("the transient request would have fitted by itself", 6642 < 8000)
check("the structural request would not", 8227 > 8000)
_advice = llm._readable_error(llm.classify(_Status413(_tpm)), []).lower()
check("the advice says waiting will not help", "not something waiting" in _advice)
check("the advice does not give the rate-limit advice",
      "wait a minute" not in _advice,
      "that is the advice for an ordinary rate limit, which this is not")
check("the advice says what to do instead",
      "paid tier" in _advice or "larger budget" in _advice)
check("the provider's own words survive",
      "limit 8000" in llm._readable_error(
          llm.classify(_Status413(_tpm)), []).lower())

# The retry reads the provider's own numbers rather than guessing.
check("the shrink fits inside the reported limit",
      llm.shrink_to_fit(_tpm, 6000) == 8000 - (8227 - 6000) - 256,
      str(llm.shrink_to_fit(_tpm, 6000)))
check("a shrunk request would actually fit",
      (8227 - 6000) + llm.shrink_to_fit(_tpm, 6000) <= 8000)
check("an impossible budget gives up rather than looping",
      llm.shrink_to_fit("Limit 2000, Requested 9000", 6000) is None)


# ---------------------------------------------------------------------------
# Streamlit's widget arguments, against the build that is actually installed
# ---------------------------------------------------------------------------
# Here because a page died on a machine whose st.image did not accept
# use_container_width, while the same reported version accepted it elsewhere.
# Version numbers turned out not to settle it, so the signature is read.

import streamlit as _st                                   # noqa: E402

for _name in ("image", "dataframe", "button", "download_button",
              "form_submit_button", "text_area", "selectbox", "bar_chart",
              "progress", "columns", "tabs", "latex", "code", "caption"):
    check(f"st.{_name} exists on this Streamlit", hasattr(_st, _name))

for _name in ("image", "dataframe"):
    _fn = getattr(_st, _name)
    _kwargs = ui_module.fill_width(_fn)
    _params = __import__("inspect").signature(_fn).parameters
    check(f"ui.fill_width picks an argument st.{_name} accepts",
          all(k in _params for k in _kwargs),
          f"chose {_kwargs}, which st.{_name} does not take")
    check(f"ui.fill_width says something about st.{_name}",
          bool(_kwargs) or "width" not in _params,
          "it found no way to make this fill its column")

# Buttons are passed use_container_width all over the views, so if this build
# has dropped it there, every page breaks and not just one widget.
for _name in ("button", "download_button", "form_submit_button"):
    _params = __import__("inspect").signature(getattr(_st, _name)).parameters
    check(f"st.{_name} still accepts use_container_width",
          "use_container_width" in _params,
          "the views pass it; they would all raise TypeError")

check("no view calls st.dataframe directly",
      not any("st.dataframe(" in p.read_text(encoding="utf-8")
              for p in ROOT.glob("views/*.py")),
      "use ui.table, which adapts to the installed Streamlit")
# Every switch_page target must be a file that exists. A rename left one
# pointing at the old pages/ layout, and nothing noticed until the button was
# pressed — a string literal is invisible to an attribute check.
_targets = re.findall(r'switch_page\(\s*["\']([^"\']+)["\']',
                      "\n".join(p.read_text(encoding="utf-8")
                                for p in list(ROOT.glob("views/*.py"))
                                + [ROOT / "app.py"]))
check("there is at least one switch_page to check", bool(_targets))
for _target in _targets:
    check(f"switch_page target {_target} exists", (ROOT / _target).is_file(),
          "the file was renamed and this string was not")

# The declared navigation must point at real files too.
_pages = re.findall(r'st\.Page\(\s*["\']([^"\']+)["\']', entry)
check("every declared page exists",
      all((ROOT / page).is_file() for page in _pages),
      ", ".join(p for p in _pages if not (ROOT / p).is_file()))
check("all ten pages are declared", len(_pages) == 10, f"{len(_pages)}")

check("no view calls st.image directly",
      not any("st.image(" in p.read_text(encoding="utf-8")
              for p in ROOT.glob("views/*.py")),
      "use ui.figure, which adapts to the installed Streamlit")


# ---------------------------------------------------------------------------
# Regenerating a lesson must replace the one on screen
# ---------------------------------------------------------------------------
# "Write a fresh lesson" appeared to do nothing: a lesson that failed
# verification was returned but never saved, the page re-read the database,
# and the reader was handed the previous lesson back — with a green "verified"
# badge on it, which is the most misleading outcome available.

_day1 = curriculum.TOPICS[0]
db.save_content("lesson-shadow-test", _day1["slug"], "lesson", "beginner",
                {"title": "AN OLDER LESSON", "cells": [], "verified": True},
                replace=True)
check("the newest lesson wins over an older row",
      db.latest_lesson(_day1["slug"])["body"]["title"] == "AN OLDER LESSON",
      "with only one row present it should be that one")

db.save_content("lesson-shadow-test-2", _day1["slug"], "lesson", "beginner",
                {"title": "A NEWER LESSON", "cells": []}, replace=True)
db.retire_other_lessons(_day1["slug"], "lesson-shadow-test-2")
check("retiring the others leaves exactly one lesson",
      len(db.list_content(_day1["slug"], "lesson", limit=10)) == 1)
check("and it is the one that was kept",
      db.latest_lesson(_day1["slug"])["body"]["title"] == "A NEWER LESSON")
db.execute("DELETE FROM content WHERE id LIKE 'lesson-shadow-test%'")

check("teach() saves the attempt it returns, verified or not",
      "_save_lesson(topic, body, response_for_save)"
      in (ROOT / "core" / "generators.py").read_text(encoding="utf-8"),
      "an unsaved return is invisible to the page after a rerun")
check("the Learn page prefers what was just generated",
      "fresh_key" in (ROOT / "views" / "learn.py").read_text(encoding="utf-8"))
check("no page reads a lesson through list_content",
      not any("list_content(topic[\"slug\"], \"lesson\"" in
              p.read_text(encoding="utf-8") for p in ROOT.glob("views/*.py")),
      "use db.latest_lesson, which orders by when it was written")


# ---------------------------------------------------------------------------
# The sandbox actually runs a lesson-shaped notebook
# ---------------------------------------------------------------------------
result = sandbox.run_notebook([
    "from core.datasets import load\ndf = load('customer_churn')\nprint(df.shape)",
    "import matplotlib.pyplot as plt\nfig, ax = plt.subplots()\n"
    "ax.hist(df['tenure_months'], bins=20)\nax.set_title('tenure')",
], timeout=90)
check("a two-cell notebook runs", result.ok, result.first_error[:200])
check("stdout is captured", "4200" in result.all_stdout())
check("a figure is captured", result.figure_count() == 1,
      f"{result.figure_count()} figures")

blocked = sandbox.run_notebook(["import requests"])
check("the network is blocked", blocked.verdict == "blocked")

honest = {
    "title": "x", "theory": "y" * 500,
    "cells": [
        {"code": "from core.datasets import load\n"
                 "df = load('customer_churn')\nprint(df.shape)",
         "explain": "load it", "says": "4200 rows and 10 columns"},
        {"code": "print(round(df['churned'].mean(), 3))",
         "explain": "how many left", "says": "about 0.46 of them churned"},
    ],
}
report = verify.verify(honest, curriculum.TOPICS[1])
check("the verifier passes an honest lesson", report.ok, report.summary)

# And the four gates each reject what they exist to reject.
check("the shape gate rejects a one-cell lesson",
      verify.verify({**honest, "cells": honest["cells"][:1]},
                    curriculum.TOPICS[1]).gate == "shape")
check("the horizon gate rejects reaching ahead",
      verify.verify({**honest, "cells": honest["cells"] + [
          {"code": "from sklearn.ensemble import RandomForestClassifier",
           "explain": "a forest", "says": "ok"}]},
                    curriculum.TOPICS[1]).gate == "horizon")
check("the execution gate rejects broken code",
      verify.verify({**honest, "cells": honest["cells"] + [
          {"code": "print(df['not_a_column'].mean())",
           "explain": "x", "says": "y"}]},
                    curriculum.TOPICS[1]).gate == "execution")
check("the claims gate rejects an invented number",
      verify.verify({**honest, "cells": [honest["cells"][0],
                     {**honest["cells"][1], "says": "about 0.92 of them churned"}]},
                    curriculum.TOPICS[1]).gate == "claims")


# ---------------------------------------------------------------------------
print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
for line in FAIL:
    print("  FAIL:", line)
sys.exit(1 if FAIL else 0)
