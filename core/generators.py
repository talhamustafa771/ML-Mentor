"""Writes the material: lessons, questions, flashcards and challenges.

Four things matter here beyond prompting.

**The horizon comes first.** Every prompt carries core/concepts.horizon(topic),
and every generated item is checked against it afterwards. The prompt alone is
not enough — the Python mentor asked nicely three times and got a function
definition on day 1 anyway. The check is what makes it true.

**Lessons are executed, not trusted.** A lesson is not stored until its code
has run against the real dataset and the numbers in its commentary match what
the code actually printed. See core/verify.py. A rejected lesson goes back with
the traceback attached; a lesson that fails twice is not shown at all, because
a wrong lesson is worse than a missing one.

**Everything is cached.** Items are keyed by a content hash, so the same
question is never paid for twice and nothing changes under you when Streamlit
reruns the page. On a free API tier this is the difference between a usable app
and one that stalls halfway through a session.

**Validation is strict and quiet.** A malformed multiple-choice question with a
correct_index of 7 out of 4 options would mark you wrong forever. Items that
fail validation are discarded rather than repaired, which is why a generator
sometimes returns fewer items than asked for.
"""
from __future__ import annotations

import hashlib
import json
import random
from typing import Any, Callable

from . import concepts, datasets, db, llm, mathpad, verify
from .config import LEARNING

GENERATOR_SYSTEM = (
    "You are a machine learning practitioner of fifteen years and an unusually "
    "good teacher. You have shipped models that made money and models that "
    "failed, and you are candid about which lessons came from which.\n\n"
    "You are writing for one person: an adult who can program in Python and "
    "whose mathematics is weak and who has said so. They are not a beginner at "
    "computing and they are a beginner at this. You never condescend and you "
    "never hand-wave.\n\n"
    "How you write:\n"
    "- Concrete before abstract. Show the thing happening on real data, then "
    "name it.\n"
    "- Every symbol you write, you name in words the first time.\n"
    "- You say what a technique costs and when it is the wrong choice, not "
    "only what it does.\n"
    "- British spelling. Plain sentences. No exclamation marks, no emoji, no "
    "'let's dive in', no 'in today's fast-paced world'.\n"
    "- You never claim a number you have not computed. If you want to comment "
    "on a result, print it in the code first and describe what was printed."
)


def _content_id(topic_slug: str, kind: str, payload: str) -> str:
    digest = hashlib.sha256(f"{topic_slug}|{kind}|{payload}".encode()).hexdigest()
    return f"{kind}-{topic_slug}-{digest[:12]}"


def _dataset_block(topic: dict[str, Any]) -> str:
    name = topic.get("dataset") or ""
    if name and name in datasets.CATALOGUE:
        columns = datasets.columns(name)
        block = datasets.describe(name)
        if columns:
            block += "\nColumns, exactly as spelled: " + ", ".join(columns)
        return block
    available = ", ".join(datasets.available())
    return ("No dataset is fixed for this day. If you use one, pick from: "
            f"{available}, and load it with  from core.datasets import load")


def _maths_block(topic: dict[str, Any]) -> str:
    # Only the day's mathematics, not its concepts. offer_for deliberately
    # includes both, because a reader may want either explained; a prompt
    # heading called "MATHEMATICS THIS DAY TOUCHES" must not list ideas that
    # are not mathematics at all.
    terms = [(key, mathpad.lookup_exact(key).title
              if mathpad.lookup_exact(key) else concepts.label(key),
              mathpad.lookup_exact(key) is not None)
             for key in topic.get("maths", ())]
    if not terms:
        # Silence here used to mean "no guidance", and the shape block's
        # instruction on how to format a formula was read as an invitation to
        # produce one. Day 1 has no mathematics at all and got a linear
        # equation with weights and an intercept three times running.
        return ("MATHEMATICS THIS DAY TOUCHES: none. Write no formula at all. "
                "No LaTeX, no symbols with subscripts, no equation. If you "
                "reach for one, the lesson has drifted off today's topic.")
    known = [label for _, label, banked in terms if banked]
    lines = ["MATHEMATICS THIS DAY TOUCHES: "
             + ", ".join(label for _, label, _ in terms) + ".",
             "The reader has said their maths is weak. Introduce each of these "
             "in words before any symbol appears, and never use one as though "
             "it were obvious."]
    if known:
        lines.append("These have a written explanation available in the app's "
                     "Math Helper, so you may refer to them by name: "
                     + ", ".join(known) + ".")
    return "\n".join(lines)


def _continuity(topic: dict[str, Any]) -> str:
    """What happened yesterday, so a lesson reads as day N and not as lesson 1."""
    from . import curriculum                        # noqa: PLC0415 - cheap

    day = int(topic.get("day", 1))
    previous = [t for t in curriculum.TOPICS if t["day"] < day][-2:]
    if not previous:
        return ("This is the first day of the course. The reader has done "
                "nothing yet. Do not refer to anything as already covered.")
    parts = [f"day {t['day']}: {t['title']} — {t['summary']}" for t in previous]
    return ("Immediately before this, the reader did:\n  "
            + "\n  ".join(parts)
            + "\nOpen by connecting to that, in one sentence, not by starting "
              "over.")


def _topic_context(topic: dict[str, Any]) -> str:
    """Everything the model needs, in the order it needs it.

    The horizon goes last on purpose. It is the constraint most often ignored,
    and a constraint at the end of a prompt is obeyed more often than one
    buried in the middle.
    """
    blocks = [
        f"TOPIC: {topic['title']}",
        f"Track: {topic['track']} | week {topic['week']} | day "
        f"{topic.get('day', '?')} of 91 | {topic['difficulty']} | "
        f"{topic['minutes']} minutes of study",
        f"Summary: {topic['summary']}",
        "Objectives, all of which the reader must be able to do afterwards:\n"
        + "\n".join(f"  - {o}" for o in topic["objectives"]),
        "Why it matters in practice:\n"
        + "\n".join(f"  - {r}" for r in topic["real_world"]),
        _continuity(topic),
        _dataset_block(topic),
        _maths_block(topic),
        concepts.horizon(topic),
    ]
    return "\n\n".join(b for b in blocks if b)


def _existing_prompts(topic_slug: str, kind: str, field: str,
                      limit: int = 25) -> str:
    items = db.list_content(topic_slug, kind, limit=limit)
    texts = [str(i["body"].get(field, ""))[:110]
             for i in items if i["body"].get(field)]
    if not texts:
        return ""
    return ("\nDo NOT repeat or lightly reword any of these, which already "
            "exist:\n" + "\n".join(f"  - {t}" for t in texts[:limit]))


def _store(topic_slug: str, kind: str, difficulty: str, body: dict[str, Any],
           response: llm.LLMResult) -> str:
    cid = _content_id(topic_slug, kind, json.dumps(body, sort_keys=True))
    body = dict(body)
    body["id"] = cid
    db.save_content(cid, topic_slug, kind, difficulty, body,
                    response.provider, response.model)
    return cid


def _unwrap(payload: Any) -> list[Any]:
    """Models wrap arrays in a key. Accept any single list-valued key."""
    if isinstance(payload, dict):
        for value in payload.values():
            if isinstance(value, list):
                return value
        return [payload]
    return payload if isinstance(payload, list) else []


# ---------------------------------------------------------------------------
# Validators — each returns (ok, reason)
# ---------------------------------------------------------------------------

def _valid_mcq(item: Any) -> tuple[bool, str]:
    if not isinstance(item, dict):
        return False, "not an object"
    question, options = item.get("question"), item.get("options")
    index = item.get("correct_index")
    if not isinstance(question, str) or len(question.strip()) < 15:
        return False, "question too short"
    if not isinstance(options, list) or len(options) != 4:
        return False, "needs exactly 4 options"
    if any(not isinstance(o, str) or not o.strip() for o in options):
        return False, "an option is empty"
    if len({o.strip().lower() for o in options}) != 4:
        return False, "two options are the same"
    if not isinstance(index, int) or not 0 <= index < 4:
        return False, f"correct_index {index!r} is out of range"
    if not isinstance(item.get("explanation"), str) or \
            len(item["explanation"].strip()) < 40:
        return False, "explanation too short to teach anything"
    return True, ""


def _valid_card(item: Any) -> tuple[bool, str]:
    if not isinstance(item, dict):
        return False, "not an object"
    front, back = item.get("front"), item.get("back")
    if not isinstance(front, str) or len(front.strip()) < 8:
        return False, "front too short"
    if not isinstance(back, str) or len(back.strip()) < 15:
        return False, "back too short"
    if len(front) > 220:
        return False, "front is a paragraph, not a prompt"
    return True, ""


def _valid_challenge(item: Any) -> tuple[bool, str]:
    if not isinstance(item, dict):
        return False, "not an object"
    for key in ("task", "check"):
        if not isinstance(item.get(key), str) or not item[key].strip():
            return False, f"'{key}' is missing"
    if len(item["task"].strip()) < 50:
        return False, "the task does not say enough to be attemptable"
    if "assert" not in item["check"]:
        return False, "the check must assert something"
    dataset = item.get("dataset", "")
    if dataset and dataset not in datasets.CATALOGUE:
        return False, f"unknown dataset {dataset!r}"
    return True, ""


def _valid_misconception(item: Any) -> tuple[bool, str]:
    if not isinstance(item, dict):
        return False, "not an object"
    for key in ("belief", "why_wrong", "what_is_true"):
        if not isinstance(item.get(key), str) or len(item[key].strip()) < 25:
            return False, f"'{key}' is missing or too short"
    return True, ""


# ---------------------------------------------------------------------------
# The shared generation path
# ---------------------------------------------------------------------------

def _generate(topic: dict[str, Any], kind: str, count: int, prompt: str,
              validator: Callable[[Any], tuple[bool, str]], *,
              settings: dict[str, str] | None = None, tier: str = "fast",
              max_tokens: int = 4000, difficulty: str | None = None,
              ) -> tuple[list[dict[str, Any]], list[str]]:
    """Returns (stored items, rejection reasons)."""
    response = llm.complete_json(GENERATOR_SYSTEM, prompt, settings=settings,
                                 tier=tier, max_tokens=max_tokens)
    if not response.ok:
        return [], [f"generation failed: {response.error}"]

    items = _unwrap(response.data)
    if not items:
        return [], ["the model did not return a list of items"]

    day = int(topic.get("day", 1))
    stored: list[dict[str, Any]] = []
    rejected: list[str] = []

    for item in items[:count]:
        ok, reason = validator(item)
        if not ok:
            rejected.append(reason)
            continue
        ahead = concepts.reaches_ahead(item, day)
        if ahead:
            rejected.append(f"discarded — it uses {ahead}")
            continue
        cid = _store(topic["slug"], kind, difficulty or topic["difficulty"],
                     item, response)
        saved = db.get_content(cid)
        if saved:
            stored.append(saved["body"])
    return stored, rejected


# ---------------------------------------------------------------------------
# The lesson
# ---------------------------------------------------------------------------

_LESSON_SHAPE = """Return ONE JSON object with exactly these keys:

{
  "title": "the lesson's own title, not the topic name repeated",

  "hook": "One paragraph. A concrete situation where getting this wrong costs
           something real. No throat-clearing.",

  "theory": "The teaching itself, as markdown, 450 to 900 words. Build the idea
             from something the reader already has. Put any formula on its own
             line in LaTeX between $$ and $$, and immediately below it name
             every symbol in words. Use ## subheadings. No code fences here —
             the code lives in 'cells'.",

  "intuition": "Two or three sentences: the mental picture. What is moving,
                what is traded against what. What should survive six months.",

  "cells": [
    {"code":    "runnable Python, 3 to 15 lines, no fences",
     "explain": "what this cell does and WHY it is the next step; one short
                 paragraph, naming any function used for the first time",
     "says":    "what to notice in the output. Any number here MUST be one
                 this cell prints."}
  ],

  "traps": [
    {"trap": "the mistake, as the thing someone actually does",
     "why":  "why it is tempting and what goes wrong",
     "fix":  "what to do instead"}
  ],

  "summary": "Three to five sentences: what was learned, what it unlocks.",

  "check_yourself": ["two or three questions answerable now, without looking"]
}"""

_LESSON_RULES = """Rules for the cells, which is where this most often goes
wrong:

1. They run IN ORDER IN ONE PROCESS and share a namespace. Cell 3 can use a
   variable cell 1 made. Do not re-import or re-load in every cell.
2. Load data only with:   from core.datasets import load
   Never a path, never a download, never make_classification.
3. Between 3 and 5 cells. Every one prints something or draws something.
4. At least one cell draws a matplotlib figure with a title and labelled axes.
   Do not call plt.show() — figures are captured automatically.
5. Use only the column names listed above, spelled exactly as listed.
6. Every number you mention in "says" must be printed by that same cell. To
   say the AUC is about 0.71, print the AUC. Do not guess it. A lesson whose
   stated numbers disagree with its printed output is rejected and rewritten.
7. Round what you print: print(round(value, 3)).
8. Write the code a practitioner would actually write, including the small
   habits — setting random_state once it is available, printing shapes after a
   split, checking a result rather than assuming it."""


def _lesson_fallback(topic: dict[str, Any], reason: str) -> dict[str, Any]:
    """What is shown when generation fails.

    Deliberately still useful. The curriculum's objectives and real-world notes
    are written by hand and are not subject to model error, so a failed lesson
    degrades to a real study plan rather than an error message.
    """
    return {
        "title": topic["title"],
        "hook": "",
        "theory": (f"**The lesson could not be generated.** {reason}\n\n"
                   f"{topic['summary']}\n\n"
                   "What you are meant to be able to do after today:\n"
                   + "\n".join(f"- {o}" for o in topic["objectives"])
                   + "\n\nWhy it matters:\n"
                   + "\n".join(f"- {r}" for r in topic["real_world"])),
        "intuition": "",
        "cells": [],
        "traps": [],
        "summary": "",
        "check_yourself": topic["objectives"][:3],
        "generated": False,
        "failure": reason,
    }


def teach(topic: dict[str, Any], *, settings: dict[str, str] | None = None,
          force: bool = False, execute: bool = True,
          on_progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    """The lesson for a topic: generated, verified, stored, returned.

    Cached, so a topic costs at most a few requests for the life of the app.
    `force` is what "Write me a fresh lesson" uses; it leaves the stored lesson
    alone unless a new one actually survives verification, so a failed rewrite
    can never leave a topic with nothing.
    """
    def say(message: str) -> None:
        if on_progress:
            on_progress(message)

    if not force:
        existing = db.latest_lesson(topic["slug"])
        if existing:
            return existing["body"]

    base = (f"{_topic_context(topic)}\n\n"
            f"Write today's lesson.\n\n{_LESSON_SHAPE}\n\n{_LESSON_RULES}")

    attempts = max(1, LEARNING.max_regeneration_attempts)
    prompt = base
    last_report: verify.Report | None = None
    best: dict[str, Any] | None = None
    response_for_save: llm.LLMResult | None = None

    for attempt in range(1, attempts + 1):
        say(f"Writing the lesson (attempt {attempt} of {attempts})...")
        response = llm.complete_json(GENERATOR_SYSTEM, prompt, settings=settings,
                                     tier="strong",
                                     max_tokens=LEARNING.lesson_max_tokens,
                                     on_wait=say)
        if not response.ok:
            return _lesson_fallback(topic, response.error or "no response")

        lesson = response.data
        if isinstance(lesson, list) and lesson:
            lesson = lesson[0]
        if not isinstance(lesson, dict):
            prompt = (base + "\n\nYour previous reply was not a single JSON "
                             "object. Return one JSON object and nothing else.")
            continue

        say("Running every code cell against the real data...")
        report = verify.verify(lesson, topic, execute=execute)
        last_report = report

        if report.ok:
            body = _finalise(lesson, topic, report)
            _save_lesson(topic, body, response)
            say(report.summary)
            return body

        best = lesson
        response_for_save = response
        say(f"Rejected: {report.summary}")
        prompt = (f"{base}\n\n--- YOUR PREVIOUS ATTEMPT ---\n"
                  f"{json.dumps(lesson)[:5000]}\n\n{report.feedback()}")
        if report.gate == "horizon":
            prompt += (
                "\n\nTo be clear about what this means: do not split the data, "
                "do not score predictions, do not count right and wrong "
                "answers into a table, and do not choose a cut-off for a "
                "probability — not with a library and not by hand. Build the "
                "lesson only from loading the data, looking at columns, "
                "grouping, and drawing. That is enough for today.\n\n"
                "If the objection names a formula or an equation, delete that "
                "whole section. Do not shorten it, do not move it into an "
                "aside, and do not replace it with a sentence describing the "
                "formula in words — the reader is not ready for it in any "
                "form, and a paragraph about an equation is still the "
                "equation.")

    # Out of attempts. The last lesson is kept and marked unverified rather
    # than thrown away, for two reasons: an unverified lesson with its
    # failures shown is more use than an error page, and on a free-tier token
    # budget a discarded generation is a real cost.
    #
    # It has to be SAVED, not merely returned. The previous version returned it
    # without writing it, the page then re-read the database, and the reader
    # was handed the old lesson back with a green "verified" badge on it —
    # which looked exactly like the button having done nothing.
    if best is not None and last_report is not None:
        body = _finalise(best, topic, last_report)
        if response_for_save is not None:
            _save_lesson(topic, body, response_for_save)
        say(f"Kept, but it did not pass the {last_report.gate} check.")
        return body
    return _lesson_fallback(topic, "the lesson did not verify after "
                                   f"{attempts} attempts")


def _finalise(lesson: dict[str, Any], topic: dict[str, Any],
              report: verify.Report) -> dict[str, Any]:
    """Attach what verification found, so the page can be honest about it."""
    body = dict(lesson)
    body["verified"] = report.ok
    body["verify_gate"] = report.gate
    body["verify_problems"] = report.problems
    body["verify_warnings"] = report.warnings
    body["verify_seconds"] = round(report.seconds, 2)
    body["maths_offered"] = [key for key, _, _ in mathpad.offer_for(topic)]
    body["dataset"] = topic.get("dataset", "")
    body["day"] = topic.get("day")
    if report.run is not None:
        # The outputs are kept so the lesson can be read on a phone, on a
        # train, without waiting three seconds for scikit-learn to import.
        # Pressing Run re-executes and replaces them.
        body["outputs"] = [
            {"stdout": cell.stdout, "figures": cell.figures,
             "seconds": round(cell.seconds, 3)}
            for cell in report.run.cells
        ]
    return body


def _save_lesson(topic: dict[str, Any], body: dict[str, Any],
                 response: llm.LLMResult) -> None:
    cid = _content_id(topic["slug"], "lesson", topic["slug"])
    body["id"] = cid
    db.save_content(cid, topic["slug"], "lesson", topic["difficulty"], body,
                    response.provider, response.model, replace=True)
    db.mark_verified(cid, "passed" if body.get("verified") else "failed")
    # Leave exactly one lesson standing for this topic, so nothing written
    # earlier under a different id can shadow what was just made.
    db.retire_other_lessons(topic["slug"], cid)


# ---------------------------------------------------------------------------
# Questions
# ---------------------------------------------------------------------------

def generate_mcqs(topic: dict[str, Any], count: int = 6, *,
                  settings: dict[str, str] | None = None,
                  difficulty: str | None = None
                  ) -> tuple[list[dict[str, Any]], list[str]]:
    difficulty = difficulty or topic["difficulty"]
    prompt = f"""{_topic_context(topic)}

Write {count} multiple-choice questions at {difficulty} level.

What makes a good question here:
- It tests judgement, not recall. "Which of these is data leakage?" beats
  "what does the C parameter stand for".
- The three wrong options are beliefs a real learner holds, not filler. The
  most tempting wrong option should be one a careless practitioner would pick.
- Prefer questions about what a result MEANS: given this confusion matrix,
  given this residual plot, given that the test score beat the training score.
- Where a snippet or a small table helps, put it in the separate "code" field
  as plain source with real line breaks. The "question" field is prose only:
  no code, no backticks, no fences.
- The explanation must say why the right answer is right AND why the most
  tempting wrong one is wrong.
- Vary which position is correct.
{_existing_prompts(topic['slug'], 'mcq', 'question')}

Return a JSON array of objects:
{{"question": "prose only", "code": "<snippet or null>",
  "options": ["...","...","...","..."], "correct_index": 0,
  "explanation": "...", "skill": "<which objective this tests>"}}"""
    return _generate(topic, "mcq", count, prompt, _valid_mcq,
                     settings=settings, difficulty=difficulty)


def generate_cards(topic: dict[str, Any], count: int = 8, *,
                   settings: dict[str, str] | None = None
                   ) -> tuple[list[dict[str, Any]], list[str]]:
    prompt = f"""{_topic_context(topic)}

Write {count} flashcards for spaced repetition.

- The front is a question or a prompt, one line, under 200 characters.
- The back is the answer in one to three sentences. Complete, not cryptic.
- Cover the ideas that must become automatic: what a metric means, when a
  method fails, which assumption a model makes, what a formula's symbols are.
- At least two cards should be "what would you check if..." rather than
  "what is...".
- No card may depend on having just read another card.
{_existing_prompts(topic['slug'], 'card', 'front')}

Return a JSON array of objects:
{{"front": "...", "back": "...", "kind": "definition|judgement|formula|trap"}}"""
    return _generate(topic, "card", count, prompt, _valid_card,
                     settings=settings)


def generate_misconceptions(topic: dict[str, Any], count: int = 3, *,
                            settings: dict[str, str] | None = None
                            ) -> tuple[list[dict[str, Any]], list[str]]:
    """The things people believe about this topic that are not true.

    Worth its own generator rather than being folded into the lesson. Naming a
    wrong belief directly — as a belief, held by real people, for an
    understandable reason — dislodges it far better than stating the correct
    version and hoping the reader notices the difference.
    """
    prompt = f"""{_topic_context(topic)}

Name {count} things people genuinely believe about this topic that are wrong.

- Each must be a belief somebody actually holds, not a straw man. Prefer the
  ones held by people who are otherwise competent.
- Say why it is believable before saying why it is wrong.
- Then state what is actually true, in one or two sentences.

Return a JSON array:
{{"belief": "...", "why_believable": "...", "why_wrong": "...",
  "what_is_true": "..."}}"""
    return _generate(topic, "misconception", count, prompt, _valid_misconception,
                     settings=settings, tier="strong")


# ---------------------------------------------------------------------------
# Challenges
# ---------------------------------------------------------------------------

_CHALLENGE_RULES = """A challenge is a task against a bundled dataset, graded by
running the learner's code and then running your hidden check in the same
namespace.

- "task" states what to do and what to leave behind, naming the variables the
  check will look for. Be exact: "leave the test AUC in a variable called auc".
- "setup" is optional code that runs BEFORE the learner's code, for loading and
  splitting data so every attempt starts from the same place. Keep it short.
- "check" is the hidden grader. It may only use assert statements and prints.
  It runs after the learner's code and sees every variable they created.
- Each assert must carry a message that teaches: assert 0.6 < auc < 0.8, \\
  "That AUC is too high to be honest on this data — check whether a column \\
  recorded after the outcome is still in your features."
- The check must catch the WRONG way of getting the right answer, not just the
  number. That is the whole point: a leaking model scores 0.99 and is useless.
- "hints" is a list of two or three, each one a nudge rather than the answer.
- "solution" is a reference answer that would pass your own check."""


def generate_challenges(topic: dict[str, Any], count: int = 2, *,
                        settings: dict[str, str] | None = None
                        ) -> tuple[list[dict[str, Any]], list[str]]:
    dataset = topic.get("dataset") or (datasets.available() or [""])[0]
    prompt = f"""{_topic_context(topic)}

Write {count} applied challenges for this day, using the '{dataset}' dataset.

{_CHALLENGE_RULES}
{_existing_prompts(topic['slug'], 'challenge', 'task')}

Return a JSON array of objects:
{{"task": "...", "dataset": "{dataset}", "setup": "<code or empty string>",
  "check": "<assert statements>", "hints": ["...","..."],
  "solution": "<reference code>", "skill": "<concept slug this tests>",
  "difficulty": "beginner|intermediate|advanced"}}"""

    stored, rejected = _generate(topic, "challenge", count, prompt,
                                 _valid_challenge, settings=settings,
                                 tier="strong", max_tokens=3500)

    # A challenge whose own reference solution fails its own check is
    # unsolvable, and the learner would spend an hour proving it. The Python
    # mentor found roughly one generated problem in six was like this, which is
    # why nothing here is trusted without being run.
    kept: list[dict[str, Any]] = []
    for challenge in stored:
        solution = str(challenge.get("solution", "")).strip()
        if not solution:
            db.retire_content(challenge["id"])
            rejected.append("discarded — no reference solution to check it with")
            continue
        result = verify.verify_challenge(challenge, solution)
        if result.ok:
            db.mark_verified(challenge["id"], "passed")
            kept.append(challenge)
        else:
            db.retire_content(challenge["id"])
            rejected.append("discarded — its own reference solution fails its "
                            f"own check: {result.first_error[:120]}")
    return kept, rejected


# ---------------------------------------------------------------------------
# Stock and sessions
# ---------------------------------------------------------------------------

def ensure_stock(topic: dict[str, Any], *,
                 settings: dict[str, str] | None = None,
                 mcqs: int | None = None, cards: int | None = None,
                 ) -> dict[str, int]:
    """Top up a topic's material to the session size. Generates only shortfalls."""
    want_mcqs = mcqs if mcqs is not None else LEARNING.mcqs_per_session
    want_cards = cards if cards is not None else LEARNING.cards_per_session

    made = {"mcq": 0, "card": 0}
    have_mcq = db.content_count(topic["slug"], "mcq")
    if have_mcq < want_mcqs:
        items, _ = generate_mcqs(topic, want_mcqs - have_mcq, settings=settings)
        made["mcq"] = len(items)
    have_card = db.content_count(topic["slug"], "card")
    if have_card < want_cards:
        items, _ = generate_cards(topic, want_cards - have_card,
                                  settings=settings)
        made["card"] = len(items)
    return made


def build_session(topic: dict[str, Any], *,
                  settings: dict[str, str] | None = None,
                  seed: int | None = None) -> dict[str, Any]:
    """Everything a practice session needs, unseen items first."""
    ensure_stock(topic, settings=settings)
    rng = random.Random(seed)

    mcqs = db.list_content(topic["slug"], "mcq", unseen_only=True,
                           limit=LEARNING.mcqs_per_session)
    if len(mcqs) < LEARNING.mcqs_per_session:
        mcqs += [m for m in db.list_content(topic["slug"], "mcq",
                                            limit=LEARNING.mcqs_per_session * 2)
                 if m["id"] not in {x["id"] for x in mcqs}]
    cards = db.list_content(topic["slug"], "card",
                            limit=LEARNING.cards_per_session)

    mcqs = [m["body"] for m in mcqs[: LEARNING.mcqs_per_session]]
    cards = [c["body"] for c in cards[: LEARNING.cards_per_session]]
    rng.shuffle(mcqs)
    rng.shuffle(cards)
    return {"topic": topic, "mcqs": mcqs, "cards": cards}


def explain_mistake(topic: dict[str, Any], question: str, chosen: str,
                    correct: str, *, settings: dict[str, str] | None = None
                    ) -> str:
    """Why the answer they gave is wrong, addressed to the belief behind it."""
    prompt = f"""{_topic_context(topic)}

The reader answered a question wrongly.

Question: {question}
They chose: {chosen}
The correct answer: {correct}

In under 150 words: name the belief that would make their choice look right,
say precisely where that belief breaks, and give one concrete case that shows
it breaking. Do not restate the correct answer as though repetition were an
argument. Address them as "you". Plain prose, no headings."""
    result = llm.complete(GENERATOR_SYSTEM, prompt, settings=settings,
                          tier="fast", max_tokens=400)
    return result.text.strip() if result.ok else ""
