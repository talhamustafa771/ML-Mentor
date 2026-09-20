"""Checks a generated lesson before anyone reads it.

The Python mentor shipped three broken lessons before this idea existed: one
that used a function six weeks early, one whose "trap" example did not actually
trap anything, and one that confidently stated an output the code never
produced. All three were found by a person reading carefully. That is not a
system, and a person reading carefully does not scale to ninety-one days.

A machine learning lesson can fail in more ways than a Python one, because its
code touches real data. The column might not exist. The model might not
converge. The number quoted in the prose might be the number the author
expected rather than the number the machine produced — and that last one is the
dangerous failure, because it is invisible: the lesson looks right, reads well,
and teaches something false.

So a lesson passes four gates, in this order, cheapest first:

  1. Shape.     Does it have the parts a lesson needs, with nothing empty?
  2. Horizon.   Does any cell use something not yet taught? (core/concepts.py)
  3. Execution. Does every cell actually run against the bundled dataset?
  4. Claims.    Does each number the prose asserts appear in what the code
                printed?

Gates 1 and 2 are free. Gate 3 costs a couple of seconds. Gate 4 costs nothing
extra, because it reads the output gate 3 already produced.

A lesson that fails any gate is not shown. It goes back to the generator with
the specific failure attached — the traceback, the offending symbol, the
claimed number against the printed one — because "try again" produces the same
lesson and "here is exactly what was wrong" produces a different one.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from . import concepts, datasets, sandbox
from .config import LEARNING


@dataclass
class Report:
    ok: bool = False
    gate: str = ""                      # which gate stopped it
    problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    run: sandbox.NotebookResult | None = None
    seconds: float = 0.0

    @property
    def summary(self) -> str:
        if self.ok:
            note = f"{len(self.run.cells)} cells ran" if self.run else "checked"
            figures = self.run.figure_count() if self.run else 0
            extra = f", {figures} figure{'s' if figures != 1 else ''}" if figures else ""
            return f"Verified: {note}{extra}."
        return f"Failed at {self.gate}: " + "; ".join(self.problems[:3])

    def feedback(self) -> str:
        """What to hand back to the generator. Specific, or it is useless."""
        if self.ok:
            return ""
        lines = [f"The previous attempt was rejected at the {self.gate} check. "
                 "Fix exactly these and return the whole lesson again:"]
        lines += [f"  - {p}" for p in self.problems]
        if self.run is not None:
            index = self.run.failed_index
            if index >= 0 and self.run.cells[index].traceback:
                lines += ["", f"The traceback from cell {index + 1}:",
                          self.run.cells[index].traceback.strip()[-1200:]]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Gate 1: shape
# ---------------------------------------------------------------------------

REQUIRED = ("title", "theory", "cells")
MIN_THEORY_CHARS = 400


def check_shape(lesson: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    for key in REQUIRED:
        if not lesson.get(key):
            problems.append(f"'{key}' is missing or empty")

    theory = lesson.get("theory")
    if isinstance(theory, str) and 0 < len(theory) < MIN_THEORY_CHARS:
        problems.append(
            f"the theory is {len(theory)} characters, which is not a lesson. "
            f"At least {MIN_THEORY_CHARS} are needed.")

    cells = lesson.get("cells") or []
    if not isinstance(cells, list):
        problems.append("'cells' must be a list")
        return problems
    if 0 < len(cells) < 2:
        problems.append("a lesson needs at least two code cells: one that "
                        "looks at the data, and one that does something to it")

    for index, cell in enumerate(cells, start=1):
        if isinstance(cell, str):
            if not cell.strip():
                problems.append(f"cell {index} is empty")
            continue
        if not isinstance(cell, dict):
            problems.append(f"cell {index} is neither a string nor an object")
            continue
        if not str(cell.get("code", "")).strip():
            problems.append(f"cell {index} has no code")
        if not str(cell.get("explain", "")).strip():
            # The user asked for every step explained. A cell without one is
            # half a lesson, so this is a rejection rather than a warning.
            problems.append(f"cell {index} has no explanation of what it does")

    return problems


def cell_sources(lesson: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for cell in lesson.get("cells") or []:
        if isinstance(cell, str):
            out.append(cell)
        elif isinstance(cell, dict):
            out.append(str(cell.get("code", "")))
    return out


# ---------------------------------------------------------------------------
# Gate 2: horizon
# ---------------------------------------------------------------------------

# How many times a future idea may be named before it counts as being taught.
# One mention in the main theory is a signpost — "in three weeks you will
# cross-validate this" — and banning that would make the lessons worse. Three
# is teaching. Commentary attached to a code cell gets no allowance at all: if
# the cell in front of the reader is doing the thing, the thing is being
# taught, whatever the import list says.
THEORY_MENTIONS_ALLOWED = 3


def check_horizon(lesson: dict[str, Any], day: int) -> list[str]:
    problems: list[str] = []
    for index, source in enumerate(cell_sources(lesson), start=1):
        for violation in concepts.check_code(source, day)[:4]:
            problems.append(f"cell {index}, {violation}")

    # The symbol scan above sees an import. It does not see a lesson that
    # shuffles the rows itself, slices off a quarter, scores the result and
    # calls it "test accuracy" — no forbidden symbol appears anywhere and four
    # ideas from later weeks have still been taught. That happened on day 1,
    # so the commentary beside each cell is read too.
    for index, cell in enumerate(lesson.get("cells") or [], start=1):
        if not isinstance(cell, dict):
            continue
        commentary = " ".join(str(cell.get(key, ""))
                              for key in ("explain", "says"))
        for key, arrives, _ in concepts.prose_reaches_ahead(commentary, day,
                                                            min_hits=1)[:3]:
            problems.append(
                f"cell {index} works with {concepts.label(key)}, which is "
                f"day {arrives}. Writing a later technique out by hand teaches "
                "it just as much as importing it would.")
    # The main theory is checked too, and this is a rejection rather than a
    # note. It was a note at first, to save a generation on a tight token
    # budget — and three lessons in a row then opened day 1 with a day-22
    # linear equation, because a warning changes what the reader is told and
    # nothing about what the model writes. One retry is cheaper than a fourth
    # wrong lesson. A single forward reference is still allowed; the threshold
    # is what separates a signpost from a syllabus.
    for key, arrives, hits in concepts.prose_reaches_ahead(
            str(lesson.get("theory", "")), day,
            min_hits=THEORY_MENTIONS_ALLOWED)[:3]:
        problems.append(
            f"the theory teaches {concepts.label(key)} ({hits} mentions), "
            f"which belongs to day {arrives}. Remove that passage entirely "
            "rather than shortening it.")

    # Fenced code inside the prose counts too; it is code the reader will read.
    for field_name in ("theory", "walkthrough", "summary"):
        text = lesson.get(field_name)
        if isinstance(text, str) and "```" in text:
            reach = concepts.reaches_ahead({field_name: text}, day)
            if reach:
                problems.append(f"the {field_name} contains code that reaches "
                                f"ahead: {reach}")
    return problems


# ---------------------------------------------------------------------------
# Gate 3: execution
# ---------------------------------------------------------------------------

def check_dataset(lesson: dict[str, Any], topic: dict[str, Any]) -> list[str]:
    """The dataset must exist, and the lesson must use the one it was given."""
    problems: list[str] = []
    wanted = topic.get("dataset", "")
    sources = "\n".join(cell_sources(lesson))

    used = set(re.findall(r"load\(\s*['\"]([a-z_]+)['\"]\s*\)", sources))
    for name in used:
        if name not in datasets.CATALOGUE:
            problems.append(f"cell code loads '{name}', which is not a bundled "
                            f"dataset. Available: {', '.join(sorted(datasets.CATALOGUE))}")
        elif not datasets.CATALOGUE[name].path.exists():
            problems.append(f"'{name}' is in the catalogue but its CSV is "
                            "missing from the datasets folder")

    if wanted and used and wanted not in used:
        problems.append(f"this topic is built on the '{wanted}' dataset, but "
                        f"the code loads {', '.join(sorted(used))}")
    return problems


def run(lesson: dict[str, Any], *, timeout: int | None = None
        ) -> sandbox.NotebookResult:
    return sandbox.run_notebook(cell_sources(lesson),
                                timeout=timeout or LEARNING.sandbox_timeout_seconds)


# ---------------------------------------------------------------------------
# Gate 4: claims
# ---------------------------------------------------------------------------
# The quiet failure. A lesson says "the AUC comes out around 0.71" because the
# author expected 0.71; the code printed 0.68; nobody notices, and the reader
# learns a number that is wrong.
#
# Every number in the prose is checked against what the code actually printed.
# The rule has to be loose in one direction and strict in the other: loose
# about rounding and formatting, because "about 0.71" and "0.7134" are the same
# claim, and strict about the number itself, because that is the whole point.

_NUMBER = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)(?![\w.])")

# Numbers that are never claims about output: figure sizes, random seeds, years,
# and the ordinary small integers of English prose ("the three steps below").
_NEVER_A_CLAIM = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
                  "12", "20", "42", "50", "100", "1000"}


def _printed_numbers(text: str) -> list[float]:
    out: list[float] = []
    for match in _NUMBER.finditer(text or ""):
        try:
            out.append(float(match.group(1)))
        except ValueError:
            continue
    return out


def _claimed_numbers(text: str) -> list[str]:
    """Numbers in prose that look like results rather than furniture.

    A claim is a decimal, or an integer large enough to be a count of
    something. Everything in the never-a-claim set is skipped: rejecting a
    lesson for saying "the three assumptions" would make this gate useless
    within a day.
    """
    out: list[str] = []
    for match in _NUMBER.finditer(text or ""):
        raw = match.group(1)
        if raw in _NEVER_A_CLAIM:
            continue
        if "." not in raw and len(raw) <= 2:
            continue
        out.append(raw)
    return out


def _matches(claim: str, printed: list[float]) -> bool:
    """Does a claimed number appear in the output, allowing for rounding?

    The tolerance is one unit in the last place the lesson bothered to write.
    A lesson that says 0.71 accepts anything that rounds to 0.71; a lesson that
    says 0.7134 has to be right to four places, which is its own fault for
    quoting four.
    """
    try:
        value = float(claim)
    except ValueError:
        return True
    decimals = len(claim.split(".")[1]) if "." in claim else 0
    tolerance = max(0.5 * 10 ** -decimals, abs(value) * 0.02)
    return any(abs(value - seen) <= tolerance for seen in printed)


def check_claims(lesson: dict[str, Any],
                 result: sandbox.NotebookResult) -> tuple[list[str], list[str]]:
    """Returns (problems, warnings).

    Only numbers stated in the same cell's explanation are treated as
    checkable claims. A number in the opening theory is usually context — "most
    production models are retrained monthly" — and holding the generator to it
    would reject good lessons for true statements about the world.
    """
    problems: list[str] = []
    warnings: list[str] = []

    cells = lesson.get("cells") or []
    for index, cell in enumerate(cells):
        if not isinstance(cell, dict):
            continue
        if index >= len(result.cells):
            break
        printed = _printed_numbers(result.cells[index].stdout)
        commentary = " ".join(str(cell.get(key, "")) for key in
                              ("says", "output_means", "result"))
        if not commentary.strip():
            continue
        for claim in _claimed_numbers(commentary)[:8]:
            if not printed:
                warnings.append(
                    f"cell {index + 1} comments on the number {claim}, but the "
                    "cell printed nothing to compare it against")
            elif not _matches(claim, printed):
                problems.append(
                    f"cell {index + 1} says {claim}, but the code printed "
                    f"{', '.join(str(p) for p in printed[:6])}")
    return problems, warnings


# ---------------------------------------------------------------------------
# The whole gate
# ---------------------------------------------------------------------------

def verify(lesson: dict[str, Any], topic: dict[str, Any], *,
           execute: bool = True, timeout: int | None = None) -> Report:
    """Run every gate. Stops at the first one that fails.

    Stopping early is deliberate: a lesson that reaches past its day will be
    rewritten anyway, and executing it first would spend three seconds proving
    something already known.
    """
    import time                                   # noqa: PLC0415 - trivial

    started = time.perf_counter()
    day = int(topic.get("day", 1))

    problems = check_shape(lesson)
    if problems:
        return Report(ok=False, gate="shape", problems=problems,
                      seconds=time.perf_counter() - started)

    problems = check_horizon(lesson, day)
    if problems:
        return Report(ok=False, gate="horizon", problems=problems,
                      seconds=time.perf_counter() - started)

    problems = check_dataset(lesson, topic)
    if problems:
        return Report(ok=False, gate="dataset", problems=problems,
                      seconds=time.perf_counter() - started)

    if not execute or not LEARNING.verify_lesson_code:
        return Report(ok=True, gate="", seconds=time.perf_counter() - started,
                      warnings=["the code was not executed"])

    result = run(lesson, timeout=timeout)
    if not result.ok:
        index = result.failed_index
        detail = (f"cell {index + 1} failed: {result.cells[index].error}"
                  if index >= 0 else result.message)
        return Report(ok=False, gate="execution", problems=[detail],
                      run=result, seconds=time.perf_counter() - started)

    problems, warnings = check_claims(lesson, result)
    if problems:
        return Report(ok=False, gate="claims", problems=problems,
                      warnings=warnings, run=result,
                      seconds=time.perf_counter() - started)

    # Passing with nothing to show is technically a pass and practically a
    # failure: the reader came for a picture and a number.
    if result.figure_count() == 0 and not result.all_stdout().strip():
        warnings.append("the code ran but printed nothing and drew nothing")

    return Report(ok=True, run=result, warnings=warnings,
                  seconds=time.perf_counter() - started)


def verify_challenge(challenge: dict[str, Any], answer: str,
                     timeout: int | None = None) -> sandbox.NotebookResult:
    """Grade an attempt at a bank challenge.

    The learner's code runs first, then a hidden check cell runs in the same
    namespace and asserts against whatever the learner left behind. That is the
    only honest way to grade this kind of task: there is no single correct
    program, only a correct thing to have found out.
    """
    setup = str(challenge.get("setup", "") or "")
    check = str(challenge.get("check", "") or "")
    cells = [c for c in (setup, answer, check) if c.strip()]
    return sandbox.run_notebook(cells, timeout=timeout)
