"""XP, levels, badges and skill mastery.

Deliberately modest, and deliberately opinionated about what it pays for.

Points that reward activity make a bar move while you get no better. So every
rule here pays for evidence of skill rather than evidence of attendance: a
challenge solved unaided pays more than one solved after four hints, a project
completed pays more than an hour logged, and nothing pays twice.

The badges are where a machine learning course can say something a generic one
cannot. Half of them reward the judgement this subject is actually about —
noticing a score that is too good to be true, beating a baseline honestly,
choosing a threshold by what a mistake costs. Those are the habits that
separate someone who can fit a model from someone who can be trusted with one,
and none of them is measured by "topics completed".

Every award carries a dedupe key. Streamlit re-runs a page top to bottom on
every click, and an award without one would fire dozens of times a session.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from . import db, scheduler
from .config import LEARNING

# ---------------------------------------------------------------------------
# XP rules
# ---------------------------------------------------------------------------

XP_RULES = {
    "mcq_correct": 4,
    "card_reviewed": 2,
    "lesson_read": 10,
    "cell_run": 3,                # running a lesson's code yourself, not reading it
    "experiment": 12,             # changing a cell and running it again
    "challenge_beginner": 25,
    "challenge_intermediate": 45,
    "challenge_advanced": 80,
    "project": 150,
    "milestone": 300,
    "capstone": 600,
    "topic_mastered": 120,
    "daily_goal": 40,
    "streak_week": 100,
    "maths_asked": 3,             # small, but it pays: asking is the right move
}

# A challenge solved with hints still pays, but less. Index is the hint level.
HINT_MULTIPLIER = [1.0, 0.8, 0.6, 0.45, 0.3]


def award(kind: str, *, reason: str = "", topic_slug: str | None = None,
          dedupe_key: str | None = None, amount: int | None = None) -> int:
    value = amount if amount is not None else XP_RULES.get(kind, 0)
    if value <= 0:
        return 0
    granted = db.award_xp(kind, value, reason=reason, topic_slug=topic_slug,
                          dedupe_key=dedupe_key)
    return value if granted else 0


def award_challenge(challenge: dict[str, Any], hints_used: int = 0,
                    first_solve: bool = True) -> int:
    """Pay for a solved challenge. Returns the XP granted, 0 if already paid."""
    if not first_solve:
        return 0
    difficulty = challenge.get("difficulty", "intermediate")
    base = XP_RULES.get(f"challenge_{difficulty}", 45)
    multiplier = HINT_MULTIPLIER[min(hints_used, len(HINT_MULTIPLIER) - 1)]
    amount = max(5, round(base * multiplier))
    granted = db.award_xp(
        "challenge", amount,
        reason=f"Solved {challenge.get('task', 'a challenge')[:60]}"
               + (f" with {hints_used} hint(s)" if hints_used else " unaided"),
        topic_slug=challenge.get("topic_slug"),
        dedupe_key=f"challenge:{challenge.get('id')}",
    )
    return amount if granted else 0


# ---------------------------------------------------------------------------
# Levels
# ---------------------------------------------------------------------------

LEVEL_NAMES = [
    "Curious", "Data Reader", "First Model", "Honest Evaluator",
    "Pipeline Builder", "Feature Engineer", "Model Selector",
    "Error Analyst", "Practitioner", "Shipping Engineer", "Machine Learner",
]


def level_for(xp: int) -> dict[str, Any]:
    """Levels widen as they go, so early progress feels quick and later levels
    mean something. Thresholds are derived, never stored."""
    level, threshold, step = 0, 0, 250
    while level < len(LEVEL_NAMES) - 1 and xp >= threshold + step:
        threshold += step
        level += 1
        step = round(step * 1.35)
    ceiling = threshold + step
    span = ceiling - threshold
    return {
        "level": level + 1,
        "name": LEVEL_NAMES[level],
        "xp": xp,
        "level_floor": threshold,
        "level_ceiling": ceiling,
        "progress": 0.0 if span == 0 else min(1.0, (xp - threshold) / span),
        "to_next": max(0, ceiling - xp),
        "is_max": level == len(LEVEL_NAMES) - 1,
    }


def status() -> dict[str, Any]:
    return level_for(db.total_xp())


# ---------------------------------------------------------------------------
# Badges
# ---------------------------------------------------------------------------

@dataclass
class Badge:
    slug: str
    name: str
    description: str
    icon: str            # short name; the UI maps it to an inline SVG
    group: str = "progress"


BADGES = [
    # Getting going
    Badge("first-fit", "First Fit", "Fit your first model", "spark", "progress"),
    Badge("week-one", "Week One", "Finish every day of week 1", "check", "progress"),
    Badge("ten-challenges", "Ten Down", "Solve 10 challenges", "stack", "progress"),
    Badge("fifty-challenges", "Half Century", "Solve 50 challenges", "trophy", "progress"),
    Badge("unaided", "No Hints Needed", "Solve 10 challenges without a hint", "target", "progress"),

    # The judgement badges — what this subject is actually about
    Badge("leak-hunter", "Leak Hunter", "Spot the leaking column before the model does",
          "magnifier", "judgement"),
    Badge("baseline-beater", "Beat the Baseline",
          "Beat a dummy baseline on a dataset where that is hard", "arrow", "judgement"),
    Badge("sceptic", "Healthy Scepticism",
          "Reject a result that looked too good to be true", "eye", "judgement"),
    Badge("threshold-setter", "Cost Aware",
          "Choose a threshold by what a mistake costs, not by default", "scales", "judgement"),
    Badge("honest-report", "Honest Numbers",
          "Report a model that did not work, and say why", "book", "judgement"),

    # Craft
    Badge("no-leak-pipeline", "Leak-Free", "Build a pipeline that provably does not leak",
          "shield", "craft"),
    Badge("from-scratch", "From Scratch",
          "Implement gradient descent or backprop yourself", "hammer", "craft"),
    Badge("visualiser", "Show, Don't Tell", "Produce 25 figures of your own", "chart", "craft"),
    Badge("tuner", "Tuner", "Run a hyperparameter search and read it correctly", "dial", "craft"),

    # Persistence and curiosity
    Badge("streak-7", "Seven Days", "Study seven days in a row", "fire", "habit"),
    Badge("streak-30", "Thirty Days", "Study thirty days in a row", "comet", "habit"),
    Badge("asker", "Asks Good Questions",
          "Use the Math Helper 25 times — the fastest way through this course",
          "question", "habit"),
    Badge("perfectionist", "Perfect Run", "Score 100% on a set of questions", "star", "habit"),

    # The end
    Badge("builder", "Builder", "Complete your first project", "blocks", "progress"),
    Badge("architect", "Architect", "Complete a milestone project", "tower", "progress"),
    Badge("graduate", "Graduate", "Complete the capstone", "cap", "progress"),
]

# Badges the app cannot infer from counts alone. They are granted explicitly by
# the page that witnesses the moment — the challenge that hides a leaking
# column, the lesson where the baseline is beaten — and recorded in the XP
# ledger, which is the only place a fact like this can live without being
# recomputed wrongly later.
EARNED_IN_THE_MOMENT = {"leak-hunter", "baseline-beater", "sceptic",
                        "threshold-setter", "honest-report",
                        "no-leak-pipeline", "from-scratch", "tuner"}


def grant(slug: str, reason: str = "") -> bool:
    """Record a badge that was earned by doing something, not by a count."""
    if slug not in {b.slug for b in BADGES}:
        return False
    return bool(db.award_xp("badge", 50, reason=reason or slug,
                            dedupe_key=f"badge:{slug}"))


def _granted() -> set[str]:
    rows = db.query("SELECT dedupe_key FROM xp_events WHERE kind='badge'")
    return {str(r["dedupe_key"]).split(":", 1)[-1] for r in rows
            if r["dedupe_key"]}


def earned_badges() -> list[dict[str, Any]]:
    """Recompute from the record every time.

    A badge table that could disagree with the data underneath it would be
    worse than no badges at all.
    """
    progress = db.bank_progress()
    solved = progress["solved"]
    streak = scheduler.streak_days()
    statuses = db.project_statuses()
    done = [s for s in statuses.values() if s == "done"]
    granted = _granted()

    counts = db.query_one(
        """SELECT
             SUM(CASE WHEN kind='code' AND correct=1 AND hints_used=0
                      THEN 1 ELSE 0 END) AS unaided,
             SUM(CASE WHEN kind='code' AND correct=1 THEN 1 ELSE 0 END) AS fits
           FROM attempts"""
    ) or {}
    unaided = counts.get("unaided") or 0
    fits = counts.get("fits") or 0

    figures = db.query_one(
        "SELECT COUNT(*) AS n FROM attempts WHERE kind='cell'") or {"n": 0}
    maths_asked = db.query_one(
        "SELECT COUNT(*) AS n FROM math_cache") or {"n": 0}

    topics = db.topics_with_mastery()
    week_one = [t for t in topics if t["week"] == 1]
    week_one_done = bool(week_one) and all(
        scheduler.is_mastered(t["slug"]) or
        db.get_project(t["slug"])["status"] == "done" for t in week_one)

    kinds = {t["slug"]: t["kind"] for t in topics}
    milestone_done = any(statuses.get(s) == "done"
                         for s, k in kinds.items() if k == "milestone")
    capstone_done = any(statuses.get(s) == "done"
                        for s, k in kinds.items() if k == "capstone")
    perfect = db.query_one(
        "SELECT 1 FROM sessions WHERE notes LIKE '%perfect%' LIMIT 1") is not None

    tests = {
        "first-fit": fits >= 1,
        "week-one": week_one_done,
        "ten-challenges": solved >= 10,
        "fifty-challenges": solved >= 50,
        "unaided": unaided >= 10,
        "visualiser": (figures.get("n") or 0) >= 25,
        "streak-7": streak >= 7,
        "streak-30": streak >= 30,
        "asker": (maths_asked.get("n") or 0) >= 25,
        "perfectionist": perfect,
        "builder": len(done) >= 1,
        "architect": milestone_done,
        "graduate": capstone_done,
    }

    out = []
    for badge in BADGES:
        earned = (badge.slug in granted if badge.slug in EARNED_IN_THE_MOMENT
                  else bool(tests.get(badge.slug)))
        out.append({"slug": badge.slug, "name": badge.name,
                    "description": badge.description, "icon": badge.icon,
                    "group": badge.group, "earned": earned})
    return out


# ---------------------------------------------------------------------------
# Skill mastery
# ---------------------------------------------------------------------------

def skill_map() -> list[dict[str, Any]]:
    """One row per skill area with how much of it you have cleared.

    This answers 'what am I actually weak at', which topic mastery cannot: you
    can have read the leakage lesson and still ship a leaking model.
    """
    progress = db.bank_progress()
    rows = []
    for skill, total in sorted(progress["skill_totals"].items()):
        solved = progress["by_skill"].get(skill, 0)
        share = solved / total if total else 0.0
        rows.append({
            "skill": skill, "solved": solved, "total": total, "share": share,
            "state": ("strong" if share >= 0.8 else
                      "working" if share >= 0.3 else
                      "started" if solved else "untouched"),
        })
    rows.sort(key=lambda r: (r["share"], -r["total"]))
    return rows


def weakest_skills(limit: int = 3) -> list[dict[str, Any]]:
    return [r for r in skill_map() if r["total"] >= 2][:limit]


# ---------------------------------------------------------------------------
# Daily goal
# ---------------------------------------------------------------------------

def daily_goal() -> dict[str, Any]:
    from . import analytics                       # noqa: PLC0415 - cyclic

    today = analytics.daily_series(1)[0]
    target = LEARNING.target_minutes_per_day
    done = today["minutes"]
    share = min(1.0, done / target) if target else 0.0

    if share >= 1.0:
        db.award_xp("daily_goal", XP_RULES["daily_goal"],
                    reason="Hit the daily study target",
                    dedupe_key=f"daily:{today['date']}")

    return {"date": today["date"], "minutes": done, "target": target,
            "share": share, "met": share >= 1.0,
            "attempts": today["attempts"],
            "challenges_solved": today.get("challenges_solved", 0)}


def summary() -> dict[str, Any]:
    """Everything the dashboard's progress strip needs, in one call."""
    level = status()
    progress = db.bank_progress()
    badges = earned_badges()
    return {
        "level": level,
        "xp": level["xp"],
        "badges_earned": sum(1 for b in badges if b["earned"]),
        "badges_total": len(badges),
        "challenges_solved": progress["solved"],
        "challenges_total": progress["total"],
        "streak": scheduler.streak_days(),
        "goal": daily_goal(),
        "weakest": weakest_skills(3),
    }
