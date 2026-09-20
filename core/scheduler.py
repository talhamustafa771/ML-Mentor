"""Decides what you study next, and remembers what you got wrong.

Two mechanisms, deliberately kept separate:

  Mastery   - an exponentially weighted moving average of per-attempt scores,
              per topic. Recent performance dominates, so a topic you have
              fixed stops being flagged, and one you have forgotten starts
              being flagged again.

  Review    - SM-2 spaced repetition on individual flashcards, independent of
              topic mastery. A topic can be mastered while three of its cards
              are still due.

The next-topic decision combines them with prerequisite gating, so the plan
never hands you graphs before you can write a loop.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any

from . import db
from .config import LEARNING


# ---------------------------------------------------------------------------
# Mastery
# ---------------------------------------------------------------------------

def record_outcome(
    topic_slug: str, score: float, *, seconds: int = 0
) -> float:
    """Fold one graded attempt into a topic's mastery score. Returns the new score.

    score is 0..1. An MCQ is 0 or 1; a code submission is the fraction of tests
    passed, so a near-miss counts for more than a blank page — which is both
    fairer and a better training signal for the scheduler.
    """
    score = max(0.0, min(1.0, float(score)))
    row = db.query_one("SELECT * FROM mastery WHERE topic_slug=?", (topic_slug,))
    if row is None:
        db.execute("INSERT OR IGNORE INTO mastery(topic_slug) VALUES(?)", (topic_slug,))
        row = db.query_one("SELECT * FROM mastery WHERE topic_slug=?", (topic_slug,))

    attempts = (row["attempts"] or 0) + 1
    correct = (row["correct"] or 0) + (1 if score >= 0.999 else 0)
    alpha = LEARNING.mastery_ewma_alpha

    if (row["attempts"] or 0) == 0:
        new_score = score           # first attempt seeds the average
    else:
        new_score = (1 - alpha) * (row["score"] or 0.0) + alpha * score

    completed_at = row["completed_at"]
    if (completed_at is None
            and new_score >= LEARNING.mastery_threshold
            and attempts >= LEARNING.min_attempts_for_mastery):
        completed_at = db.now()

    db.execute(
        """UPDATE mastery
           SET score=?, attempts=?, correct=?, seconds_spent=seconds_spent+?,
               first_seen_at=COALESCE(first_seen_at, ?), last_seen_at=?,
               completed_at=?
           WHERE topic_slug=?""",
        (new_score, attempts, correct, int(seconds), db.now(), db.now(),
         completed_at, topic_slug),
    )
    return new_score


def mastery_of(topic_slug: str) -> dict[str, Any]:
    row = db.query_one("SELECT * FROM mastery WHERE topic_slug=?", (topic_slug,))
    if row is None:
        return {"score": 0.0, "attempts": 0, "correct": 0, "completed_at": None,
                "seconds_spent": 0}
    return dict(row)


def _mastered_from(row: dict[str, Any]) -> bool:
    """is_mastered, decided from a row already in hand rather than a new query.

    Kept beside is_mastered so the two rules cannot drift apart; whenever the
    threshold changes, both read it from the same constants.
    """
    return (float(row.get("score") or 0.0) >= LEARNING.mastery_threshold
            and int(row.get("attempts") or 0) >= LEARNING.min_attempts_for_mastery)


def is_mastered(topic_slug: str) -> bool:
    m = mastery_of(topic_slug)
    return (m["score"] >= LEARNING.mastery_threshold
            and m["attempts"] >= LEARNING.min_attempts_for_mastery)


def prereqs_met(topic: dict[str, Any],
                known: dict[str, dict[str, Any]] | None = None
                ) -> tuple[bool, list[str]]:
    """A prerequisite counts as met at a slightly lower bar than full mastery.
    Requiring full mastery of every prereq makes the plan stall on one weak
    topic; requiring nothing makes gating meaningless. The struggling threshold
    is the compromise, and it is a named constant so it can be tuned."""
    missing = []
    for slug in topic.get("prereqs", []):
        m = known.get(slug) if known is not None else mastery_of(slug)
        if m is None:
            m = {"attempts": 0, "score": 0.0}
        if (m.get("attempts") or 0) == 0 or (m.get("score") or 0.0) < LEARNING.struggling_threshold:
            missing.append(slug)
    return (not missing), missing


# ---------------------------------------------------------------------------
# Spaced repetition (SM-2)
# ---------------------------------------------------------------------------

def schedule_review(content_id: str, topic_slug: str, grade: int) -> dict[str, Any]:
    """Apply SM-2 after a flashcard review.

    grade is 0-5 in the original algorithm. The UI exposes four buttons mapped
    to 1 (Again), 3 (Hard), 4 (Good), 5 (Easy), which is the mapping Anki uses
    and which learners rate more reliably than a 6-point scale.
    """
    grade = max(0, min(5, int(grade)))
    row = db.query_one("SELECT * FROM reviews WHERE content_id=?", (content_id,))

    if row is None:
        ease, interval, reps, lapses = LEARNING.sm2_initial_ease, 0, 0, 0
    else:
        ease = row["ease"]
        interval = row["interval_days"]
        reps = row["repetitions"]
        lapses = row["lapses"]

    if grade < 3:
        # Failed recall: reset the interval but keep the ease penalty bounded.
        reps = 0
        interval = 1
        lapses += 1
        ease = max(LEARNING.sm2_min_ease, ease - 0.20)
    else:
        reps += 1
        if reps == 1:
            interval = 1
        elif reps == 2:
            interval = 6
        else:
            interval = max(1, round(interval * ease))
        ease = max(
            LEARNING.sm2_min_ease,
            ease + (0.1 - (5 - grade) * (0.08 + (5 - grade) * 0.02)),
        )

    due = (date.today() + timedelta(days=interval)).isoformat()
    db.execute(
        """INSERT INTO reviews(content_id, topic_slug, ease, interval_days,
                               repetitions, due_date, lapses, last_grade, updated_at)
           VALUES(?,?,?,?,?,?,?,?,?)
           ON CONFLICT(content_id) DO UPDATE SET
             ease=excluded.ease, interval_days=excluded.interval_days,
             repetitions=excluded.repetitions, due_date=excluded.due_date,
             lapses=excluded.lapses, last_grade=excluded.last_grade,
             updated_at=excluded.updated_at""",
        (content_id, topic_slug, ease, interval, reps, due, lapses, grade, db.now()),
    )
    return {"ease": round(ease, 2), "interval_days": interval,
            "repetitions": reps, "due_date": due, "lapses": lapses}


def due_reviews(limit: int = 50) -> list[dict[str, Any]]:
    """Cards due today or overdue, most overdue first."""
    rows = db.query(
        """SELECT r.*, c.body, c.kind, t.title AS topic_title
           FROM reviews r
           JOIN content c ON c.id = r.content_id
           JOIN topics  t ON t.slug = r.topic_slug
           WHERE r.due_date <= ? AND c.retired = 0
           ORDER BY r.due_date ASC LIMIT ?""",
        (date.today().isoformat(), limit),
    )
    import json
    out = []
    for r in rows:
        d = dict(r)
        d["body"] = json.loads(d["body"])
        out.append(d)
    return out


def due_count() -> int:
    row = db.query_one(
        "SELECT COUNT(*) AS n FROM reviews WHERE due_date <= ?",
        (date.today().isoformat(),),
    )
    return row["n"] if row else 0


# ---------------------------------------------------------------------------
# Next-topic selection
# ---------------------------------------------------------------------------

@dataclass
class Recommendation:
    topic: dict[str, Any]
    reason: str
    kind: str                # new | remediate | review | blocked_fallback
    urgency: float           # 0..1, for ordering the dashboard


def recommend_next(max_items: int = 3) -> list[Recommendation]:
    """What to do next, in priority order.

    Priority, highest first:
      1. A topic you have started and are failing (score below the struggling
         threshold with enough attempts to trust it). Moving on from a topic you
         cannot do is how week 8 becomes impossible.
      2. A topic whose mastery has decayed below threshold after being learned.
      3. The next unstarted topic in plan order whose prerequisites are met.
      4. If everything ahead is blocked, the weakest unmet prerequisite.
    """
    # One query for the plan and one for mastery, then everything below reads
    # from memory. It used to ask per topic inside four separate loops, and
    # prereqs_met asked again per prerequisite on top of that.
    topics = db.topics_with_mastery()
    by_slug = {t["slug"]: t for t in topics}
    known = {t["slug"]: t for t in topics}
    out: list[Recommendation] = []

    # 1. Actively struggling
    for t in topics:
        m = t
        if (m["attempts"] >= LEARNING.min_attempts_for_mastery
                and m["score"] < LEARNING.struggling_threshold):
            out.append(Recommendation(
                t,
                f"You are at {int(m['score'] * 100)}% on this after {m['attempts']} "
                f"attempts. Clearing it now prevents it blocking later topics.",
                "remediate",
                0.95,
            ))

    # 2. Decayed after being learned
    for t in topics:
        m = t
        if (m.get("completed_at")
                and m["score"] < LEARNING.mastery_threshold
                and m["score"] >= LEARNING.struggling_threshold):
            out.append(Recommendation(
                t,
                f"You mastered this earlier but recent attempts dropped it to "
                f"{int(m['score'] * 100)}%. A short refresh should restore it.",
                "review",
                0.7,
            ))

    # 3. Next unstarted in plan order
    for t in topics:
        if t["attempts"] > 0:
            continue
        ok, missing = prereqs_met(t, known)
        if ok:
            out.append(Recommendation(
                t,
                f"Next in the plan (day {t['day']}, week {t['week']}). "
                "Prerequisites are in place.",
                "new",
                0.6,
            ))
            break

    # 4. Everything blocked: surface the weakest blocker
    if not out:
        for t in topics:
            if t["attempts"] > 0:
                continue
            ok, missing = prereqs_met(t, known)
            if not ok and missing:
                blocker = by_slug.get(missing[0])
                if blocker:
                    out.append(Recommendation(
                        blocker,
                        f"'{t['title']}' is blocked until you strengthen this one.",
                        "blocked_fallback",
                        0.9,
                    ))
                    break

    # Everything mastered
    if not out:
        weakest = min(topics, key=lambda t: t["score"])
        out.append(Recommendation(
            weakest,
            "Every topic is at or above the mastery bar. This is your weakest, "
            "so it is the best use of a session.",
            "review",
            0.4,
        ))

    out.sort(key=lambda r: -r.urgency)
    return out[:max_items]


def plan_status() -> dict[str, Any]:
    """Headline numbers for the dashboard. All derived, none stored, so they
    cannot drift out of sync with the attempts table."""
    # Every figure below comes from the rows already fetched. This used to ask
    # the database again for each topic, four times over — 364 queries to
    # compute numbers that were sitting in memory the whole time. Against a
    # local file that was merely wasteful; against a hosted database it was
    # a minute of staring at a spinner.
    topics = db.topics_with_mastery()
    total = len(topics)
    mastered = sum(1 for t in topics if _mastered_from(t))
    started = sum(1 for t in topics if t["attempts"] > 0)
    seconds = sum(t.get("seconds_spent") or 0 for t in topics)

    struggling = [
        t for t in topics
        if (t["attempts"] >= LEARNING.min_attempts_for_mastery
            and t["score"] < LEARNING.struggling_threshold)
    ]

    first_row = db.query_one("SELECT MIN(created_at) AS first FROM attempts")
    started_on = first_row["first"] if first_row and first_row["first"] else None
    day_index = 1
    if started_on:
        delta = (date.today() - datetime.fromisoformat(started_on).date()).days
        day_index = max(1, delta + 1)

    return {
        "total_topics": total,
        "mastered": mastered,
        "started": started,
        "not_started": total - started,
        "percent_complete": round(100 * mastered / total, 1) if total else 0.0,
        "hours_spent": round(seconds / 3600, 1),
        "struggling": struggling,
        "due_reviews": due_count(),
        "day_index": min(day_index, LEARNING.plan_weeks * 7),
        "plan_days": LEARNING.plan_weeks * 7,
        "on_track": mastered >= (day_index - 3),   # 3 days of slack before nagging
        "started_on": started_on,
    }


def streak_days() -> int:
    """Consecutive days ending today (or yesterday) with at least one attempt."""
    rows = db.query(
        "SELECT DISTINCT substr(created_at, 1, 10) AS d FROM attempts ORDER BY d DESC"
    )
    days = [r["d"] for r in rows]
    if not days:
        return 0
    today_ = date.today()
    # Allow the streak to survive until the end of the following day, so a late
    # study session at 1am does not read as a broken streak.
    first = date.fromisoformat(days[0])
    if (today_ - first).days > 1:
        return 0
    streak = 0
    cursor = first
    for d in days:
        if date.fromisoformat(d) == cursor:
            streak += 1
            cursor -= timedelta(days=1)
        else:
            break
    return streak


def weak_topics(limit: int = 5) -> list[dict[str, Any]]:
    """Started topics with the lowest mastery. Feeds the weekly report and the
    interview-mode question selection."""
    topics = [t for t in db.topics_with_mastery() if t["attempts"] > 0]
    topics.sort(key=lambda t: t["score"])
    return topics[:limit]


def next_topic() -> dict[str, Any] | None:
    """The single topic to open now, as a topic dict.

    recommend_next answers "what are the three best things to do", which is the
    right question for a study plan and the wrong one for a dashboard that has
    to put one button on the screen. This collapses it, and falls back to plan
    order when the recommender has nothing to say — a new install has no
    attempts, so nothing is struggling, decayed or due, and the honest answer
    is simply day 1.
    """
    picks = recommend_next(1)
    if picks:
        return picks[0].topic
    for topic in db.topics_with_mastery():
        if not _mastered_from(topic):
            return topic
    return None


def weakest_topics(limit: int = 3) -> list[dict[str, Any]]:
    """Started topics with the lowest mastery, for the dashboard."""
    return weak_topics(limit)
