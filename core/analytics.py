"""Derived statistics for the dashboard and the weekly report.

Everything here is computed from the attempts and submissions tables on
demand. Nothing is cached in a summary table that could drift out of sync with
the underlying rows, which is the usual way progress dashboards start lying.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime, timedelta
from typing import Any

from . import db, scheduler
from .config import LEARNING


def _day_of(iso_timestamp: str) -> str:
    return iso_timestamp[:10]


def daily_series(days: int = 28) -> list[dict[str, Any]]:
    """One row per calendar day, including days with no activity.

    Zero-activity days are included deliberately: a trend chart that silently
    skips them compresses a week off into nothing and makes a broken habit look
    like a continuous one.
    """
    rows = db.attempts_since(days)
    buckets: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"attempts": 0, "correct": 0, "seconds": 0, "topics": set(),
                 "challenges": 0, "challenges_solved": 0}
    )
    for row in rows:
        day = _day_of(row["created_at"])
        bucket = buckets[day]
        bucket["attempts"] += 1
        bucket["correct"] += row["correct"]
        bucket["seconds"] += row["seconds"]
        bucket["topics"].add(row["topic_slug"])
        if row["kind"] == "code":
            bucket["challenges"] += 1
            bucket["challenges_solved"] += row["correct"]

    out = []
    today = date.today()
    for offset in range(days - 1, -1, -1):
        day_obj = today - timedelta(days=offset)
        key = day_obj.isoformat()
        bucket = buckets.get(key)
        if bucket:
            accuracy = (bucket["correct"] / bucket["attempts"]) if bucket["attempts"] else 0.0
            out.append({
                "date": key,
                "label": day_obj.strftime("%d %b"),
                "weekday": day_obj.strftime("%a"),
                "attempts": bucket["attempts"],
                "correct": bucket["correct"],
                "accuracy": round(accuracy, 3),
                "minutes": round(bucket["seconds"] / 60),
                "topics": len(bucket["topics"]),
                "challenges": bucket["challenges"],
                "challenges_solved": bucket["challenges_solved"],
            })
        else:
            out.append({
                "date": key, "label": day_obj.strftime("%d %b"),
                "weekday": day_obj.strftime("%a"), "attempts": 0, "correct": 0,
                "accuracy": None, "minutes": 0, "topics": 0,
                "challenges": 0, "challenges_solved": 0,
            })
    return out


def accuracy_by_kind(days: int = 7) -> dict[str, dict[str, Any]]:
    """Accuracy split by MCQ, flashcard and code. These measure different
    things — recognising an answer, recalling a fact, and writing working code —
    so a single blended accuracy number would hide the one that matters."""
    rows = db.attempts_since(days)
    buckets: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "correct": 0})
    for row in rows:
        bucket = buckets[row["kind"]]
        bucket["n"] += 1
        bucket["correct"] += row["correct"]
    return {
        kind: {
            "attempts": b["n"],
            "correct": b["correct"],
            "accuracy": round(b["correct"] / b["n"], 3) if b["n"] else None,
        }
        for kind, b in sorted(buckets.items())
    }


def mastery_by_week() -> list[dict[str, Any]]:
    """Mastery rolled up per curriculum week — the heatmap in the report."""
    topics = db.topics_with_mastery()
    weeks: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for topic in topics:
        weeks[topic["week"]].append(topic)

    out = []
    for week in sorted(weeks):
        items = weeks[week]
        # The rows came from topics_with_mastery, which already joined these
        # in. Asking again is one round trip per topic for data in hand.
        scores = [t.get("score") or 0.0 for t in items]
        started = [t for t in items if (t.get("attempts") or 0) > 0]
        out.append({
            "week": week,
            "topics": items,
            "mean_score": round(sum(scores) / len(scores), 3) if scores else 0.0,
            "mastered": sum(1 for t in items if scheduler.is_mastered(t["slug"])),
            "started": len(started),
            "total": len(items),
        })
    return out


def topic_table() -> list[dict[str, Any]]:
    """Every topic with its mastery state, for the table view that accompanies
    the heatmap. An accessible chart always ships a table alongside it."""
    rows = []
    for topic in db.topics_with_mastery():
        m = scheduler.mastery_of(topic["slug"])
        if m["attempts"] == 0:
            state = "not started"
        elif scheduler.is_mastered(topic["slug"]):
            state = "mastered"
        elif m["score"] < LEARNING.struggling_threshold:
            state = "struggling"
        else:
            state = "in progress"
        rows.append({
            "slug": topic["slug"], "title": topic["title"], "week": topic["week"],
            "day": topic["day"], "track": topic["track"],
            "difficulty": topic["difficulty"],
            "score": round(m["score"], 3), "attempts": m["attempts"],
            "minutes": round(m["seconds_spent"] / 60),
            "state": state,
        })
    return rows


def challenge_stats(days: int = 7) -> dict[str, Any]:
    """Code-specific numbers: first-try rate, hint reliance, retry depth."""
    cutoff = (datetime.now() - timedelta(days=days)).isoformat(timespec="seconds")
    rows = db.query(
        "SELECT * FROM submissions WHERE created_at >= ? ORDER BY created_at",
        (cutoff,),
    )
    if not rows:
        return {"submissions": 0, "solved": 0, "first_try": 0,
                "first_try_rate": None, "mean_attempts": None,
                "timeouts": 0, "errors": 0}

    by_challenge: dict[Any, list[Any]] = defaultdict(list)
    for row in rows:
        by_challenge[row["content_id"]].append(row)

    solved = first_try = 0
    attempt_counts = []
    for subs in by_challenge.values():
        subs.sort(key=lambda r: r["attempt_no"])
        winner = next((s for s in subs if s["verdict"] == "pass"), None)
        if winner:
            solved += 1
            attempt_counts.append(winner["attempt_no"])
            if winner["attempt_no"] == 1:
                first_try += 1

    hint_rows = db.query(
        "SELECT SUM(hints_used) AS h, COUNT(*) AS n FROM attempts "
        "WHERE kind='code' AND created_at >= ?", (cutoff,),
    )
    hints = hint_rows[0]["h"] or 0 if hint_rows else 0
    code_attempts = hint_rows[0]["n"] or 0 if hint_rows else 0

    return {
        "submissions": len(rows),
        "challenges_attempted": len(by_challenge),
        "solved": solved,
        "first_try": first_try,
        "first_try_rate": round(first_try / len(by_challenge), 3) if by_challenge else None,
        "mean_attempts": round(sum(attempt_counts) / len(attempt_counts), 2)
                         if attempt_counts else None,
        "timeouts": sum(1 for r in rows if r["verdict"] == "timeout"),
        "errors": sum(1 for r in rows if r["verdict"] == "error"),
        "hints_used": hints,
        "hints_per_challenge": round(hints / len(by_challenge), 2) if by_challenge else None,
        "code_attempts": code_attempts,
    }


def week_summary(week_offset: int = 0) -> dict[str, Any]:
    """Everything the weekly report needs, for the 7 days ending today (offset 0)
    or a previous week (offset 1 = the week before)."""
    end = date.today() - timedelta(days=7 * week_offset)
    start = end - timedelta(days=6)

    series = [d for d in daily_series(28)
              if start.isoformat() <= d["date"] <= end.isoformat()]

    attempts = sum(d["attempts"] for d in series)
    correct = sum(d["correct"] for d in series)
    minutes = sum(d["minutes"] for d in series)
    active_days = sum(1 for d in series if d["attempts"] > 0)

    status = scheduler.plan_status()
    weak = scheduler.weak_topics(limit=5)

    return {
        "start": start.isoformat(),
        "end": end.isoformat(),
        "series": series,
        "attempts": attempts,
        "correct": correct,
        "accuracy": round(correct / attempts, 3) if attempts else None,
        "minutes": minutes,
        "hours": round(minutes / 60, 1),
        "target_hours": round(7 * LEARNING.target_minutes_per_day / 60, 1),
        "active_days": active_days,
        "by_kind": accuracy_by_kind(7),
        "challenges": challenge_stats(7),
        "mastery_by_week": mastery_by_week(),
        "topic_table": topic_table(),
        "weak_topics": [
            {"title": t["title"], "slug": t["slug"],
             "score": round(t.get("score") or 0.0, 3),
             "attempts": t.get("attempts") or 0}
            for t in weak
        ],
        "status": status,
        "streak": scheduler.streak_days(),
        "recent_mistakes": db.recent_mistakes(8),
        "next_up": [
            {"title": r.topic["title"], "slug": r.topic["slug"],
             "reason": r.reason, "kind": r.kind}
            for r in scheduler.recommend_next(3)
        ],
    }
