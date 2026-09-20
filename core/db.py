"""Persistence layer.

One connection per thread, created lazily, over whichever database
core/dbdriver.py selected: the local SQLite file on a laptop, or a hosted
libSQL server when the app runs somewhere whose disk is wiped on restart. The
SQL is identical for both.

Every table is created by init_db(), which is idempotent and safe to call on
every app start. Schema changes go in MIGRATIONS as additive statements; each
runs once and failures on already-applied statements are swallowed
deliberately (see _apply_migrations).
"""
from __future__ import annotations

import hashlib
import json
import threading
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Iterator

from . import dbdriver
from .config import DB_PATH

_local = threading.local()

SCHEMA = """
CREATE TABLE IF NOT EXISTS settings (
    key         TEXT PRIMARY KEY,
    value       TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

-- One row per curriculum topic, seeded from core/curriculum.py.
CREATE TABLE IF NOT EXISTS topics (
    slug         TEXT PRIMARY KEY,
    title        TEXT NOT NULL,
    track        TEXT NOT NULL,            -- foundations|supervised|unsupervised|advanced
    week         INTEGER NOT NULL,
    day          INTEGER NOT NULL,         -- 1..91, the planned day
    difficulty   TEXT NOT NULL,            -- beginner | intermediate | advanced
    minutes      INTEGER NOT NULL,
    summary      TEXT NOT NULL,
    objectives   TEXT NOT NULL,            -- JSON list[str]
    real_world   TEXT NOT NULL,            -- JSON list[str]
    prereqs      TEXT NOT NULL,            -- JSON list[slug]
    sort_order   INTEGER NOT NULL,
    kind         TEXT NOT NULL DEFAULT 'lesson',  -- lesson|project|milestone|capstone
    -- The ideas this day introduces and the mathematics it leans on. Stored
    -- so the horizon and the Math Helper can be answered from the database
    -- rather than by importing the curriculum on every page.
    concepts     TEXT NOT NULL DEFAULT '[]',      -- JSON list[str]
    maths        TEXT NOT NULL DEFAULT '[]',      -- JSON list[str]
    dataset      TEXT NOT NULL DEFAULT ''         -- the bundled dataset it uses
);

-- Rolling mastery per topic. Updated by scheduler.record_outcome().
CREATE TABLE IF NOT EXISTS mastery (
    topic_slug     TEXT PRIMARY KEY REFERENCES topics(slug),
    score          REAL NOT NULL DEFAULT 0.0,   -- EWMA of per-attempt correctness
    attempts       INTEGER NOT NULL DEFAULT 0,
    correct        INTEGER NOT NULL DEFAULT 0,
    seconds_spent  INTEGER NOT NULL DEFAULT 0,
    first_seen_at  TEXT,
    last_seen_at   TEXT,
    completed_at   TEXT                          -- set when score crosses threshold
);

-- A study session groups attempts so the dashboard can show "today".
CREATE TABLE IF NOT EXISTS sessions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    topic_slug  TEXT REFERENCES topics(slug),
    mode        TEXT NOT NULL,             -- learn | practice | code | interview
    started_at  TEXT NOT NULL,
    ended_at    TEXT,
    seconds     INTEGER NOT NULL DEFAULT 0,
    notes       TEXT
);

-- Every graded interaction: one MCQ answer, one flashcard grade, one code run.
CREATE TABLE IF NOT EXISTS attempts (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id   INTEGER REFERENCES sessions(id),
    topic_slug   TEXT NOT NULL REFERENCES topics(slug),
    item_id      TEXT,                     -- content.id when applicable
    kind         TEXT NOT NULL,            -- mcq | card | code | activity
    correct      INTEGER NOT NULL,         -- 0/1
    score        REAL NOT NULL DEFAULT 0.0,-- 0..1, partial credit for code
    seconds      INTEGER NOT NULL DEFAULT 0,
    hints_used   INTEGER NOT NULL DEFAULT 0,
    payload      TEXT,                     -- JSON: answer given, verdict, etc.
    created_at   TEXT NOT NULL
);

-- Generated content cache. Keyed by a deterministic hash so the same topic and
-- difficulty does not burn tokens twice, and so questions are stable across
-- reruns of a Streamlit page.
CREATE TABLE IF NOT EXISTS content (
    id          TEXT PRIMARY KEY,
    topic_slug  TEXT NOT NULL REFERENCES topics(slug),
    kind        TEXT NOT NULL,             -- lesson|mcq|card|experiment|project|realworld
    difficulty  TEXT NOT NULL,
    body        TEXT NOT NULL,             -- JSON payload, shape depends on kind
    provider    TEXT,
    model       TEXT,
    times_shown INTEGER NOT NULL DEFAULT 0,
    retired     INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT NOT NULL,
    source      TEXT NOT NULL DEFAULT 'generated',  -- generated | bank
    -- Set once the code in this item has been executed against the real
    -- dataset: 'passed', or the cell number and error that stopped it. A
    -- lesson is never shown until this says passed.
    verified    TEXT NOT NULL DEFAULT ''
);

-- Answers from the Math Helper, cached so a term is only ever paid for once.
-- Deliberately not part of `content`: a maths question belongs to a term and a
-- day, not to a topic, and it must survive the content cache being cleared.
CREATE TABLE IF NOT EXISTS math_cache (
    id          TEXT PRIMARY KEY,          -- hash of the term and the day
    term        TEXT NOT NULL,
    day         INTEGER NOT NULL,
    body        TEXT NOT NULL,             -- JSON: title, plain, formula, ...
    provider    TEXT,
    model       TEXT,
    times_shown INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT NOT NULL
);

-- SM-2 spaced repetition state, one row per flashcard actually reviewed.
CREATE TABLE IF NOT EXISTS reviews (
    content_id   TEXT PRIMARY KEY REFERENCES content(id),
    topic_slug   TEXT NOT NULL,
    ease         REAL NOT NULL DEFAULT 2.5,
    interval_days INTEGER NOT NULL DEFAULT 0,
    repetitions  INTEGER NOT NULL DEFAULT 0,
    due_date     TEXT NOT NULL,
    lapses       INTEGER NOT NULL DEFAULT 0,
    last_grade   INTEGER,
    updated_at   TEXT NOT NULL
);

-- Full history of code the user wrote, with verdict and review. This is what
-- makes "where was I wrong last time" answerable.
CREATE TABLE IF NOT EXISTS submissions (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    -- Deliberately NOT a foreign key. A submission is a permanent record of
    -- what you wrote; it must survive the content cache being cleared, and an
    -- FK violation mid-session would lose the attempt entirely.
    content_id   TEXT,
    topic_slug   TEXT NOT NULL,
    attempt_no   INTEGER NOT NULL,
    code         TEXT NOT NULL,
    verdict      TEXT NOT NULL,            -- pass | fail | error | timeout
    passed       INTEGER NOT NULL DEFAULT 0,
    total        INTEGER NOT NULL DEFAULT 0,
    failures     TEXT,                     -- JSON list of failing cases
    stderr       TEXT,
    review       TEXT,                     -- JSON: strengths, issues, optimized code
    seconds      INTEGER NOT NULL DEFAULT 0,
    created_at   TEXT NOT NULL
);

-- Conversation log. Kept so the tutor has continuity and the scheduler can see
-- what the user asked about, not just what they scored.
CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    topic_slug  TEXT,
    role        TEXT NOT NULL,             -- user | assistant | system
    content     TEXT NOT NULL,
    created_at  TEXT NOT NULL
);

-- Daily rollup, written by analytics.rollup_day(). Cheap dashboard reads.
CREATE TABLE IF NOT EXISTS daily_stats (
    day             TEXT PRIMARY KEY,      -- ISO date
    seconds         INTEGER NOT NULL DEFAULT 0,
    attempts        INTEGER NOT NULL DEFAULT 0,
    correct         INTEGER NOT NULL DEFAULT 0,
    topics_touched  INTEGER NOT NULL DEFAULT 0,
    experiments_done INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_attempts_topic   ON attempts(topic_slug);
CREATE INDEX IF NOT EXISTS idx_attempts_created ON attempts(created_at);
CREATE INDEX IF NOT EXISTS idx_content_topic    ON content(topic_slug, kind, retired);
CREATE INDEX IF NOT EXISTS idx_reviews_due      ON reviews(due_date);
CREATE INDEX IF NOT EXISTS idx_subs_topic       ON submissions(topic_slug, created_at);
"""

# Additive-only. Each statement runs on every start; already-applied ones raise
# OperationalError, which we swallow. Never put a destructive statement here.
MIGRATIONS: list[str] = [
    "ALTER TABLE topics ADD COLUMN kind TEXT NOT NULL DEFAULT 'lesson'",
    "ALTER TABLE topics ADD COLUMN concepts TEXT NOT NULL DEFAULT '[]'",
    "ALTER TABLE topics ADD COLUMN maths TEXT NOT NULL DEFAULT '[]'",
    "ALTER TABLE topics ADD COLUMN dataset TEXT NOT NULL DEFAULT ''",
    "ALTER TABLE content ADD COLUMN source TEXT NOT NULL DEFAULT 'generated'",
    "ALTER TABLE content ADD COLUMN verified TEXT NOT NULL DEFAULT ''",
    "CREATE INDEX IF NOT EXISTS idx_content_verified ON content(kind, verified)",
    # Project work: one row per project, milestone or capstone day.
    """CREATE TABLE IF NOT EXISTS project_work (
        topic_slug  TEXT PRIMARY KEY,
        status      TEXT NOT NULL DEFAULT 'not_started',
        notes       TEXT,
        code        TEXT,
        checklist   TEXT,
        started_at  TEXT,
        finished_at TEXT,
        updated_at  TEXT NOT NULL
    )""",
    # The XP ledger. Levels are derived from the sum, never stored, so they can
    # never disagree with the events that produced them.
    """CREATE TABLE IF NOT EXISTS xp_events (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        kind       TEXT NOT NULL,
        amount     INTEGER NOT NULL,
        reason     TEXT,
        topic_slug TEXT,
        dedupe_key TEXT UNIQUE,
        created_at TEXT NOT NULL
    )""",
    "CREATE INDEX IF NOT EXISTS idx_xp_created ON xp_events(created_at)",
    # Bookmarks and personal notes on any item.
    """CREATE TABLE IF NOT EXISTS bookmarks (
        ref        TEXT PRIMARY KEY,
        kind       TEXT NOT NULL,
        label      TEXT,
        note       TEXT,
        created_at TEXT NOT NULL
    )""",
]


def _connect() -> dbdriver.Connection:
    """Open the database the configuration points at.

    Which one that is — the local file or a hosted libSQL server — is decided
    in core/dbdriver.py from TURSO_DATABASE_URL. Nothing below this line cares:
    the statements are identical and the rows come back as dicts either way.
    """
    return dbdriver.connect(DB_PATH)


def get_conn() -> dbdriver.Connection:
    conn = getattr(_local, "conn", None)
    if conn is None:
        conn = _connect()
        _local.conn = conn
    return conn


def reset_conn() -> None:
    """Drop this thread's connection so the next call reopens it.

    Used when the credentials change in Settings, so switching between the
    local file and the hosted database takes effect without a restart.
    """
    conn = getattr(_local, "conn", None)
    if conn is not None:
        conn.close()
    _local.conn = None


@contextmanager
def tx() -> Iterator[dbdriver.Connection]:
    """Transaction scope. Commits on success, rolls back on any exception."""
    conn = get_conn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def _apply_migrations(conn: dbdriver.Connection) -> None:
    for statement in MIGRATIONS:
        try:
            conn.execute(statement)
        except Exception as exc:  # noqa: BLE001 - narrowed on the next line
            # Already applied is the expected outcome for an additive
            # migration that has run before. Anything else is a real fault and
            # is raised, so a typo in a migration cannot hide here.
            if not dbdriver.is_already_applied(exc):
                raise
    conn.commit()


def init_db() -> None:
    conn = get_conn()
    conn.executescript(SCHEMA)
    _apply_migrations(conn)
    conn.commit()


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def today() -> str:
    return date.today().isoformat()


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def query(sql: str, params: Iterable[Any] = ()) -> list[dict[str, Any]]:
    return get_conn().execute(sql, tuple(params)).fetchall()


def query_one(sql: str, params: Iterable[Any] = ()) -> dict[str, Any] | None:
    return get_conn().execute(sql, tuple(params)).fetchone()


def execute(sql: str, params: Iterable[Any] | dict[str, Any] = ()) -> int:
    with tx() as conn:
        # A dict means named placeholders (:name). The driver either passes
        # them straight through or rewrites them, depending on the backend.
        cur = conn.execute(sql, params)
        return cur.lastrowid or cur.rowcount


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

def get_settings() -> dict[str, str]:
    return {r["key"]: r["value"] for r in query("SELECT key, value FROM settings")}


def set_setting(key: str, value: str) -> None:
    execute(
        "INSERT INTO settings(key, value, updated_at) VALUES(?,?,?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
        (key, value, now()),
    )


def get_setting(key: str, default: str | None = None) -> str | None:
    row = query_one("SELECT value FROM settings WHERE key=?", (key,))
    return row["value"] if row else default


# ---------------------------------------------------------------------------
# Topics
# ---------------------------------------------------------------------------

# How many rows to write per statement. SQLite's parameter ceiling is 999 by
# default, and the widest row here carries 14 values, so 60 leaves ample room
# while cutting a 91-row seed from 91 statements to two.
_BATCH = 60


def _chunks(items: list[Any], size: int) -> Iterator[list[Any]]:
    for start in range(0, len(items), size):
        yield items[start:start + size]


def seed_topics(topics: list[dict[str, Any]]) -> int:
    """Insert or update curriculum rows. Never deletes mastery data."""
    written = 0
    with tx() as conn:
        # Written in batches rather than one statement per topic. Against a
        # local file the difference is invisible; against a hosted database
        # every statement is a round trip to another continent, and 182 of them
        # is the difference between a boot and a wait.
        rows = [
            (
                t["slug"], t["title"], t["track"], t["week"], t["day"],
                t["difficulty"], t["minutes"], t["summary"],
                json.dumps(t["objectives"]), json.dumps(t["real_world"]),
                json.dumps(t.get("prereqs", [])), i,
                t.get("kind", "lesson"), json.dumps(t.get("concepts", [])),
                json.dumps(t.get("maths", [])), t.get("dataset", ""),
            )
            for i, t in enumerate(topics)
        ]
        for chunk in _chunks(rows, _BATCH):
            values = ",".join(["(" + ",".join("?" * 16) + """)"""] * len(chunk))
            conn.execute(
                f"""INSERT INTO topics(slug,title,track,week,day,difficulty,minutes,
                                      summary,objectives,real_world,prereqs,sort_order,
                                      kind,concepts,maths,dataset)
                   VALUES{values}
                   ON CONFLICT(slug) DO UPDATE SET
                     title=excluded.title, track=excluded.track, week=excluded.week,
                     day=excluded.day, difficulty=excluded.difficulty,
                     minutes=excluded.minutes, summary=excluded.summary,
                     objectives=excluded.objectives, real_world=excluded.real_world,
                     prereqs=excluded.prereqs, sort_order=excluded.sort_order,
                     kind=excluded.kind, concepts=excluded.concepts,
                     maths=excluded.maths, dataset=excluded.dataset""",
                [v for row in chunk for v in row],
            )
            written += len(chunk)

        slugs = [t["slug"] for t in topics]
        for chunk in _chunks(slugs, _BATCH):
            marks = ",".join(["(?)"] * len(chunk))
            conn.execute(
                f"INSERT OR IGNORE INTO mastery(topic_slug) VALUES{marks}", chunk)

        # Drop topics the curriculum no longer contains. Without this, changing
        # the plan leaves the old topics behind and every count — mastered,
        # percent complete, the week strip — is computed over a mixture of two
        # curricula. Mastery, attempts and submissions for a removed topic go
        # with it, because they refer to something that no longer exists.
        keep = [t["slug"] for t in topics]
        marks = ",".join("?" * len(keep))
        stale = [r["slug"] for r in conn.execute(
            f"SELECT slug FROM topics WHERE slug NOT IN ({marks})", keep)]
        for slug in stale:
            conn.execute("DELETE FROM attempts WHERE topic_slug=?", (slug,))
            conn.execute("DELETE FROM sessions WHERE topic_slug=?", (slug,))
            conn.execute("DELETE FROM reviews WHERE topic_slug=?", (slug,))
            conn.execute("DELETE FROM messages WHERE topic_slug=?", (slug,))
            conn.execute("DELETE FROM content WHERE topic_slug=?", (slug,))
            conn.execute("DELETE FROM project_work WHERE topic_slug=?", (slug,))
            conn.execute("DELETE FROM mastery WHERE topic_slug=?", (slug,))
            conn.execute("DELETE FROM topics WHERE slug=?", (slug,))
    return written


def get_topic(slug: str) -> dict[str, Any] | None:
    row = query_one("SELECT * FROM topics WHERE slug=?", (slug,))
    return _topic_row_to_dict(row) if row else None


def all_topics() -> list[dict[str, Any]]:
    return [_topic_row_to_dict(r) for r in query("SELECT * FROM topics ORDER BY sort_order")]


def _topic_row_to_dict(row: dict[str, Any]) -> dict[str, Any]:
    d = dict(row)
    for field_name in ("objectives", "real_world", "prereqs", "concepts",
                       "maths"):
        raw = d.get(field_name)
        d[field_name] = json.loads(raw) if raw else []
    d.setdefault("kind", "lesson")
    d.setdefault("dataset", "")
    return d


def topics_with_mastery() -> list[dict[str, Any]]:
    rows = query(
        """SELECT t.*, m.score, m.attempts, m.correct, m.seconds_spent,
                  m.last_seen_at, m.completed_at
           FROM topics t LEFT JOIN mastery m ON m.topic_slug = t.slug
           ORDER BY t.sort_order"""
    )
    out = []
    for r in rows:
        d = _topic_row_to_dict(r)
        # The join already carries every mastery column. Callers used to ask
        # again per topic, which cost one query each — 91 topics, four asks
        # apiece, and on a hosted database every one of those is a round trip
        # to another continent. They are carried through instead.
        d["score"] = r["score"] or 0.0
        d["attempts"] = r["attempts"] or 0
        d["correct"] = r["correct"] or 0
        d["seconds_spent"] = r["seconds_spent"] or 0
        d["completed_at"] = r["completed_at"]
        d["last_seen_at"] = r["last_seen_at"]
        out.append(d)
    return out


def all_mastery() -> dict[str, dict[str, Any]]:
    """Every topic's mastery row, keyed by slug, in one query.

    For anything that walks the whole plan. Asking per topic is the same
    information at ninety-one times the cost, which is invisible against a
    local file and crippling against a hosted one.
    """
    rows = query("SELECT * FROM mastery")
    return {r["topic_slug"]: dict(r) for r in rows}


# ---------------------------------------------------------------------------
# Sessions and attempts
# ---------------------------------------------------------------------------

def start_session(topic_slug: str | None, mode: str) -> int:
    return execute(
        "INSERT INTO sessions(topic_slug, mode, started_at) VALUES(?,?,?)",
        (topic_slug, mode, now()),
    )


def end_session(session_id: int, seconds: int, notes: str | None = None) -> None:
    execute(
        "UPDATE sessions SET ended_at=?, seconds=?, notes=? WHERE id=?",
        (now(), seconds, notes, session_id),
    )


def log_attempt(
    *,
    topic_slug: str,
    kind: str,
    correct: bool,
    score: float = 0.0,
    session_id: int | None = None,
    item_id: str | None = None,
    seconds: int = 0,
    hints_used: int = 0,
    payload: dict[str, Any] | None = None,
) -> int:
    return execute(
        """INSERT INTO attempts(session_id, topic_slug, item_id, kind, correct,
                                score, seconds, hints_used, payload, created_at)
           VALUES(?,?,?,?,?,?,?,?,?,?)""",
        (
            session_id, topic_slug, item_id, kind, int(correct), float(score),
            int(seconds), int(hints_used),
            json.dumps(payload or {}), now(),
        ),
    )


def recent_attempts(limit: int = 200) -> list[dict[str, Any]]:
    rows = query(
        "SELECT * FROM attempts ORDER BY created_at DESC LIMIT ?", (limit,)
    )
    return [dict(r) for r in rows]


def attempts_since(days: int) -> list[dict[str, Any]]:
    cutoff = (datetime.now() - timedelta(days=days)).isoformat(timespec="seconds")
    return [dict(r) for r in query(
        "SELECT * FROM attempts WHERE created_at >= ? ORDER BY created_at", (cutoff,)
    )]


# ---------------------------------------------------------------------------
# Content cache
# ---------------------------------------------------------------------------

def save_content(
    content_id: str, topic_slug: str, kind: str, difficulty: str,
    body: dict[str, Any], provider: str | None = None, model: str | None = None,
    replace: bool = False,
) -> None:
    """Store a generated item.

    Most ids are a hash of the body, so a conflict means the identical item was
    generated again and there is nothing to write — that is the de-duplication
    that stops the same question being paid for twice.

    A lesson is different: its id is fixed per topic, so regenerating one always
    conflicts. This used to be `ON CONFLICT DO NOTHING`, which meant "Generate a
    fresh explanation" retired the old lesson, threw the new one away, and left
    the topic with no lesson at all and no way to get one back — the row stayed
    retired and every retry hit the same conflict. Callers deliberately
    replacing an item pass replace=True, which overwrites the body and brings
    the row back out of retirement.
    """
    flag = 1 if replace else 0
    execute(
        """INSERT INTO content(id, topic_slug, kind, difficulty, body, provider,
                               model, created_at)
           VALUES(?,?,?,?,?,?,?,?)
           ON CONFLICT(id) DO UPDATE SET
             body       = CASE WHEN ? THEN excluded.body       ELSE content.body END,
             difficulty = CASE WHEN ? THEN excluded.difficulty ELSE content.difficulty END,
             provider   = CASE WHEN ? THEN excluded.provider   ELSE content.provider END,
             model      = CASE WHEN ? THEN excluded.model      ELSE content.model END,
             retired    = CASE WHEN ? THEN 0                   ELSE content.retired END""",
        (content_id, topic_slug, kind, difficulty, json.dumps(body),
         provider, model, now(), flag, flag, flag, flag, flag),
    )


def get_content(content_id: str) -> dict[str, Any] | None:
    row = query_one("SELECT * FROM content WHERE id=?", (content_id,))
    if not row:
        return None
    d = dict(row)
    d["body"] = json.loads(d["body"])
    return d


def list_content(
    topic_slug: str, kind: str, difficulty: str | None = None,
    unseen_only: bool = False, limit: int = 50,
) -> list[dict[str, Any]]:
    sql = "SELECT * FROM content WHERE topic_slug=? AND kind=? AND retired=0"
    params: list[Any] = [topic_slug, kind]
    if difficulty:
        sql += " AND difficulty=?"
        params.append(difficulty)
    if unseen_only:
        sql += " AND times_shown=0"
    sql += " ORDER BY times_shown ASC, created_at ASC LIMIT ?"
    params.append(limit)
    out = []
    for r in query(sql, params):
        d = dict(r)
        d["body"] = json.loads(d["body"])
        out.append(d)
    return out


def latest_lesson(topic_slug: str) -> dict[str, Any] | None:
    """The lesson to show for a topic: the most recent one, explicitly.

    list_content orders by times_shown then created_at, which is right for
    handing out unseen questions and wrong for this. A topic should only ever
    have one lesson row, but "should" is not "does" — an id scheme that
    changed, or a save that half-happened, leaves two, and the oldest then
    wins forever. A freshly written lesson silently losing to the one it was
    meant to replace is indistinguishable from the button not working.
    """
    row = query_one(
        "SELECT * FROM content WHERE topic_slug=? AND kind='lesson' "
        "AND retired=0 ORDER BY created_at DESC, rowid DESC LIMIT 1",
        (topic_slug,))
    if not row:
        return None
    out = dict(row)
    out["body"] = json.loads(out["body"])
    return out


def retire_other_lessons(topic_slug: str, keep_id: str) -> int:
    """Retire every lesson for a topic except this one, so there is just one."""
    rows = query("SELECT id FROM content WHERE topic_slug=? AND kind='lesson' "
                 "AND retired=0 AND id != ?", (topic_slug, keep_id))
    for row in rows:
        retire_content(row["id"])
    return len(rows)


def mark_shown(content_id: str) -> None:
    execute("UPDATE content SET times_shown = times_shown + 1 WHERE id=?", (content_id,))


def retire_content(content_id: str) -> None:
    execute("UPDATE content SET retired=1 WHERE id=?", (content_id,))


def retire_unteachable_content(kinds: tuple[str, ...] =
                               ("lesson", "mcq", "card", "experiment")) -> int:
    """Retire stored material whose code reaches past the day it belongs to.

    Anything generated before the horizon existed was written without the
    constraint, and a week-2 lesson that cross-validates is not repairable in
    place — the whole explanation was built around a tool the reader has not
    met. Retiring it means the next visit regenerates it under the constraint.

    Submissions are kept: a retired item still counts in the history.
    """
    from . import concepts                   # local: concepts imports curriculum

    days = {t["slug"]: t["day"] for t in all_topics()}
    marks = ",".join("?" * len(kinds))
    retired = 0
    for row in query(f"SELECT id, topic_slug, body FROM content "
                     f"WHERE kind IN ({marks}) AND retired=0 AND source!='bank'",
                     list(kinds)):
        day = days.get(row["topic_slug"])
        if day is None:
            continue
        body = json.loads(row["body"]) if isinstance(row["body"], str) else row["body"]
        if isinstance(body, dict) and concepts.reaches_ahead(body, day):
            retire_content(row["id"])
            retired += 1
    return retired


def retire_unverified_lessons() -> int:
    """Retire lessons whose code was never executed successfully.

    The promise this app makes is that every line of code in a lesson has been
    run against the real dataset before the reader sees it. A lesson stored
    without that stamp predates the promise, or failed and was written anyway;
    either way it must not be shown.
    """
    rows = query("SELECT id FROM content WHERE kind='lesson' AND retired=0 "
                 "AND verified!='passed'")
    for row in rows:
        retire_content(row["id"])
    return len(rows)


def mark_verified(content_id: str, status: str) -> None:
    """Record the outcome of running an item's code. 'passed' or a reason."""
    execute("UPDATE content SET verified=? WHERE id=?", (status, content_id))


def unverified(kind: str = "lesson", limit: int = 50) -> list[dict[str, Any]]:
    return query("SELECT id, topic_slug, verified FROM content "
                 "WHERE kind=? AND retired=0 AND verified='' LIMIT ?",
                 (kind, limit))


# ---------------------------------------------------------------------------
# The Math Helper's cache
# ---------------------------------------------------------------------------

def save_math(cache_id: str, term: str, day: int, body: dict[str, Any],
              provider: str | None = None, model: str | None = None) -> None:
    execute(
        """INSERT INTO math_cache(id, term, day, body, provider, model, created_at)
           VALUES(?,?,?,?,?,?,?)
           ON CONFLICT(id) DO UPDATE SET body=excluded.body,
                                         provider=excluded.provider,
                                         model=excluded.model""",
        (cache_id, term, int(day), json.dumps(body), provider, model, now()),
    )


def get_math(cache_id: str) -> dict[str, Any] | None:
    row = query_one("SELECT * FROM math_cache WHERE id=?", (cache_id,))
    if not row:
        return None
    execute("UPDATE math_cache SET times_shown = times_shown + 1 WHERE id=?",
            (cache_id,))
    out = dict(row)
    out["body"] = json.loads(out["body"])
    return out


def math_history(limit: int = 40) -> list[dict[str, Any]]:
    """What has been asked, most asked first.

    Worth showing on its own: a term looked up five times is a term the course
    has not explained well enough, and that is a fact about the material rather
    than about the reader.
    """
    rows = query("SELECT term, day, times_shown, created_at FROM math_cache "
                 "ORDER BY times_shown DESC, created_at DESC LIMIT ?", (limit,))
    return [dict(r) for r in rows]


def seed_fingerprint(*parts: Any) -> str:
    """A short digest of the material that seeding writes.

    Seeding is idempotent, so running it twice is harmless — but it is 485
    statements, and against a hosted database that is 485 round trips to
    another continent on every single visit. Storing this digest lets a boot
    that would change nothing skip the work entirely, while a real curriculum
    edit still changes the digest and re-seeds automatically.
    """
    digest = hashlib.sha256()
    for part in parts:
        digest.update(json.dumps(part, sort_keys=True, default=str).encode())
    return digest.hexdigest()[:16]


def seeded_with() -> str:
    """The fingerprint of whatever is currently seeded, or empty."""
    row = query_one("SELECT value FROM settings WHERE key='seed_fingerprint'")
    return row["value"] if row else ""


def mark_seeded(fingerprint: str) -> None:
    set_setting("seed_fingerprint", fingerprint)


def content_count(topic_slug: str, kind: str) -> int:
    row = query_one(
        "SELECT COUNT(*) AS n FROM content WHERE topic_slug=? AND kind=? AND retired=0",
        (topic_slug, kind),
    )
    return row["n"] if row else 0


# ---------------------------------------------------------------------------
# Submissions
# ---------------------------------------------------------------------------

def save_submission(
    *, content_id: str | None, topic_slug: str, code: str, verdict: str,
    attempt_no: int | None = None, passed: int = 0, total: int = 1,
    failures: list[Any] | None = None, stderr: str | None = None,
    review: dict[str, Any] | None = None, seconds: int = 0,
) -> int:
    """Record what was written and how it went.

    attempt_no is counted here rather than by the caller. Every page that
    saved a submission had to work it out, and a page that got it wrong
    produced a history where the third attempt claimed to be the first.
    """
    if attempt_no is None:
        row = query_one(
            "SELECT COUNT(*) AS n FROM submissions WHERE content_id IS ? "
            "AND topic_slug=?", (content_id, topic_slug))
        attempt_no = int((row or {}).get("n") or 0) + 1
    return execute(
        """INSERT INTO submissions(content_id, topic_slug, attempt_no, code, verdict,
                                   passed, total, failures, stderr, review, seconds,
                                   created_at)
           VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
        (content_id, topic_slug, attempt_no, code, verdict, passed, total,
         json.dumps(failures or []), stderr, json.dumps(review) if review else None,
         seconds, now()),
    )


def submissions_for(content_id: str) -> list[dict[str, Any]]:
    rows = query(
        "SELECT * FROM submissions WHERE content_id=? ORDER BY attempt_no", (content_id,)
    )
    out = []
    for r in rows:
        d = dict(r)
        d["failures"] = json.loads(d["failures"] or "[]")
        d["review"] = json.loads(d["review"]) if d["review"] else None
        out.append(d)
    return out


def recent_mistakes(limit: int = 20) -> list[dict[str, Any]]:
    """Failed code submissions, newest first. Feeds the 'where I went wrong' view
    and gives the scheduler evidence beyond MCQ scores."""
    rows = query(
        """SELECT s.*, t.title FROM submissions s
           JOIN topics t ON t.slug = s.topic_slug
           WHERE s.passed < s.total OR s.verdict != 'pass'
           ORDER BY s.created_at DESC LIMIT ?""",
        (limit,),
    )
    out = []
    for r in rows:
        d = dict(r)
        d["failures"] = json.loads(d["failures"] or "[]")
        d["review"] = json.loads(d["review"]) if d["review"] else None
        out.append(d)
    return out


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------

def log_message(role: str, content: str, topic_slug: str | None = None) -> None:
    execute(
        "INSERT INTO messages(topic_slug, role, content, created_at) VALUES(?,?,?,?)",
        (topic_slug, role, content, now()),
    )


def recent_messages(topic_slug: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
    if topic_slug:
        rows = query(
            "SELECT * FROM messages WHERE topic_slug=? ORDER BY id DESC LIMIT ?",
            (topic_slug, limit),
        )
    else:
        rows = query("SELECT * FROM messages ORDER BY id DESC LIMIT ?", (limit,))
    return [dict(r) for r in reversed(rows)]


def reset_all(keep_settings: bool = True) -> None:
    """Wipe learning data. Destructive; the Settings page requires a typed
    confirmation before calling this."""
    tables = ["attempts", "sessions", "submissions", "reviews", "messages",
              "daily_stats", "content", "mastery", "project_work", "xp_events",
              "bookmarks"]
    # math_cache survives deliberately. Wiping progress is a decision about
    # your own record; the glossary answers are not progress, they cost money
    # to fetch, and nobody resetting a course wants to pay for "what is sigma"
    # a second time.
    with tx() as conn:
        for table in tables:
            conn.execute(f"DELETE FROM {table}")
        if not keep_settings:
            conn.execute("DELETE FROM settings")
        conn.execute("INSERT OR IGNORE INTO mastery(topic_slug) SELECT slug FROM topics")


# ---------------------------------------------------------------------------
# The curated challenge bank
# ---------------------------------------------------------------------------
# The Python mentor's bank held 150 interview problems, each a function with
# hidden test cases. Machine learning does not decompose that way: the skill
# being tested is rarely "write this function" and almost always "do the right
# thing to this data, and notice what it tells you".
#
# So a challenge here is a task against one of the bundled datasets, and it is
# graded by running the learner's code and then running a hidden check against
# the variables it left behind. Fit a model and leave the test AUC in a
# variable called `auc`; the check asserts that `auc` exists, that it is in a
# plausible range, and — this is the part a test case cannot do — that it was
# not obtained by leaving the leaking column in.

def seed_challenge_bank(entries: list[dict[str, Any]]) -> int:
    """Load the hand-written challenges into the content table.

    They live alongside generated material so every consumer works unchanged;
    `source` distinguishes them, and re-seeding updates in place rather than
    duplicating, so an improved task reaches an existing installation.
    """
    written = 0
    with tx() as conn:
        # One query to learn which topic sits on which day, rather than one
        # per challenge. On a hosted database that lookup alone was the
        # difference between a boot and a wait.
        by_day = _topics_by_day(conn)
        fallback = _first_topic(conn)
        stamp = now()

        rows = []
        for entry in entries:
            # A challenge is credited to the day it becomes attemptable, which
            # the entry works out from its own code. Filing it under the day
            # its skill is first mentioned would offer week-2 readers a task
            # that fits a model they meet in week 5.
            topic_slug = (entry.get("topic_slug")
                          or _topic_on_or_after(by_day, entry.get("unlock_day", 1))
                          or fallback)
            if topic_slug is None:
                continue
            rows.append((entry["id"], topic_slug, "challenge",
                         entry.get("difficulty", "intermediate"),
                         json.dumps(entry), "bank", "curated", stamp, "bank",
                         "passed"))

        for chunk in _chunks(rows, _BATCH):
            values = ",".join(["(" + ",".join("?" * 10) + ")"] * len(chunk))
            conn.execute(
                f"""INSERT INTO content(id, topic_slug, kind, difficulty, body,
                                       provider, model, created_at, source,
                                       verified)
                   VALUES{values}
                   ON CONFLICT(id) DO UPDATE SET
                     body=excluded.body, difficulty=excluded.difficulty,
                     topic_slug=excluded.topic_slug, retired=0""",
                [v for row in chunk for v in row],
            )
            written += len(chunk)
    return written


def _topics_by_day(conn: dbdriver.Connection) -> dict[int, str]:
    """Day number -> topic slug, in one pass."""
    return {int(row["day"]): row["slug"]
            for row in conn.execute("SELECT slug, day FROM topics")}


def _topic_on_or_after(by_day: dict[int, str], day: int) -> str | None:
    """The topic on that day, or the next one that exists.

    A challenge whose unlock day is 0 — its code could never be legal — is
    refused rather than filed somewhere harmless, because a challenge nobody
    can ever attempt is a bug wearing a row in the database.
    """
    day = int(day or 1)
    if day <= 0:
        return None
    for candidate in range(day, max(by_day, default=day) + 1):
        if candidate in by_day:
            return by_day[candidate]
    return None


def _first_topic(conn: dbdriver.Connection) -> str | None:
    """Where a challenge goes when no topic claims its skill."""
    row = conn.execute(
        "SELECT slug FROM topics ORDER BY sort_order LIMIT 1").fetchone()
    return row["slug"] if row else None


def bank_challenges(skill: str | None = None, difficulty: str | None = None,
                    solved: bool | None = None, max_day: int | None = None,
                    limit: int = 500) -> list[dict[str, Any]]:
    """Curated challenges, optionally filtered to what has been taught.

    `max_day` is the one that matters in practice: a challenge is only offered
    once its lesson has been reached, so the bank grows as the course does
    rather than presenting ninety locked rows on day one.
    """
    sql = ["""SELECT c.*, t.day AS topic_day FROM content c
              JOIN topics t ON t.slug = c.topic_slug
              WHERE c.source='bank' AND c.kind='challenge' AND c.retired=0"""]
    params: list[Any] = []
    if difficulty:
        sql.append("AND c.difficulty=?")
        params.append(difficulty)
    if max_day is not None:
        sql.append("AND t.day <= ?")
        params.append(int(max_day))
    sql.append("ORDER BY t.sort_order LIMIT ?")
    params.append(limit)

    solved_ids = _solved_ids()
    out = []
    for row in query(" ".join(sql), params):
        body = json.loads(row["body"])
        if skill and body.get("skill") != skill:
            continue
        body["topic_slug"] = row["topic_slug"]
        body["topic_day"] = row["topic_day"]
        body["solved"] = row["id"] in solved_ids
        if solved is None or body["solved"] == solved:
            out.append(body)
    return out


def _solved_ids() -> set[str]:
    """Every challenge id with a passing submission, in one query.

    Asking per challenge was the N+1 that made the old bank page slow: ninety
    rows meant ninety round trips to answer a question one query answers.
    """
    return {r["content_id"] for r in query(
        "SELECT DISTINCT content_id FROM submissions WHERE verdict='pass' "
        "AND content_id IS NOT NULL")}


def is_solved(content_id: str) -> bool:
    row = query_one(
        "SELECT 1 FROM submissions WHERE content_id=? AND verdict='pass' LIMIT 1",
        (content_id,),
    )
    return row is not None


def bank_progress() -> dict[str, Any]:
    """Solved counts overall, by skill and by difficulty.

    Everything is derived from the submissions table, so it cannot drift from
    what actually happened.
    """
    total = query_one(
        "SELECT COUNT(*) AS n FROM content "
        "WHERE source='bank' AND kind='challenge' AND retired=0")["n"]

    by_skill: dict[str, int] = {}
    by_difficulty: dict[str, int] = {}
    totals: dict[str, int] = {}
    solved = 0

    # One pass over the bank, joined to its submissions, rather than a query
    # per skill. The bodies are already being read; the counts come free.
    solved_ids = _solved_ids()
    for row in query("SELECT id, difficulty, body FROM content "
                     "WHERE source='bank' AND kind='challenge' AND retired=0"):
        body = json.loads(row["body"])
        skill = body.get("skill", "other")
        totals[skill] = totals.get(skill, 0) + 1
        if row["id"] in solved_ids:
            solved += 1
            by_skill[skill] = by_skill.get(skill, 0) + 1
            by_difficulty[row["difficulty"]] = by_difficulty.get(
                row["difficulty"], 0) + 1

    return {"total": total, "solved": solved, "by_skill": by_skill,
            "skill_totals": totals, "by_difficulty": by_difficulty}


# ---------------------------------------------------------------------------

def get_project(topic_slug: str) -> dict[str, Any]:
    row = query_one("SELECT * FROM project_work WHERE topic_slug=?", (topic_slug,))
    if row is None:
        return {"topic_slug": topic_slug, "status": "not_started", "notes": "",
                "code": "", "checklist": [], "started_at": None,
                "finished_at": None}
    d = dict(row)
    d["checklist"] = json.loads(d["checklist"] or "[]")
    return d


def save_project(topic_slug: str, *, status: str | None = None,
                 notes: str | None = None, code: str | None = None,
                 checklist: list[Any] | None = None) -> None:
    existing = get_project(topic_slug)
    status = status or existing["status"]
    started = existing["started_at"] or (now() if status != "not_started" else None)
    finished = now() if status == "done" else existing["finished_at"]
    execute(
        """INSERT INTO project_work(topic_slug, status, notes, code, checklist,
                                    started_at, finished_at, updated_at)
           VALUES(?,?,?,?,?,?,?,?)
           ON CONFLICT(topic_slug) DO UPDATE SET
             status=excluded.status, notes=excluded.notes, code=excluded.code,
             checklist=excluded.checklist, started_at=excluded.started_at,
             finished_at=excluded.finished_at, updated_at=excluded.updated_at""",
        (topic_slug, status,
         existing["notes"] if notes is None else notes,
         existing["code"] if code is None else code,
         json.dumps(existing["checklist"] if checklist is None else checklist),
         started, finished, now()),
    )


def project_statuses() -> dict[str, str]:
    return {r["topic_slug"]: r["status"] for r in
            query("SELECT topic_slug, status FROM project_work")}


# ---------------------------------------------------------------------------
# XP
# ---------------------------------------------------------------------------

def award_xp(kind: str, amount: int, reason: str = "",
             topic_slug: str | None = None, dedupe_key: str | None = None) -> bool:
    """Record an XP event. Returns True if it was new.

    `dedupe_key` makes an award idempotent: solving the same challenge twice, or
    a page rerunning, must not pay twice. Streamlit reruns a script on every
    interaction, so without this the number would inflate constantly.
    """
    try:
        with tx() as conn:
            conn.execute(
                """INSERT INTO xp_events(kind, amount, reason, topic_slug,
                                         dedupe_key, created_at)
                   VALUES(?,?,?,?,?,?)""",
                (kind, int(amount), reason, topic_slug, dedupe_key, now()),
            )
        return True
    except Exception as exc:  # noqa: BLE001 - narrowed on the next line
        if not dbdriver.is_unique_violation(exc):
            raise
        return False  # already awarded


def total_xp() -> int:
    row = query_one("SELECT COALESCE(SUM(amount), 0) AS total FROM xp_events")
    return row["total"] if row else 0


def recent_xp(limit: int = 20) -> list[dict[str, Any]]:
    return [dict(r) for r in query(
        "SELECT * FROM xp_events ORDER BY id DESC LIMIT ?", (limit,))]


def xp_by_day(days: int = 30) -> dict[str, int]:
    cutoff = (datetime.now() - timedelta(days=days)).isoformat(timespec="seconds")
    rows = query(
        "SELECT substr(created_at, 1, 10) AS d, SUM(amount) AS total "
        "FROM xp_events WHERE created_at >= ? GROUP BY d", (cutoff,))
    return {r["d"]: r["total"] for r in rows}


# ---------------------------------------------------------------------------
# Bookmarks
# ---------------------------------------------------------------------------

def toggle_bookmark(ref: str, kind: str, label: str = "") -> bool:
    """Returns True if the item is now bookmarked."""
    if query_one("SELECT 1 FROM bookmarks WHERE ref=?", (ref,)):
        execute("DELETE FROM bookmarks WHERE ref=?", (ref,))
        return False
    execute("INSERT INTO bookmarks(ref, kind, label, created_at) VALUES(?,?,?,?)",
            (ref, kind, label, now()))
    return True


def is_bookmarked(ref: str) -> bool:
    return query_one("SELECT 1 FROM bookmarks WHERE ref=?", (ref,)) is not None


def bookmarks(kind: str | None = None) -> list[dict[str, Any]]:
    if kind:
        rows = query("SELECT * FROM bookmarks WHERE kind=? ORDER BY created_at DESC",
                     (kind,))
    else:
        rows = query("SELECT * FROM bookmarks ORDER BY created_at DESC")
    return [dict(r) for r in rows]


def set_note(ref: str, note: str, kind: str = "note", label: str = "") -> None:
    execute(
        """INSERT INTO bookmarks(ref, kind, label, note, created_at)
           VALUES(?,?,?,?,?)
           ON CONFLICT(ref) DO UPDATE SET note=excluded.note""",
        (ref, kind, label, note, now()),
    )
