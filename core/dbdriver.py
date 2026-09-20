"""Where the database actually lives.

The app has two homes now. On the laptop it is a SQLite file next to the code,
which is fast, needs no account and works with no network. Hosted, it is a
libSQL database reached over the internet, because a hosted container wipes its
own disk every time it restarts — a SQLite file there would silently take the
study record with it.

libSQL is SQLite's engine with a server in front, so every statement in db.py
works unchanged on both. That is the whole reason for choosing it over
PostgreSQL: the alternative was rewriting a hundred queries, and a rewrite of
that size is a rewrite of a hundred chances to change behaviour by accident.

What this module smooths over is the smaller differences:

  * the libSQL driver has no row_factory, so both backends are normalised to
    plain dicts — which every call site already accepted, since sqlite3.Row
    supports the same ["column"] access;
  * the two raise different exceptions for "that column already exists", which
    the additive migrations rely on catching;
  * PRAGMA journal_mode and a local file path mean nothing to a remote server.

Selecting a backend: set TURSO_DATABASE_URL (and TURSO_AUTH_TOKEN) and the
hosted database is used. Leave them unset and it is the local file. Nothing
else in the app knows which one it got.
"""
from __future__ import annotations

import os
import re
import sqlite3
import tempfile
from pathlib import Path
from typing import Any, Iterable, Sequence

# Errors that mean "this migration was already applied". Caught rather than
# inspected, because the two drivers word them differently and the statements
# are additive, so a no-op is the expected outcome most of the time.
try:                                     # only installed where it is needed
    import libsql                        # type: ignore
except Exception:                        # noqa: BLE001 - absence is normal
    libsql = None                        # type: ignore


# The two drivers report the same conditions as different exception types —
# libSQL raises a plain ValueError where sqlite3 raises OperationalError — so
# they are told apart by what the message says rather than by class. Matching
# on text is unpleasant, but the alternative is catching everything, and a
# migration that fails for a real reason would then pass unnoticed.

# These are predicates rather than exception classes on purpose. A class with a
# custom __instancecheck__ looks like it would let `except ALREADY_APPLIED`
# work, but Python's except clause compares classes directly and never calls
# __instancecheck__, so such a class silently matches nothing. The caller
# catches Exception and asks one of these.

_ALREADY_APPLIED_TEXT = ("duplicate column", "already exists")
_UNIQUE_TEXT = ("unique constraint", "constraint_unique", "unique index")


def is_already_applied(exc: BaseException) -> bool:
    """True when a migration failed only because it had already been run."""
    return any(p in str(exc).lower() for p in _ALREADY_APPLIED_TEXT)


def is_unique_violation(exc: BaseException) -> bool:
    """True when an insert clashed with an existing row on a UNIQUE column.

    The XP ledger relies on this to make an award happen once: it inserts with
    a dedupe key and reads the clash as "already given", so a page rerun cannot
    pay twice.
    """
    return any(p in str(exc).lower() for p in _UNIQUE_TEXT)


class Cursor:
    """A cursor whose rows are dicts, whichever driver produced them."""

    def __init__(self, cursor: Any):
        self._cursor = cursor

    def _column_names(self) -> list[str]:
        """The column names, read when the rows are, not before.

        This was read once in __init__, immediately after execute. The remote
        libSQL driver does not have the names ready at that moment — it fills
        `description` in once the result actually arrives — so every row became
        dict(zip([], values)), an empty dict, and the first caller to ask for a
        column died with a KeyError a long way from the cause.

        Names are also normalised, for two reasons seen in the wild:

        A driver may qualify the name as "settings.key", so only the column
        part is kept.

        And the hosted engine returns a column whose name is an SQL reserved
        word in capitals — `SELECT key, value FROM settings` comes back
        describing them as "KEY" and "value". That single capital letter is
        what made `row["key"]` raise KeyError on the deployed app while every
        other column worked and the local file was fine. Lowercasing settles
        it: every identifier and alias in this codebase is lowercase, and
        SQLite treats names case-insensitively anyway, so this makes the two
        backends agree exactly rather than almost.
        """
        names = []
        for entry in (getattr(self._cursor, "description", None) or ()):
            name = entry[0] if isinstance(entry, (tuple, list)) else str(entry)
            names.append(str(name).rsplit(".", 1)[-1].lower())
        return names

    def _row(self, raw: Any) -> dict[str, Any]:
        if raw is None:
            return {}
        # Every path lowercases its keys, so a row means the same thing
        # whichever driver produced it. See _column_names for why.
        if isinstance(raw, dict):
            return {str(k).lower(): v for k, v in raw.items()}
        if isinstance(raw, sqlite3.Row):
            return {str(k).lower(): raw[k] for k in raw.keys()}
        if hasattr(raw, "keys"):                 # any other mapping-like row
            return {str(k).lower(): raw[k] for k in raw.keys()}

        columns = self._column_names()
        if not columns:
            # Returning {} here is what produced the original KeyError. A row
            # with no names attached is a driver fault, and saying so beats
            # handing back something that looks like an empty result.
            raise RuntimeError(
                "The database driver returned a row without column names. "
                f"Row had {len(raw)} value(s). This is a driver problem, not "
                "a query problem."
            )
        return dict(zip(columns, raw))

    def fetchall(self) -> list[dict[str, Any]]:
        return [self._row(r) for r in self._cursor.fetchall()]

    def __iter__(self):
        """sqlite3 cursors iterate directly, and some call sites do.

        Kept so `for row in conn.execute(...)` keeps working, rather than
        making every seeding loop say .fetchall() for the benefit of a driver
        it should not have to know about.
        """
        return iter(self.fetchall())

    def fetchone(self) -> dict[str, Any] | None:
        raw = self._cursor.fetchone()
        return self._row(raw) if raw is not None else None

    @property
    def lastrowid(self) -> int | None:
        return getattr(self._cursor, "lastrowid", None)

    @property
    def rowcount(self) -> int:
        return getattr(self._cursor, "rowcount", -1)


class Connection:
    """The subset of the DB-API that this app uses, over either driver."""

    def __init__(self, raw: Any, *, remote: bool, replica: bool = False):
        self._raw = raw
        self.remote = remote
        # True when a hosted database is being read through a local synced
        # copy rather than over the wire for every statement.
        self.replica = replica

    def execute(self, sql: str, params: Iterable[Any] = ()) -> Cursor:
        if isinstance(params, dict):
            sql, params = _to_positional(sql, params)
        elif not isinstance(params, (list, tuple)):
            params = tuple(params)
        return Cursor(self._raw.execute(sql, tuple(params)))

    def executescript(self, script: str) -> None:
        runner = getattr(self._raw, "executescript", None)
        if runner is not None:
            runner(script)
            return
        # The libSQL driver has no executescript, so the schema is applied one
        # statement at a time. Splitting on ";" is safe here because the schema
        # is ours and contains no semicolon inside a string or a trigger body.
        for statement in script.split(";"):
            if statement.strip():
                self._raw.execute(statement)

    def commit(self) -> None:
        self._raw.commit()

    def rollback(self) -> None:
        try:
            self._raw.rollback()
        except Exception:               # noqa: BLE001
            # A remote driver can refuse a rollback when no transaction is
            # open. Losing that is harmless; losing the original exception,
            # which is what raising here would do, is not.
            pass

    def sync(self) -> None:
        syncer = getattr(self._raw, "sync", None)
        if syncer is not None:
            syncer()

    def close(self) -> None:
        try:
            self._raw.close()
        except Exception:               # noqa: BLE001
            pass


_NAMED = re.compile(r":([A-Za-z_][A-Za-z0-9_]*)")


def _to_positional(sql: str, params: dict[str, Any]) -> tuple[str, list[Any]]:
    """Rewrite :name placeholders as ?, repeating values where a name recurs.

    The libSQL driver takes positional parameters only. Matching the whole
    identifier means a parameter called :rep can never swallow part of
    :replace, and a name used three times contributes its value three times, in
    the order the statement reads.
    """
    values: list[Any] = []

    def swap(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in params:
            return match.group(0)        # not ours; leave it alone
        values.append(params[name])
        return "?"

    return _NAMED.sub(swap, sql), values


def _replica_path() -> Path:
    """Where the local copy of a hosted database lives.

    Somewhere writable and temporary. On a hosted container this is wiped on
    every restart, which is fine — the copy is rebuilt from the server, and the
    server is what actually holds the study record.

    Named for this app specifically. The Python mentor uses the same driver
    and the same temp directory, and two apps sharing one replica file would
    each keep overwriting the other's copy with rows from a different schema.
    """
    base = Path(os.environ.get("MLMENTOR_REPLICA_DIR") or tempfile.gettempdir())
    base.mkdir(parents=True, exist_ok=True)
    return base / "mlmentor-replica.db"


def _connect_hosted(url: str, token: str) -> Connection:
    """Open a hosted database through a synced local copy.

    Two ways to reach a hosted libSQL database, and the difference decides
    whether this app works at all.

    A pure remote connection sends every statement over the wire and reads the
    answer back. On the deployed app every row came back without its column
    names attached, so `row["key"]` raised KeyError on the first query. Falling
    back to it is falling back to a guaranteed crash, so this no longer does.

    An embedded replica keeps a real SQLite file on local disk, kept in step
    with the server. Reads are answered by SQLite itself — the same engine the
    laptop uses, the same tuples, the same column names, verified locally — and
    writes go through to the server. It is both the reliable option and the
    fast one, since a page running a dozen queries no longer makes a dozen
    round trips to another continent.

    If the replica cannot be made, this raises and says why. A clear sentence
    naming the cause is worth more than a KeyError three files away.
    """
    replica = _replica_path()
    try:
        raw = libsql.connect(str(replica), sync_url=url, auth_token=token)
        sync = getattr(raw, "sync", None)
        if sync is not None:
            sync()                        # pull what the server already has
        conn = Connection(raw, remote=True, replica=True)
        # Prove rows come back named before trusting it, since that is the
        # exact thing that failed before.
        probe = conn.execute("SELECT 1 AS probe").fetchall()
        if not probe or "probe" not in probe[0]:
            raise RuntimeError(
                f"the probe query returned {probe!r}, which has no column names"
            )
        return conn
    except Exception as exc:  # noqa: BLE001 - reported, not swallowed
        _REPLICA_ERROR.append(f"{type(exc).__name__}: {exc}")

    if os.environ.get("PYDSA_ALLOW_REMOTE_ROWS") == "1":
        # An escape hatch, in case a future driver version fixes the remote
        # row format and someone wants to use it.
        return Connection(libsql.connect(url, auth_token=token),
                          remote=True, replica=False)

    raise RuntimeError(
        "Could not open the hosted database through a local synced copy, and "
        "the direct connection is not used because its rows arrive without "
        "column names.\n\nThe reason the copy failed:\n  "
        + (replica_problem() or "unknown")
        + f"\n\nDatabase: {url.split('://')[-1].split('/')[0]}"
        + f"\nLocal copy would live at: {replica}"
    )


# Kept so the Settings page can say why the fast path was not used, rather than
# leaving a silent performance cliff for someone to rediscover.
_REPLICA_ERROR: list[str] = []


def replica_problem() -> str:
    return _REPLICA_ERROR[-1] if _REPLICA_ERROR else ""


def connect(path: Path, *, url: str | None = None,
            token: str | None = None) -> Connection:
    """Open the database. A url means hosted; no url means the local file."""
    url = url or os.environ.get("TURSO_DATABASE_URL") or ""
    token = token or os.environ.get("TURSO_AUTH_TOKEN") or ""

    if url:
        if libsql is None:
            raise RuntimeError(
                "TURSO_DATABASE_URL is set but the 'libsql' package is not "
                "installed. Add libsql to requirements.txt, or unset the "
                "variable to use the local file."
            )
        return _connect_hosted(url, token)

    path.parent.mkdir(parents=True, exist_ok=True)

    # PYDSA_DB_DRIVER=libsql runs the libSQL driver against the local file.
    # That is how the test suite exercises the driver the hosted app will
    # actually use, without an account or a network: the driver is the part
    # that differs, not where the bytes live. It is never set in normal use.
    if os.environ.get("PYDSA_DB_DRIVER", "").lower() == "libsql":
        if libsql is None:
            raise RuntimeError("PYDSA_DB_DRIVER=libsql but libsql is not installed")
        return Connection(libsql.connect(str(path)), remote=False)

    raw = sqlite3.connect(path, timeout=30, check_same_thread=False)
    raw.row_factory = sqlite3.Row
    # Write-ahead logging lets a long-running page and a background job coexist.
    # It is a local-file concern and has no meaning against a server.
    raw.execute("PRAGMA journal_mode=WAL")
    raw.execute("PRAGMA foreign_keys=ON")
    raw.execute("PRAGMA busy_timeout=5000")
    return Connection(raw, remote=False)


def probe_report(conn: Connection | None = None) -> str:
    """Dump exactly what the driver returns, for when reasoning has run out.

    Written because a KeyError said a column was missing while a probe query
    said column naming worked. Both cannot be true, and the only way to settle
    it from the outside is to print the raw shapes: what the driver calls the
    columns, what type a row is, and what it actually contains.

    Every step is guarded, because this runs when something is already broken
    and a diagnostic that crashes tells nobody anything.
    """
    lines: list[str] = []

    def note(label: str, value: Any) -> None:
        text = repr(value)
        lines.append(f"{label}: {text[:300]}")

    try:
        note("libsql version", getattr(libsql, "__version__", "unknown")
             if libsql else "not installed")
        if conn is None:
            lines.append("connection: none was supplied")
            return "\n".join(lines)

        note("mode", "local synced copy" if conn.replica else
             ("direct remote" if conn.remote else "local file"))
        note("replica problem", replica_problem() or "none")

        raw = conn._raw                     # noqa: SLF001 - this is diagnostics
        for sql in ("SELECT 1 AS probe",
                    "SELECT name FROM sqlite_master WHERE type='table'",
                    "SELECT key, value FROM settings",
                    "SELECT * FROM settings"):
            lines.append("")
            lines.append(f"--- {sql}")
            try:
                cur = raw.execute(sql, ())
                rows = cur.fetchall()
                note("  description", getattr(cur, "description", None))
                note("  row count", len(rows))
                if rows:
                    note("  first row type", type(rows[0]).__name__)
                    note("  first row", rows[0])
                    note("  has keys()", hasattr(rows[0], "keys"))
                    wrapped = Cursor(cur)._row(rows[0])   # noqa: SLF001
                    note("  wrapped as", wrapped)
            except Exception as exc:        # noqa: BLE001 - report and continue
                note("  raised", f"{type(exc).__name__}: {exc}")
    except Exception as exc:                # noqa: BLE001
        lines.append(f"the diagnostic itself failed: {type(exc).__name__}: {exc}")

    return "\n".join(lines)


def describe(url: str | None = None, conn: Connection | None = None) -> str:
    """One line for the Settings page saying where the data is kept.

    It names the connection mode as well as the host, because the difference
    between a synced local copy and a statement-by-statement remote connection
    is the difference between a page that loads and one that crawls — and that
    is worth being able to see rather than guess at.
    """
    url = url if url is not None else os.environ.get("TURSO_DATABASE_URL", "")
    if not url:
        return "a file on this computer"
    host = url.split("://")[-1].split("/")[0]
    if conn is not None and getattr(conn, "replica", False):
        return f"a hosted database ({host}), read through a fast local copy"
    problem = replica_problem()
    if problem:
        return (f"a hosted database ({host}), queried one statement at a time "
                f"— the fast local copy could not be created: {problem[:120]}")
    return f"a hosted database ({host})"
