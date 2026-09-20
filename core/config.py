"""Configuration and provider registry.

Keys are read from (in priority order): the settings table in the database,
then environment variables, then a .env file sitting next to app.py.

Nothing here talks to the network. See core/llm.py for that.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = APP_ROOT / "data" / "mlmentor.db"
REPORTS_DIR = APP_ROOT / "data" / "reports"

# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------
# Both providers speak the OpenAI chat-completions protocol, so one client
# class serves both; only base_url, key and model id change.
#
# Model ids go stale as providers retire checkpoints. The Settings page has a
# "Refresh model list" button that calls GET /models on the live endpoint, so a
# retired default here is a one-click fix and never a code edit.


@dataclass(frozen=True)
class Provider:
    key_name: str          # environment variable / settings key holding the API key
    label: str
    base_url: str
    signup_url: str
    # "fast" runs the high-volume work (MCQs, cards, hints, chat).
    # "strong" runs code review, project briefs and exams.
    default_fast_model: str
    default_strong_model: str
    # Some endpoints reject response_format=json_object. When False we fall
    # back to prompt-level JSON coercion plus the extractor in llm.py.
    supports_json_mode: bool = True


PROVIDERS: dict[str, Provider] = {
    "groq": Provider(
        key_name="GROQ_API_KEY",
        label="Groq",
        base_url="https://api.groq.com/openai/v1",
        signup_url="https://console.groq.com/keys",
        default_fast_model="llama-3.3-70b-versatile",
        default_strong_model="openai/gpt-oss-120b",
        supports_json_mode=True,
    ),
    "nvidia": Provider(
        key_name="NVIDIA_API_KEY",
        label="NVIDIA Build (NIM)",
        base_url="https://integrate.api.nvidia.com/v1",
        signup_url="https://build.nvidia.com/",
        default_fast_model="meta/llama-3.1-70b-instruct",
        default_strong_model="qwen/qwen3-next-80b-a3b-instruct",
        supports_json_mode=False,
    ),
}

PROVIDER_ORDER = ["groq", "nvidia"]  # failover order; overridable in Settings


# ---------------------------------------------------------------------------
# Learning parameters
# ---------------------------------------------------------------------------

@dataclass
class LearningConfig:
    """Knobs that change how hard the app pushes.

    Named constants rather than magic numbers buried in the scheduler, so the
    difference between "this course is demanding" and "this course is
    punishing" is one file rather than a search."""

    plan_weeks: int = 13
    target_minutes_per_day: int = 150          # 2.5 h/day -> ~263 h total

    mastery_threshold: float = 0.80
    struggling_threshold: float = 0.55
    mastery_ewma_alpha: float = 0.30
    min_attempts_for_mastery: int = 6

    # Session composition. Fewer MCQs than the Python course and more code:
    # machine learning is learned by running things on data and reading what
    # came back, not by recalling definitions.
    mcqs_per_session: int = 6
    cards_per_session: int = 8
    experiments_per_session: int = 3

    # Running a generated cell against a real dataset. Longer than the Python
    # sandbox because fitting a forest on 4,200 rows is not instant, and the
    # verifier runs every cell before the lesson is ever shown.
    sandbox_timeout_seconds: int = 45
    sandbox_memory_mb: int = 1024
    max_output_chars: int = 12000
    max_attempts_before_walkthrough: int = 3

    # Spaced repetition (SM-2)
    sm2_initial_ease: float = 2.5
    sm2_min_ease: float = 1.3

    interview_minutes_default: int = 45

    # Generation. A lesson is long and must be verified, so it gets more room
    # than a multiple-choice question does.
    generation_temperature: float = 0.6
    review_temperature: float = 0.2
    # Sized to fit inside a free-tier per-minute token budget alongside the
    # prompt, because providers count the RESERVED reply against that budget,
    # not the reply actually produced. Groq's free tier allows 8000 a minute;
    # a ~1700-token prompt plus this leaves comfortable headroom, including
    # for the verifier's second attempt within the same minute.
    lesson_max_tokens: int = 4200
    # What a free key is assumed to allow per minute. Only used by the tests,
    # which fail if a prompt plus its reservation would not fit.
    provider_token_budget: int = 8000
    llm_timeout_seconds: int = 120
    llm_max_retries: int = 2
    prefetch_buffer: int = 10

    # Every code cell in a lesson is executed against the bundled dataset
    # before the lesson is shown. A cell that fails is sent back with its
    # traceback rather than displayed, because a lesson whose code does not
    # run teaches the wrong thing twice.
    verify_lesson_code: bool = True
    max_regeneration_attempts: int = 2


LEARNING = LearningConfig()


# ---------------------------------------------------------------------------
# Key resolution
# ---------------------------------------------------------------------------

def _secrets_file_exists() -> bool:
    """Is there a secrets.toml anywhere Streamlit would look?

    This check exists because of a real failure. Reading `st.secrets` when no
    secrets file is present does not merely raise — on some versions and
    platforms Streamlit *renders an error onto the page*. That render counts
    as a Streamlit command, and if it happens while `core` is still being
    imported, the app's own `st.set_page_config()` is no longer the first
    command in the script and Streamlit refuses to start.

    So the file is checked with plain pathlib before Streamlit is asked
    anything. No file, no question, no page output.
    """
    candidates = (
        APP_ROOT / ".streamlit" / "secrets.toml",
        Path.home() / ".streamlit" / "secrets.toml",
        Path("/etc/streamlit/secrets.toml"),
    )
    for path in candidates:
        try:
            if path.is_file():
                return True
        except OSError:                  # an unreadable path is not a file
            continue
    return False


def _load_streamlit_secrets_once() -> None:
    """Copy Streamlit's secrets into the environment, once, at import.

    On a hosted deployment there is no .env file and no database to read
    settings from until the database credentials themselves are known, so
    secrets are the only place the first boot can get anything. Copying them
    into os.environ means the rest of this module, and core/dbdriver.py, keep
    a single way of asking.
    """
    if not _secrets_file_exists():
        return

    try:
        import streamlit as st           # noqa: PLC0415 - optional at import
        secrets = st.secrets
    except Exception:                    # noqa: BLE001 - not running in Streamlit
        return

    for name in ("GROQ_API_KEY", "NVIDIA_API_KEY", "PROVIDER_ORDER",
                 "TURSO_DATABASE_URL", "TURSO_AUTH_TOKEN"):
        try:
            value = secrets.get(name)
        except Exception:                # noqa: BLE001 - malformed or partial
            return
        if value:
            # A real environment variable still wins, so a local run can
            # override without editing the deployment.
            os.environ.setdefault(name, str(value))


def _load_dotenv_once() -> None:
    """Minimal .env reader. Avoids a python-dotenv dependency for ~10 lines."""
    env_file = APP_ROOT / ".env"
    if not env_file.exists():
        return
    for raw in env_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        name = name.strip()
        value = value.strip().strip('"').strip("'")
        # Real environment variables win over the file.
        os.environ.setdefault(name, value)


# Order matters. A real environment variable beats both, then .env (a laptop),
# then Streamlit secrets (a deployment), because each of the two later sources
# only uses setdefault.
_load_dotenv_once()
_load_streamlit_secrets_once()


def resolve_api_key(provider_id: str, db_settings: dict[str, str] | None = None) -> str | None:
    """Return the API key for a provider, or None if it is not configured."""
    provider = PROVIDERS[provider_id]
    if db_settings:
        stored = db_settings.get(provider.key_name)
        if stored:
            return stored
    value = os.environ.get(provider.key_name)
    return value or None


def configured_providers(db_settings: dict[str, str] | None = None) -> list[str]:
    """Provider ids that have a usable key, in failover order."""
    order = PROVIDER_ORDER
    if db_settings and db_settings.get("PROVIDER_ORDER"):
        candidate = [p.strip() for p in db_settings["PROVIDER_ORDER"].split(",")]
        order = [p for p in candidate if p in PROVIDERS] or PROVIDER_ORDER
    return [p for p in order if resolve_api_key(p, db_settings)]


def database_location() -> str:
    """Plain words for the Settings page: where this run keeps its data."""
    from . import db, dbdriver            # local: both import config, not us
    try:
        return dbdriver.describe(conn=db.get_conn())
    except Exception:                     # noqa: BLE001 - never break Settings
        return dbdriver.describe()
