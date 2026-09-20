"""Dual-provider LLM client that repairs its own calls.

Groq and NVIDIA Build both expose the OpenAI chat-completions protocol, so one
client serves both and only base_url, key and model id change.

**Why this module is defensive.** A free-tier provider changes under you: a
model is decommissioned, JSON mode is rejected for a particular checkpoint, a
token ceiling moves. The first version of this file lumped all of those into
"other" and discarded the provider's actual message, which turned a
one-line-fix problem into an unreadable "Every configured provider failed".

So three rules now hold:

  1. **The provider's real message is never thrown away.** Whatever the API
     said comes back in LLMResult.error and is shown in the UI.
  2. **A recognised failure is repaired, not reported.** JSON mode rejected ->
     retry in prose and parse the JSON out. Model gone -> ask the provider what
     models it actually has and switch to the best match. Token ceiling ->
     halve and retry. Each repair is attempted once per call and recorded.
  3. **An unrecognised failure still names itself.** No error is ever reduced
     to a category word.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .config import LEARNING, PROVIDERS, configured_providers, resolve_api_key

try:
    from openai import OpenAI
    _OPENAI_AVAILABLE = True
except ImportError:  # pragma: no cover - surfaced in the UI as a setup message
    OpenAI = None  # type: ignore
    _OPENAI_AVAILABLE = False


# Providers whose key was rejected. Cleared by forget_bad_keys().
_dead_providers: set[str] = set()
# provider_id -> sorted list of model ids, fetched once on demand.
_model_cache: dict[str, list[str]] = {}
# (provider_id, requested_model) -> model that actually works.
_model_substitutions: dict[tuple[str, str], str] = {}
# Providers known to reject response_format=json_object for a given model.
_no_json_mode: set[tuple[str, str]] = set()


def forget_bad_keys() -> None:
    _dead_providers.clear()
    _model_cache.clear()
    _model_substitutions.clear()
    _no_json_mode.clear()


def diagnostics() -> dict[str, Any]:
    """What the client has learned this session. Shown in Settings so a
    silent substitution is never invisible."""
    return {
        "rejected_keys": sorted(_dead_providers),
        "model_substitutions": {f"{p}/{m}": r for (p, m), r in _model_substitutions.items()},
        "json_mode_disabled_for": sorted(f"{p}/{m}" for p, m in _no_json_mode),
        "model_lists_cached": {p: len(v) for p, v in _model_cache.items()},
    }


@dataclass
class LLMResult:
    ok: bool
    text: str = ""
    data: Any = None
    provider: str | None = None
    model: str | None = None
    error: str | None = None
    attempts: list[str] = field(default_factory=list)
    repairs: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.ok


def _client(provider_id: str, api_key: str):
    if not _OPENAI_AVAILABLE:
        raise RuntimeError("The 'openai' package is not installed. "
                           "Run: pip install -r requirements.txt")
    provider = PROVIDERS[provider_id]
    return OpenAI(api_key=api_key, base_url=provider.base_url,
                  timeout=LEARNING.llm_timeout_seconds, max_retries=0)


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------

@dataclass
class Failure:
    kind: str          # auth | rate_limit | timeout | server | bad_model |
                       # bad_json_mode | too_many_tokens | bad_request | other
    message: str       # the provider's own words, as far as they can be read
    status: int | None = None

    def __str__(self) -> str:
        head = f"HTTP {self.status} " if self.status else ""
        return f"{head}{self.kind}: {self.message}"


def _extract_message(exc: Exception) -> str:
    """Dig the provider's actual message out of an SDK exception.

    The SDK puts it in different places depending on version and error type, so
    every known location is tried before falling back to str(exc).
    """
    for attribute in ("message", "body"):
        value = getattr(exc, attribute, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, dict):
            error = value.get("error")
            if isinstance(error, dict) and error.get("message"):
                return str(error["message"])
            if isinstance(error, str):
                return error
            if value.get("message"):
                return str(value["message"])

    response = getattr(exc, "response", None)
    if response is not None:
        try:
            payload = response.json()
            error = payload.get("error", payload)
            if isinstance(error, dict) and error.get("message"):
                return str(error["message"])
            return json.dumps(payload)[:500]
        except Exception:  # noqa: BLE001 - body was not JSON
            text = getattr(response, "text", "")
            if text:
                return str(text)[:500]

    return f"{type(exc).__name__}: {exc}"


def classify(exc: Exception) -> Failure:
    name = type(exc).__name__.lower()
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    message = _extract_message(exc)
    lowered = message.lower()

    if status in (401, 403) or "authentication" in name or "permission" in name:
        return Failure("auth", message, status)

    # A per-minute token budget comes in two shapes that need opposite
    # responses, and telling them apart is the whole point of this block.
    #
    #   "Request too large ... Limit 8000, Requested 8227"
    #       The single request does not fit. No amount of waiting helps; the
    #       reservation has to shrink.
    #
    #   "Rate limit reached ... Limit 8000, Used 3379, Requested 6642.
    #    Please try again in 15.15s"
    #       The request fits fine — 6642 is under 8000 — but this minute's
    #       budget is already part-spent. Waiting is exactly what fixes it,
    #       and the provider says how long.
    #
    # Getting this backwards gives the worst possible advice either way, so
    # the presence of "Used" or a stated wait decides it.
    if status == 413 or "request too large" in lowered:
        return Failure("token_budget", message, status)
    if "tokens per minute" in lowered or "(tpm)" in lowered:
        if _RETRY_AFTER.search(message) or _USED.search(message):
            return Failure("token_wait", message, status)
        return Failure("token_budget", message, status)

    if status == 429 or "ratelimit" in name or "rate limit" in lowered:
        return Failure("rate_limit", message, status)
    if "timeout" in name or "timed out" in lowered:
        return Failure("timeout", message, status)
    if isinstance(status, int) and status >= 500:
        return Failure("server", message, status)
    if "connection" in name:
        return Failure("server", message, status)

    # 400-class errors carry the interesting detail. Order matters: a message
    # can mention both a model and json, and the model problem dominates.
    if any(token in lowered for token in (
            "decommission", "model_not_found", "does not exist",
            "is not a valid model", "unknown model", "model not found",
            "no longer supported", "has been deprecated", "unsupported model")):
        return Failure("bad_model", message, status)
    if "model" in lowered and status == 404:
        return Failure("bad_model", message, status)
    if "response_format" in lowered or "json_object" in lowered or (
            "json" in lowered and status == 400):
        return Failure("bad_json_mode", message, status)
    if any(token in lowered for token in (
            "max_tokens", "context length", "context_length",
            "maximum context", "too many tokens", "reduce the length")):
        return Failure("too_many_tokens", message, status)
    if status == 400:
        return Failure("bad_request", message, status)

    return Failure("other", message, status)


# ---------------------------------------------------------------------------
# Model resolution
# ---------------------------------------------------------------------------

# Preference order when a configured model is gone. Matching is by substring
# against the provider's live list, so a version bump still matches.
_FALLBACK_PREFERENCES = {
    "fast": ["llama-3.3-70b", "llama-3.1-70b", "gpt-oss-20b", "llama-3.1-8b",
             "llama3-70b", "mixtral", "gemma2", "qwen", "instruct"],
    "strong": ["gpt-oss-120b", "llama-3.3-70b", "nemotron", "qwen3",
               "llama-3.1-405b", "llama-3.1-70b", "deepseek", "instruct"],
}

# Never auto-select these for text generation.
_MODEL_EXCLUDE = ("whisper", "tts", "embed", "guard", "vision", "rerank",
                  "moderation", "distil-whisper", "playai")


def fetch_models(provider_id: str, settings: dict[str, str] | None = None,
                 force: bool = False) -> list[str]:
    if not force and provider_id in _model_cache:
        return _model_cache[provider_id]
    api_key = resolve_api_key(provider_id, settings or {})
    if not api_key:
        return []
    try:
        client = _client(provider_id, api_key)
        ids = sorted(m.id for m in client.models.list().data)
        _model_cache[provider_id] = ids
        return ids
    except Exception:  # noqa: BLE001 - absence of a list is not fatal
        _model_cache[provider_id] = []
        return []


def pick_replacement(provider_id: str, tier: str,
                     settings: dict[str, str] | None = None) -> str | None:
    """Choose a live model when the configured one is gone."""
    available = [m for m in fetch_models(provider_id, settings, force=True)
                 if not any(bad in m.lower() for bad in _MODEL_EXCLUDE)]
    if not available:
        return None
    for preference in _FALLBACK_PREFERENCES.get(tier, []):
        for model in available:
            if preference in model.lower():
                return model
    return available[0]


def _model_for(provider_id: str, tier: str, settings: dict[str, str]) -> str:
    provider = PROVIDERS[provider_id]
    override = settings.get(f"MODEL_{provider_id.upper()}_{tier.upper()}")
    base = override or (provider.default_fast_model if tier == "fast"
                        else provider.default_strong_model)
    return _model_substitutions.get((provider_id, base), base)


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------

_FENCE = re.compile(r"```(?:json|python)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str) -> Any:
    """Pull a JSON value out of a model response.

    This is why JSON mode is optional rather than required: prose-wrapped and
    fenced output parse just as well, so a provider that rejects
    response_format costs nothing.
    """
    if not text:
        return None

    # Exact parses first: the whole reply, then anything inside a fence.
    for candidate in [text.strip()] + [b.strip() for b in _FENCE.findall(text)]:
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue

    # Then the outermost container, salvaged if it was cut off. This is tried
    # before the narrower spans below because a reply truncated mid-array still
    # contains a complete first object, and returning that one object instead
    # of the eight-item list would be quietly wrong rather than obviously
    # broken — the worse of the two failures.
    outer = _from_first_opener(text)
    if outer is not None:
        salvaged = _salvage_truncated(outer)
        if salvaged is not None:
            return salvaged
        # A truncated {"questions": [ ... wrapper has no complete top-level
        # pair, so the salvage above finds nothing. The array inside it is the
        # part that matters, and _generate unwraps that shape anyway.
        inner = _INNER_ARRAY.search(outer)
        if inner:
            salvaged = _salvage_truncated(outer[inner.start(1):])
            if salvaged is not None:
                return salvaged

    # Last resort: the widest complete-looking span of each bracket type.
    for opener, closer in (("[", "]"), ("{", "}")):
        start, end = text.find(opener), text.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except (json.JSONDecodeError, ValueError):
                continue
    return None


def _parses_whole(text: str) -> bool:
    try:
        json.loads((text or "").strip())
        return True
    except (json.JSONDecodeError, ValueError):
        return False


def _from_first_opener(text: str) -> str | None:
    """Everything from the first bracket onwards, prose stripped off the front."""
    starts = [text.find(c) for c in "[{" if text.find(c) != -1]
    return text[min(starts):] if starts else None


_CLOSER = {"[": "]", "{": "}"}

# The array inside a {"questions": [...]} style wrapper.
_INNER_ARRAY = re.compile(r':\s*(\[)')


def _snippet(text: str, limit: int = 160) -> str:
    """The head of a reply, on one line, for an error a person has to act on."""
    flat = " ".join((text or "").split())
    if not flat:
        return "'' (the model returned nothing at all)"
    return repr(flat[:limit] + ("…" if len(flat) > limit else ""))


def _blunt(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """The same request, restated for a model that just answered in prose."""
    retry = [dict(m) for m in messages]
    retry[-1]["content"] = retry[-1]["content"].rstrip() + (
        "\n\nYour previous answer could not be parsed. Output raw JSON and "
        "nothing else. No explanation before it, no explanation after it, no "
        "markdown fences, no ``` characters. The very first character you "
        "write must be [ or {, and the very last must be ] or }."
    )
    return retry


def _salvage_truncated(text: str) -> Any:
    """Parse the complete prefix of a cut-off JSON array or object.

    Walks the text tracking string state and bracket depth, then rewinds to the
    last point where a top-level element finished and closes the container
    there. Returns None when there is no complete element to keep, so a reply
    that was garbage from the first character is still reported as garbage.
    """
    text = text.strip()
    if not text or text[0] not in _CLOSER:
        return None

    opener = text[0]
    stack: list[str] = []
    in_string = False
    escaped = False
    last_complete = -1        # index after the last finished top-level element

    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char in "[{":
            stack.append(_CLOSER[char])
        elif char in "]}":
            if not stack or stack[-1] != char:
                return None
            stack.pop()
            if len(stack) == 1:
                last_complete = index + 1
        elif char == "," and len(stack) == 1:
            last_complete = index

    if not stack or last_complete <= 0:
        return None                      # complete already, or nothing usable

    prefix = text[:last_complete].rstrip().rstrip(",")
    try:
        return json.loads(prefix + _CLOSER[opener])
    except (json.JSONDecodeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# The call
# ---------------------------------------------------------------------------

_LIMIT = re.compile(r"limit\s+(\d+)", re.I)
_REQUESTED = re.compile(r"requested\s+(\d+)", re.I)
_USED = re.compile(r"used\s+(\d+)", re.I)
_RETRY_AFTER = re.compile(r"try again in\s+([\d.]+)\s*s", re.I)


def retry_after(message: str, default: float = 20.0) -> float:
    """How long the provider says to wait, in seconds.

    Groq states it to nine decimal places, which is charming and useless; a
    second is added and the whole thing capped, because a minute-long budget
    cannot need more than a minute to refill and a learner staring at a
    spinner deserves an upper bound.
    """
    found = _RETRY_AFTER.search(message or "")
    if not found:
        return default
    try:
        return min(60.0, float(found.group(1)) + 1.0)
    except ValueError:
        return default


def shrink_to_fit(message: str, max_tokens: int,
                  margin: int = 256) -> int | None:
    """A completion budget that would fit, read out of the provider's refusal.

    Providers that enforce a per-minute token cap say exactly what they
    counted: "Limit 8000, Requested 8227". Since the request asked for
    `max_tokens` of completion, the prompt must have been the difference — so
    the largest completion that would have fitted is computable rather than
    guessable, and the app adapts to whatever tier the key is on instead of
    hard-coding one provider's free-tier number.

    Returns None when no budget would work, so the caller can say so plainly
    rather than retrying forever.
    """
    limit = _LIMIT.search(message or "")
    requested = _REQUESTED.search(message or "")
    if not (limit and requested):
        return max(512, max_tokens // 2) if max_tokens > 1024 else None

    allowed = int(limit.group(1))
    asked = int(requested.group(1))
    prompt_tokens = max(0, asked - max_tokens)
    room = allowed - prompt_tokens - margin

    # Below this a lesson comes back truncated, which is worse than an honest
    # failure: it looks like a lesson and stops mid-sentence.
    if room < 700:
        return None
    return min(room, max_tokens - 1)


def _attempt(client, model: str, messages: list[dict[str, str]],
             temperature: float, max_tokens: int, json_mode: bool) -> str:
    kwargs: dict[str, Any] = {
        "model": model, "messages": messages,
        "temperature": temperature, "max_tokens": max_tokens,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    response = client.chat.completions.create(**kwargs)
    return (response.choices[0].message.content or "").strip()


def _run_on_provider(
    provider_id: str, api_key: str, tier: str, messages: list[dict[str, str]],
    *, settings: dict[str, str], json_mode: bool, temperature: float,
    max_tokens: int, trail: list[str], repairs: list[str],
    on_wait: Callable[[str], None] | None = None,
) -> tuple[str | None, Failure | None, str]:
    """Try one provider, repairing recognised failures. Returns (text, failure, model)."""
    provider = PROVIDERS[provider_id]
    model = _model_for(provider_id, tier, settings)
    use_json = json_mode and provider.supports_json_mode
    if (provider_id, model) in _no_json_mode:
        use_json = False

    tried_model_swap = False
    tried_json_drop = False
    tried_token_cut = False
    token_cuts = 0
    token_waits = 0
    last: Failure | None = None

    for attempt in range(LEARNING.llm_max_retries + 3):
        try:
            client = _client(provider_id, api_key)
            return _attempt(client, model, messages, temperature,
                            max_tokens, use_json), None, model
        except Exception as exc:  # noqa: BLE001 - classified immediately below
            failure = classify(exc)
            last = failure
            trail.append(f"{provider_id}/{model}: {failure}")

            if failure.kind == "auth":
                _dead_providers.add(provider_id)
                return None, failure, model

            if failure.kind == "bad_json_mode" and use_json and not tried_json_drop:
                # The model will not take response_format. It can still produce
                # JSON as text, and extract_json handles that, so drop the flag
                # rather than failing the call.
                use_json = False
                tried_json_drop = True
                _no_json_mode.add((provider_id, model))
                repairs.append(
                    f"{provider.label} rejected JSON mode for {model}; "
                    "switched to plain text and parsed the JSON out"
                )
                continue

            if failure.kind == "bad_model" and not tried_model_swap:
                tried_model_swap = True
                replacement = pick_replacement(provider_id, tier, settings)
                if replacement and replacement != model:
                    _model_substitutions[(provider_id, model)] = replacement
                    repairs.append(
                        f"{provider.label} no longer serves '{model}'; "
                        f"switched to '{replacement}'"
                    )
                    model = replacement
                    use_json = json_mode and provider.supports_json_mode
                    continue
                return None, failure, model

            if failure.kind == "too_many_tokens" and not tried_token_cut:
                tried_token_cut = True
                max_tokens = max(512, max_tokens // 2)
                repairs.append(f"reduced max_tokens to {max_tokens} for {model}")
                continue

            if failure.kind == "token_wait" and token_waits < 3:
                token_waits += 1
                pause = retry_after(failure.message)
                if on_wait:
                    on_wait(f"The free tier's per-minute token budget is "
                            f"already part-spent. Waiting {pause:.0f}s, then "
                            "carrying on — nothing is lost.")
                repairs.append(
                    f"waited {pause:.0f}s for {provider.label}'s per-minute "
                    "token budget to refill")
                time.sleep(pause)
                continue

            if failure.kind == "token_budget" and token_cuts < 2:
                fitted = shrink_to_fit(failure.message, max_tokens)
                if fitted is None:
                    return None, failure, model
                token_cuts += 1
                max_tokens = fitted
                repairs.append(
                    f"{provider.label} caps this key at a per-minute token "
                    f"budget; asked for {max_tokens} output tokens instead"
                )
                continue

            if failure.kind in ("rate_limit", "server", "timeout"):
                if attempt < LEARNING.llm_max_retries:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                return None, failure, model

            return None, failure, model

    return None, last, model


def complete(
    system: str,
    user: str,
    *,
    settings: dict[str, str] | None = None,
    tier: str = "fast",
    json_mode: bool = False,
    temperature: float | None = None,
    max_tokens: int = 4096,
    on_wait: Callable[[str], None] | None = None,
) -> LLMResult:
    settings = settings or {}
    providers = [p for p in configured_providers(settings) if p not in _dead_providers]
    if not providers:
        if configured_providers(settings):
            return LLMResult(ok=False, error=(
                "Your API key was rejected. Open Settings, check the key, and "
                "press Save — that clears the rejection."))
        return LLMResult(ok=False, error=(
            "No AI provider is configured. Open Settings and paste a Groq or "
            "NVIDIA Build API key — both have free tiers."))

    if temperature is None:
        temperature = (LEARNING.review_temperature if tier == "strong"
                       else LEARNING.generation_temperature)

    if json_mode:
        # Groq's response_format=json_object constrains the model to emit an
        # object, while most of these prompts ask for an array. Left
        # unaddressed the model is being pulled two ways and sometimes answers
        # in prose instead, which is the failure the Practice page hit. Saying
        # the wrapper is acceptable resolves it; every caller already unwraps a
        # single list-valued key.
        user = user.rstrip() + (
            "\n\nReturn valid json and nothing else — no commentary, no "
            "markdown fences. If the format above is an array and you must "
            "return an object, return {\"items\": [ ...that array... ]}."
        )

    messages = [{"role": "system", "content": system},
                {"role": "user", "content": user}]

    trail: list[str] = []
    repairs: list[str] = []
    last: Failure | None = None

    for provider_id in providers:
        api_key = resolve_api_key(provider_id, settings)
        if not api_key:
            continue

        text, failure, model = _run_on_provider(
            provider_id, api_key, tier, messages, settings=settings,
            json_mode=json_mode, temperature=temperature,
            max_tokens=max_tokens, trail=trail, repairs=repairs,
            on_wait=on_wait,
        )

        if text is not None:
            if not json_mode:
                return LLMResult(True, text, None, provider_id, model,
                                 attempts=trail, repairs=repairs)
            parsed = extract_json(text)
            if parsed is not None:
                if not _parses_whole(text):
                    # It worked, but only after repair. Say so, so a model that
                    # is quietly being truncated on every call is visible in
                    # Settings rather than showing up as short lesson sets.
                    repairs.append(
                        f"{provider_id}/{model} returned JSON that was cut off "
                        "or wrapped in prose; the complete part was recovered"
                    )
                return LLMResult(True, text, parsed, provider_id, model,
                                 attempts=trail, repairs=repairs)

            # Keep what the model actually said. A bare "not usable JSON" tells
            # nobody anything; the first line of the reply usually says plainly
            # whether it refused, moralised, wrote prose, or ran out of room.
            trail.append(f"{provider_id}/{model}: reply was not parseable as "
                         f"JSON — it began {_snippet(text)}")

            # One reroll on the same provider before giving up on it. JSON mode
            # is dropped and the instruction made blunt, because a model that
            # has just produced prose under a polite instruction tends to do it
            # again under the same one.
            retry_text, retry_failure, _ = _run_on_provider(
                provider_id, api_key, tier, _blunt(messages), settings=settings,
                json_mode=False, temperature=min(temperature, 0.3),
                max_tokens=max_tokens, trail=trail, repairs=repairs,
            )
            if retry_text is not None:
                parsed = extract_json(retry_text)
                if parsed is not None:
                    repairs.append(
                        f"{provider_id}/{model} first replied with unparseable "
                        "text; asked again for raw JSON only and that worked"
                    )
                    return LLMResult(True, retry_text, parsed, provider_id,
                                     model, attempts=trail, repairs=repairs)
                trail.append(f"{provider_id}/{model}: the retry was not JSON "
                             f"either — it began {_snippet(retry_text)}")

            last = retry_failure or Failure(
                "bad_json_reply",
                f"the model replied but not with usable JSON. It began "
                f"{_snippet(text)}")
            continue

        last = failure

    return LLMResult(
        ok=False,
        error=_readable_error(last, trail),
        attempts=trail,
        repairs=repairs,
    )


def _readable_error(failure: Failure | None, trail: list[str]) -> str:
    """One sentence a learner can act on, then the raw detail.

    The point is that the provider's own words always survive: the previous
    version collapsed everything to a category and left nothing to debug with.
    """
    if failure is None:
        return "No provider could be reached. " + "; ".join(trail)

    advice = {
        "auth": "Your API key was rejected. Generate a new one and paste it "
                "into Settings.",
        "rate_limit": "You have hit the provider's free-tier rate limit. Wait a "
                      "minute, or add the second provider in Settings so the "
                      "app can fail over.",
        "timeout": "The provider did not respond in time. Try again; if it "
                   "keeps happening, switch provider in Settings.",
        "server": "The provider is having problems on their end. Try again "
                  "shortly.",
        "bad_model": "The configured model no longer exists and no replacement "
                     "could be found. Open Settings → Models and press Refresh "
                     "to pick one from the live list.",
        "bad_json_mode": "The model refused structured-output mode and the "
                         "plain-text fallback also failed.",
        "too_many_tokens": "The request was too long for this model even after "
                           "being reduced.",
        "token_wait": "The free tier's per-minute token budget was already "
                      "part-spent when this ran, and it did not refill in "
                      "time. This one IS fixed by waiting: press the button "
                      "again in about half a minute. Generating a lesson and "
                      "then re-checking it are two requests, which is more "
                      "than one free-tier minute holds.",
        "token_budget": "Your API key has a per-minute token budget, and this "
                        "request did not fit even after being shrunk. This is "
                        "not something waiting will fix. Either pick a model "
                        "with a larger budget in Settings → Models, or use a "
                        "key on a paid tier. Groq's free tier allows 8000 "
                        "tokens a minute and counts the reserved reply against "
                        "it.",
        "bad_request": "The provider rejected the request as malformed.",
        "bad_json_reply": "The model answered, but not in the format this "
                          "screen needs, and asking it again a second way did "
                          "not help either. What it actually said is below — "
                          "if it looks like a refusal or a complaint, the "
                          "prompt is the problem; if it looks like it stops "
                          "mid-sentence, the model ran out of room. A "
                          "different model in Settings usually fixes both.",
    }.get(failure.kind, "The request failed.")

    return f"{advice}\n\nProvider said: {failure.message}"


def complete_json(system: str, user: str, **kwargs: Any) -> LLMResult:
    return complete(system, user, json_mode=True, **kwargs)


def chat(
    messages: list[dict[str, str]], *, settings: dict[str, str] | None = None,
    tier: str = "fast", temperature: float = 0.6, max_tokens: int = 2048,
) -> LLMResult:
    """Multi-turn variant for the tutor, with the same repair behaviour."""
    settings = settings or {}
    providers = [p for p in configured_providers(settings) if p not in _dead_providers]
    if not providers:
        return LLMResult(ok=False, error="No AI provider is configured.")

    trail: list[str] = []
    repairs: list[str] = []
    last: Failure | None = None

    for provider_id in providers:
        api_key = resolve_api_key(provider_id, settings)
        if not api_key:
            continue
        text, failure, model = _run_on_provider(
            provider_id, api_key, tier, messages, settings=settings,
            json_mode=False, temperature=temperature, max_tokens=max_tokens,
            trail=trail, repairs=repairs,
        )
        if text is not None:
            return LLMResult(True, text, None, provider_id, model,
                             attempts=trail, repairs=repairs)
        last = failure

    return LLMResult(ok=False, error=_readable_error(last, trail),
                     attempts=trail, repairs=repairs)


def list_models(provider_id: str, settings: dict[str, str] | None = None
                ) -> tuple[bool, list[str] | str]:
    api_key = resolve_api_key(provider_id, settings or {})
    if not api_key:
        return False, f"No API key set for {PROVIDERS[provider_id].label}."
    try:
        client = _client(provider_id, api_key)
        ids = sorted(m.id for m in client.models.list().data)
        _model_cache[provider_id] = ids
        return True, ids
    except Exception as exc:  # noqa: BLE001
        failure = classify(exc)
        return False, str(failure)


def health_check(settings: dict[str, str] | None = None) -> dict[str, str]:
    settings = settings or {}
    out: dict[str, str] = {}
    for provider_id, provider in PROVIDERS.items():
        if not resolve_api_key(provider_id, settings):
            out[provider.label] = "no key"
            continue
        ok, payload = list_models(provider_id, settings)
        out[provider.label] = (f"connected ({len(payload)} models)" if ok
                               else f"failed - {payload}")
    return out


def smoke_test(provider_id: str, tier: str = "fast",
               settings: dict[str, str] | None = None) -> dict[str, Any]:
    """Make one real call and report exactly what happened.

    This is what the Settings page runs when generation fails, so the answer is
    the provider's own words rather than a guess.
    """
    settings = settings or {}
    api_key = resolve_api_key(provider_id, settings)
    if not api_key:
        return {"ok": False, "stage": "key",
                "error": f"No API key set for {PROVIDERS[provider_id].label}."}

    ok, models = list_models(provider_id, settings)
    if not ok:
        return {"ok": False, "stage": "models", "error": models}

    configured = _model_for(provider_id, tier, settings)
    present = configured in models

    plain = complete("Reply with one word.", "Say OK.", settings=settings,
                     tier=tier, max_tokens=16)
    structured = complete_json(
        "You return json.",
        'Return exactly this json object: {"ok": true}',
        settings=settings, tier=tier, max_tokens=64,
    )

    return {
        "ok": plain.ok and structured.ok,
        "stage": "call",
        "configured_model": configured,
        "model_in_live_list": present,
        "models_available": len(models),
        "plain_text_call": "worked" if plain.ok else f"failed — {plain.error}",
        "json_call": "worked" if structured.ok else f"failed — {structured.error}",
        "model_actually_used": structured.model or plain.model,
        "repairs_applied": (plain.repairs or []) + (structured.repairs or []),
        "closest_matches": [m for m in models
                            if configured.split("/")[-1][:8] in m][:5],
    }
