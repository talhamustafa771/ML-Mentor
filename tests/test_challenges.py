"""Every challenge is solved by its own reference answer, for real.

A challenge whose reference solution fails its own hidden check is unsolvable,
and the learner would spend an hour proving that rather than learning anything.
The Python mentor measured roughly one generated problem in six failing this
way, which is why nothing in this bank is trusted without being executed.

This is the slow suite — each challenge starts a real Python process and fits
real models, so the whole run takes a minute or two. It is worth it: it is the
only check that can tell the difference between a challenge that is hard and a
challenge that is broken.

Run:  python tests/test_challenges.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import challenges, concepts, curriculum, datasets, verify  # noqa: E402

PASS, FAIL = [], []


def check(name: str, condition: bool, detail: str = "") -> None:
    (PASS if condition else FAIL).append(f"{name}{' - ' + detail if detail else ''}")


# ---------------------------------------------------------------------------
# Shape
# ---------------------------------------------------------------------------
ids = [c["id"] for c in challenges.CHALLENGES]
check("every id is unique", len(ids) == len(set(ids)))
check("the bank is not trivially small", len(challenges.CHALLENGES) >= 30,
      f"{len(challenges.CHALLENGES)} challenges")

for challenge in challenges.CHALLENGES:
    cid = challenge["id"]
    check(f"{cid} names a real dataset",
          challenge["dataset"] in datasets.CATALOGUE)
    check(f"{cid} maps to a taught concept",
          challenge["skill"] in concepts.EVERYTHING,
          f"skill {challenge['skill']!r}")
    check(f"{cid} states its task", len(challenge["task"]) > 60)
    check(f"{cid} asserts something", "assert" in challenge["check"])
    check(f"{cid} has hints", len(challenge["hints"]) >= 2)
    check(f"{cid} has a reference solution", len(challenge["solution"]) > 10)

    # Every assert must carry a message. A bare AssertionError tells the
    # learner they were wrong and nothing else, which is the one thing a
    # teaching tool must never do.
    for line in challenge["check"].splitlines():
        stripped = line.strip()
        if stripped.startswith("assert ") and not stripped.endswith("("):
            check(f"{cid}: '{stripped[:40]}...' explains itself",
                  "," in stripped or stripped.endswith("(") or '"' in stripped)

    # The unlock day is derived from the code, so the solution cannot reach
    # past it — but that derivation is itself worth checking, because it is
    # what stops a week-2 reader being handed a week-5 task.
    day = challenge["unlock_day"]
    check(f"{cid} unlocks on a real day", 0 < day <= 91, f"day {day}")
    check(f"{cid} unlocks no earlier than its skill",
          day >= concepts.introduced_on(challenge["skill"]),
          f"day {day} against skill day "
          f"{concepts.introduced_on(challenge['skill'])}")
    ahead = concepts.check_code(challenge["solution"], day)
    check(f"{cid}'s solution stays inside day {day}",
          not ahead, "; ".join(str(v) for v in ahead[:3]))
    ahead_setup = concepts.check_code(challenge["setup"], day)
    check(f"{cid}'s setup stays inside day {day}",
          not ahead_setup, "; ".join(str(v) for v in ahead_setup[:3]))


# ---------------------------------------------------------------------------
# Coverage: the bank must reach across the course, not cluster in week 1
# ---------------------------------------------------------------------------
days = sorted(c["unlock_day"] for c in challenges.CHALLENGES)
weeks = {next(t["week"] for t in curriculum.TOPICS if t["day"] == d)
         for d in days if d}
check("the bank spans at least eight weeks", len(weeks) >= 8,
      f"weeks covered: {sorted(weeks)}")
check("something unlocks in the first week", min(days) <= 7,
      f"earliest day {min(days)}")
check("something unlocks after week 8", max(days) >= 50,
      f"latest day {max(days)}")
check("the bank grows as the course does",
      len(challenges.unlocked_by(10)) < len(challenges.unlocked_by(40))
      < len(challenges.unlocked_by(91)),
      f"{len(challenges.unlocked_by(10))} / {len(challenges.unlocked_by(40))} "
      f"/ {len(challenges.unlocked_by(91))}")

levels = challenges.counts()["by_difficulty"]
for level in ("beginner", "intermediate", "advanced"):
    check(f"there are {level} challenges", levels.get(level, 0) >= 5,
          f"{levels.get(level, 0)}")


# ---------------------------------------------------------------------------
# The real work: run every one
# ---------------------------------------------------------------------------
print(f"Running {len(challenges.CHALLENGES)} challenges against their own "
      "graders. This takes a minute.\n")

started = time.perf_counter()
slow: list[tuple[str, float]] = []

for challenge in challenges.CHALLENGES:
    cid = challenge["id"]
    result = verify.verify_challenge(challenge, challenge["solution"],
                                     timeout=120)
    check(f"{cid}: the reference solution passes its own check", result.ok,
          result.first_error[:300])
    slow.append((cid, result.seconds))
    print(f"  {'ok  ' if result.ok else 'FAIL'} {cid:24s} {result.seconds:5.1f}s")

elapsed = time.perf_counter() - started
slow.sort(key=lambda p: -p[1])
check("no single challenge is punishingly slow", slow[0][1] < 30,
      f"{slow[0][0]} took {slow[0][1]:.0f}s")


# ---------------------------------------------------------------------------
# A wrong answer must actually fail
# ---------------------------------------------------------------------------
# A check that passes whatever you do is worse than no check: it tells the
# learner they understood something they did not. Empty answers are the
# cheapest way to catch that, so every challenge is attempted with nothing.

print("\nConfirming an empty answer fails each one...\n")
empty_passes: list[str] = []
for challenge in challenges.CHALLENGES:
    result = verify.verify_challenge(challenge, "pass", timeout=60)
    if result.ok:
        empty_passes.append(challenge["id"])

check("no challenge is passed by doing nothing", not empty_passes,
      ", ".join(empty_passes))


# ---------------------------------------------------------------------------
print(f"\n{len(PASS)} passed, {len(FAIL)} failed  ({elapsed:.0f}s of execution)")
for line in FAIL:
    print("  FAIL:", line)
sys.exit(1 if FAIL else 0)
