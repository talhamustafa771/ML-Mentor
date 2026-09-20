"""Runs your code, and runs every lesson's code before you ever see it.

Honest scope note: this is a *safety net for accidents*, not a security
sandbox. It runs on your own machine, executing code you wrote yourself, so the
threat model is "an infinite loop, a runaway allocation, or an accidental
`os.remove`", not a hostile attacker. Three layers handle that:

  1. A static AST scan rejects a short list of genuinely dangerous calls before
     anything executes.
  2. Execution happens in a separate `python -I -B` process, so a crash, a
     `sys.exit` or a memory blow-up cannot take the app down with it.
  3. A wall-clock timeout always applies. On Linux and macOS, CPU and address
     space limits apply as well via the `resource` module; Windows has no
     equivalent, so only the timeout protects there. This is stated rather than
     glossed over because it is a real difference in behaviour.

If you ever point this at code you did not write, run it in a container.
"""
from __future__ import annotations

import ast
import json
import os
import platform
import subprocess
import sys
import tempfile
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import APP_ROOT, LEARNING

IS_WINDOWS = platform.system() == "Windows"

try:
    import resource  # POSIX only
    _HAS_RESOURCE = True
except ImportError:  # pragma: no cover - Windows
    resource = None  # type: ignore
    _HAS_RESOURCE = False


# Modules whose use inside a practice problem is almost always a mistake.
# `os` and `sys` are allowed because `os.path` and `sys.setrecursionlimit` are
# legitimately useful; the dangerous *calls* are blocked below instead.
BLOCKED_MODULES = {
    "subprocess", "shutil", "socket", "ctypes", "multiprocessing",
    "http", "urllib", "urllib3", "requests", "httpx", "ftplib", "smtplib",
    "telnetlib", "pickle", "marshal", "importlib", "pty", "webbrowser",
    # Not installed, and deliberately so: a deep-learning import in a
    # generated cell means the lesson wandered out of the syllabus, and
    # torch alone would not fit in the free hosting tier.
    "torch", "tensorflow", "keras", "jax",
}

# Blocking the network here does a second job beyond safety. Every dataset in
# this course is bundled, and a lesson that quietly downloads one cannot be
# verified, cannot be reproduced, and breaks on the hosted plan where the disk
# is wiped on every restart. A generated cell that reaches for the internet is
# a broken lesson, and this is where it stops.

BLOCKED_CALLS = {
    ("os", "system"), ("os", "popen"), ("os", "remove"), ("os", "unlink"),
    ("os", "rmdir"), ("os", "removedirs"), ("os", "rename"), ("os", "kill"),
    ("os", "execv"), ("os", "fork"), ("os", "chmod"), ("os", "chown"),
    ("shutil", "rmtree"), ("sys", "exit"),
}

BLOCKED_NAMES = {"eval", "exec", "compile", "__import__", "breakpoint", "input"}


@dataclass
class CaseResult:
    index: int
    passed: bool
    args: Any = None
    expected: Any = None
    got: Any = None
    error: str | None = None
    hidden: bool = False
    seconds: float = 0.0


@dataclass
class RunResult:
    verdict: str                       # pass | fail | error | timeout | blocked
    passed: int = 0
    total: int = 0
    cases: list[CaseResult] = field(default_factory=list)
    stdout: str = ""
    stderr: str = ""
    seconds: float = 0.0
    message: str = ""

    @property
    def ok(self) -> bool:
        return self.verdict == "pass"

    @property
    def score(self) -> float:
        return (self.passed / self.total) if self.total else 0.0

    def first_failure(self) -> CaseResult | None:
        return next((c for c in self.cases if not c.passed), None)


# ---------------------------------------------------------------------------
# Static scan
# ---------------------------------------------------------------------------

def _blocked_module_message(root: str) -> str:
    """Say why, not just no — the reason is itself part of the course."""
    if root in {"urllib", "urllib3", "requests", "httpx", "socket", "http",
                "ftplib", "smtplib", "telnetlib"}:
        return (f"'{root}' is blocked because nothing in this course downloads "
                "anything. Every dataset is bundled: use  "
                "from core.datasets import load")
    if root in {"pickle", "marshal"}:
        return (f"'{root}' is blocked here. Day 65 covers why loading a "
                "pickle you did not create is dangerous; joblib is the tool "
                "this course uses to save a model.")
    if root in {"torch", "tensorflow", "keras", "jax"}:
        return (f"'{root}' is not installed. This is a classical machine "
                "learning course: neural networks are built from numpy and "
                "scikit-learn's MLP, which is enough to see how they work.")
    return (f"'{root}' is not available here. This course runs on numpy, "
            "pandas, scikit-learn, matplotlib and seaborn.")


def static_check(code: str, *, allow_input: bool = False) -> tuple[bool, str]:
    """Return (allowed, reason). Syntax errors are reported here rather than
    being discovered as a confusing subprocess traceback.

    `input()` is blocked by default because a function-style problem that waits
    on stdin just hangs until the timeout. Script problems feed stdin
    deliberately, so they pass allow_input=True.
    """
    if not code.strip():
        return False, "You have not written any code yet."

    blocked_names = BLOCKED_NAMES - {"input"} if allow_input else BLOCKED_NAMES

    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        line = exc.lineno or "?"
        return False, f"SyntaxError on line {line}: {exc.msg}"

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in BLOCKED_MODULES:
                    return False, _blocked_module_message(root)
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root in BLOCKED_MODULES:
                return False, _blocked_module_message(root)
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in blocked_names:
                return False, f"'{func.id}()' is not allowed here."
            if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
                if (func.value.id, func.attr) in BLOCKED_CALLS:
                    return False, f"'{func.value.id}.{func.attr}()' is not allowed here."
            if isinstance(func, ast.Name) and func.id == "open":
                return False, ("Reading and writing files is not available here. The datasets are loaded with  from core.datasets import load")

    return True, ""


def find_function_names(code: str) -> list[str]:
    """Top-level function names, so the UI can tell you that you defined
    `two_sums` when the problem asked for `two_sum`."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    return [n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

_HARNESS = '''
import json as _json, sys as _sys, time as _time

_RESULTS = []
_CASES = _json.loads(_sys.argv[1]) if len(_sys.argv) > 1 else []
_FN_NAME = _sys.argv[2] if len(_sys.argv) > 2 else ""

def _normalise(value):
    """Tuples and lists compare equal; sets and dicts sort for stability."""
    if isinstance(value, tuple):
        return [_normalise(v) for v in value]
    if isinstance(value, list):
        return [_normalise(v) for v in value]
    if isinstance(value, set):
        try:
            return sorted(_normalise(v) for v in value)
        except TypeError:
            return sorted(map(repr, value))
    if isinstance(value, dict):
        return {str(k): _normalise(v) for k, v in value.items()}
    return value

def _matches(got, expected, mode):
    g, e = _normalise(got), _normalise(expected)
    if mode == "set":
        try:
            return sorted(map(repr, g)) == sorted(map(repr, e))
        except TypeError:
            return g == e
    if mode == "sorted":
        try:
            return sorted(g) == sorted(e)
        except TypeError:
            return g == e
    if mode == "approx":
        try:
            return abs(float(g) - float(e)) < 1e-6
        except (TypeError, ValueError):
            return g == e
    return g == e

_fn = globals().get(_FN_NAME)
if _fn is None or not callable(_fn):
    print("__HARNESS__" + _json.dumps({
        "fatal": "missing_function",
        "expected_name": _FN_NAME,
        "defined": [k for k, v in list(globals().items())
                    if callable(v) and not k.startswith("_")],
    }))
    raise SystemExit(0)

for _i, _case in enumerate(_CASES):
    _args = _case.get("args", [])
    _kwargs = _case.get("kwargs", {})
    _mode = _case.get("check", "eq")
    _start = _time.perf_counter()
    try:
        _got = _fn(*_args, **_kwargs)
        _ok = _matches(_got, _case.get("expected"), _mode)
        _RESULTS.append({
            "index": _i, "passed": bool(_ok),
            "got": _normalise(_got), "error": None,
            "seconds": round(_time.perf_counter() - _start, 6),
        })
    except RecursionError:
        _RESULTS.append({"index": _i, "passed": False, "got": None,
                         "error": "RecursionError: too deep. Check your base case.",
                         "seconds": round(_time.perf_counter() - _start, 6)})
    except Exception as _exc:
        _RESULTS.append({"index": _i, "passed": False, "got": None,
                         "error": type(_exc).__name__ + ": " + str(_exc),
                         "seconds": round(_time.perf_counter() - _start, 6)})

print("__HARNESS__" + _json.dumps({"results": _RESULTS}))
'''


def _limits(timeout: int | None = None) -> Any:
    """preexec_fn applying CPU and memory caps. POSIX only.

    The CPU cap is set one second ABOVE the caller's wall-clock timeout on
    purpose. A pure busy-loop would otherwise be killed by the CPU limit first,
    which surfaces as a generic "the process was killed" message; letting the
    wall clock win gives the learner the specific "this looks like an infinite
    loop" explanation instead. The CPU cap remains as the backstop for a
    process the wall clock somehow misses.
    """
    if not _HAS_RESOURCE:
        return None

    cpu = (timeout or LEARNING.sandbox_timeout_seconds) + 1

    def apply() -> None:  # pragma: no cover - runs in the child process
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 1))
        mem = LEARNING.sandbox_memory_mb * 1024 * 1024
        try:
            resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
        except (ValueError, OSError):
            pass  # some platforms reject RLIMIT_AS; the CPU cap still applies
        resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))
        resource.setrlimit(resource.RLIMIT_FSIZE, (1024 * 1024, 1024 * 1024))

    return apply


def run_tests(
    code: str,
    tests: list[dict[str, Any]],
    function_name: str,
    *,
    timeout: int | None = None,
    reveal_hidden: bool = False,
) -> RunResult:
    """Execute `code`, call `function_name` against `tests`, return a verdict.

    Each test is {"args": [...], "expected": ..., "kwargs": {...}?,
    "check": "eq"|"set"|"sorted"|"approx", "hidden": bool}.
    Hidden cases count toward the verdict but their inputs are not shown until
    `reveal_hidden`, which the UI sets only after the walkthrough is unlocked.
    """
    timeout = timeout or LEARNING.sandbox_timeout_seconds

    allowed, reason = static_check(code)
    if not allowed:
        return RunResult(verdict="blocked", total=len(tests), message=reason)

    script = code.rstrip() + "\n\n" + textwrap.dedent(_HARNESS)
    payload = json.dumps(tests)

    tmp_dir = Path(tempfile.mkdtemp(prefix="pydsa_"))
    script_path = tmp_dir / "submission.py"
    script_path.write_text(script, encoding="utf-8")

    env = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONIOENCODING": "utf-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
    }
    if IS_WINDOWS:
        # Windows needs these to locate the interpreter's own DLLs.
        for key in ("SYSTEMROOT", "TEMP", "TMP", "PATHEXT", "COMSPEC"):
            if key in os.environ:
                env[key] = os.environ[key]

    popen_kwargs: dict[str, Any] = {
        "cwd": str(tmp_dir),
        "env": env,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "text": True,
    }
    limiter = _limits(timeout)
    if limiter is not None:
        popen_kwargs["preexec_fn"] = limiter

    started = time.perf_counter()
    try:
        proc = subprocess.run(
            [sys.executable, "-I", "-B", str(script_path), payload, function_name],
            timeout=timeout,
            **popen_kwargs,
        )
        stdout, stderr, returncode = proc.stdout, proc.stderr, proc.returncode
    except subprocess.TimeoutExpired:
        return RunResult(
            verdict="timeout", total=len(tests),
            seconds=time.perf_counter() - started,
            message=(f"Your code did not finish within {timeout} seconds. "
                     "That usually means an infinite loop, a `while` condition "
                     "that never becomes False, or a recursion with no base case."),
        )
    except Exception as exc:  # noqa: BLE001 - subprocess launch failure
        return RunResult(verdict="error", total=len(tests),
                         message=f"Could not start the sandbox: {exc}")
    finally:
        _cleanup(tmp_dir)

    elapsed = time.perf_counter() - started
    marker = "__HARNESS__"
    harness_line = next(
        (ln for ln in stdout.splitlines() if ln.startswith(marker)), None
    )
    user_stdout = "\n".join(
        ln for ln in stdout.splitlines() if not ln.startswith(marker)
    )[: LEARNING.max_output_chars]

    if harness_line is None:
        # The process died before the harness printed: a top-level exception,
        # a memory limit kill, or a syntax error the AST pass let through.
        return RunResult(
            verdict="error", total=len(tests), stdout=user_stdout,
            stderr=(stderr or "")[: LEARNING.max_output_chars], seconds=elapsed,
            message=_explain_crash(stderr, returncode),
        )

    parsed = json.loads(harness_line[len(marker):])

    if parsed.get("fatal") == "missing_function":
        defined = parsed.get("defined", [])
        hint = f" You defined: {', '.join(defined)}." if defined else ""
        return RunResult(
            verdict="error", total=len(tests), stdout=user_stdout,
            stderr=stderr[: LEARNING.max_output_chars], seconds=elapsed,
            message=(f"No function named `{function_name}` was found.{hint} "
                     "The name must match exactly, including underscores."),
        )

    cases: list[CaseResult] = []
    for raw in parsed.get("results", []):
        i = raw["index"]
        spec = tests[i] if i < len(tests) else {}
        hidden = bool(spec.get("hidden")) and not reveal_hidden
        cases.append(CaseResult(
            index=i,
            passed=bool(raw["passed"]),
            args=None if hidden else spec.get("args"),
            expected=None if hidden else spec.get("expected"),
            got=None if hidden else raw.get("got"),
            error=raw.get("error"),
            hidden=hidden,
            seconds=raw.get("seconds", 0.0),
        ))

    passed = sum(1 for c in cases if c.passed)
    verdict = "pass" if passed == len(tests) and len(tests) > 0 else "fail"
    return RunResult(
        verdict=verdict, passed=passed, total=len(tests), cases=cases,
        stdout=user_stdout, stderr=stderr[: LEARNING.max_output_chars],
        seconds=elapsed,
    )


def run_script_tests(
    code: str,
    tests: list[dict[str, Any]],
    *,
    timeout: int | None = None,
    reveal_hidden: bool = False,
) -> RunResult:
    """Grade a whole program by what it prints, not by what it returns.

    This is how exercises work before functions are taught on day 43. A student
    on day 2 knows `print` and a variable; asking them to `return` a value from
    a `def` is asking for something six weeks away, so the exercise instead
    says "make your program print exactly this" and the check is a comparison
    of stdout.

    Each test is {"stdin": "...", "expected_output": "...", "hidden": bool}.
    Before `input()` is taught there is usually one test with empty stdin; once
    it is, stdin drives the cases and the answer can no longer be hard-coded.
    """
    timeout = timeout or LEARNING.sandbox_timeout_seconds

    allowed, reason = static_check(code, allow_input=True)
    if not allowed:
        return RunResult(verdict="blocked", total=len(tests), message=reason)
    if not tests:
        return RunResult(verdict="error", total=0,
                         message="This exercise has no expected output to check against.")

    tmp_dir = Path(tempfile.mkdtemp(prefix="pydsa_script_"))
    script_path = tmp_dir / "submission.py"
    script_path.write_text(code.rstrip() + "\n", encoding="utf-8")

    env = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONIOENCODING": "utf-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
    }
    if IS_WINDOWS:
        for key in ("SYSTEMROOT", "TEMP", "TMP", "PATHEXT", "COMSPEC"):
            if key in os.environ:
                env[key] = os.environ[key]

    cases: list[CaseResult] = []
    first_stdout = ""
    first_stderr = ""
    started = time.perf_counter()

    try:
        for index, spec in enumerate(tests):
            stdin_text = str(spec.get("stdin", "") or "")
            expected = str(spec.get("expected_output", ""))
            hidden = bool(spec.get("hidden")) and not reveal_hidden

            popen_kwargs: dict[str, Any] = {
                "cwd": str(tmp_dir), "env": env,
                "stdout": subprocess.PIPE, "stderr": subprocess.PIPE,
                "input": stdin_text, "text": True,
            }
            limiter = _limits(timeout)
            if limiter is not None:
                popen_kwargs["preexec_fn"] = limiter

            case_started = time.perf_counter()
            try:
                proc = subprocess.run(
                    [sys.executable, "-I", "-B", str(script_path)],
                    timeout=timeout, **popen_kwargs,
                )
                out, err, code_ = proc.stdout, proc.stderr, proc.returncode
            except subprocess.TimeoutExpired:
                return RunResult(
                    verdict="timeout", total=len(tests),
                    seconds=time.perf_counter() - started,
                    message=(f"Your program did not finish within {timeout} "
                             "seconds. That usually means a loop that never "
                             "ends, or an `input()` waiting for something the "
                             "exercise does not provide."),
                )
            except Exception as exc:  # noqa: BLE001
                return RunResult(verdict="error", total=len(tests),
                                 message=f"Could not start the sandbox: {exc}")

            if index == 0:
                first_stdout = out[: LEARNING.max_output_chars]
                first_stderr = (err or "")[: LEARNING.max_output_chars]

            if code_ != 0:
                cases.append(CaseResult(
                    index=index, passed=False,
                    args=None if hidden else stdin_text,
                    expected=None if hidden else expected,
                    got=None if hidden else out.strip(),
                    error=_explain_crash(err, code_), hidden=hidden,
                    seconds=time.perf_counter() - case_started,
                ))
                continue

            passed = _output_matches(out, expected)
            cases.append(CaseResult(
                index=index, passed=passed,
                args=None if hidden else stdin_text,
                expected=None if hidden else expected,
                got=None if hidden else out.strip(),
                hidden=hidden,
                seconds=time.perf_counter() - case_started,
            ))
    finally:
        _cleanup(tmp_dir)

    passed = sum(1 for c in cases if c.passed)
    return RunResult(
        verdict="pass" if passed == len(tests) else "fail",
        passed=passed, total=len(tests), cases=cases,
        stdout=first_stdout, stderr=first_stderr,
        seconds=time.perf_counter() - started,
    )


def _output_matches(actual: str, expected: str) -> bool:
    """Compare printed output forgivingly enough to be fair, strictly enough
    to mean something.

    Trailing whitespace on a line, a missing final newline and blank lines at
    either end are the student's editor, not their understanding. Everything
    else — spelling, capitalisation, punctuation, the order of lines — counts,
    because "print exactly this" is the whole exercise.
    """
    def norm(text: str) -> list[str]:
        lines = [line.rstrip() for line in (text or "").replace("\r\n", "\n").split("\n")]
        while lines and not lines[0]:
            lines.pop(0)
        while lines and not lines[-1]:
            lines.pop()
        return lines

    return norm(actual) == norm(expected)


def run_scratch(code: str, timeout: int | None = None) -> RunResult:
    """Run a snippet for its output only, with no tests. Used by the Learn page
    so an example can be executed and altered in place."""
    timeout = timeout or LEARNING.sandbox_timeout_seconds
    allowed, reason = static_check(code)
    if not allowed:
        return RunResult(verdict="blocked", message=reason)

    tmp_dir = Path(tempfile.mkdtemp(prefix="pydsa_scratch_"))
    script_path = tmp_dir / "scratch.py"
    script_path.write_text(code, encoding="utf-8")

    env = {"PATH": os.environ.get("PATH", ""), "PYTHONIOENCODING": "utf-8",
           "PYTHONDONTWRITEBYTECODE": "1"}
    if IS_WINDOWS:
        for key in ("SYSTEMROOT", "TEMP", "TMP", "PATHEXT", "COMSPEC"):
            if key in os.environ:
                env[key] = os.environ[key]

    kwargs: dict[str, Any] = {"cwd": str(tmp_dir), "env": env, "text": True,
                              "stdout": subprocess.PIPE, "stderr": subprocess.PIPE}
    limiter = _limits(timeout)
    if limiter is not None:
        kwargs["preexec_fn"] = limiter

    started = time.perf_counter()
    try:
        proc = subprocess.run([sys.executable, "-I", "-B", str(script_path)],
                              timeout=timeout, **kwargs)
        return RunResult(
            verdict="pass" if proc.returncode == 0 else "error",
            stdout=proc.stdout[: LEARNING.max_output_chars],
            stderr=proc.stderr[: LEARNING.max_output_chars],
            seconds=time.perf_counter() - started,
            message="" if proc.returncode == 0 else _explain_crash(proc.stderr, proc.returncode),
        )
    except subprocess.TimeoutExpired:
        return RunResult(verdict="timeout",
                         message=f"Stopped after {timeout}s — likely an infinite loop.")
    finally:
        _cleanup(tmp_dir)


def _cleanup(tmp_dir: Path) -> None:
    try:
        for child in tmp_dir.iterdir():
            child.unlink(missing_ok=True)
        tmp_dir.rmdir()
    except OSError:
        pass  # a leftover temp dir is harmless; the OS reclaims it


def _explain_crash(stderr: str, returncode: int) -> str:
    """Turn a raw traceback into one sentence a learner can act on."""
    text = stderr or ""
    tail = text.strip().splitlines()[-1] if text.strip() else ""
    table = {
        "IndentationError": "Your indentation is inconsistent — check that every block under a `:` is indented the same way.",
        "NameError": "You used a name that was never defined. Check spelling, and that the variable is assigned before use.",
        "TypeError": "A value had the wrong type for the operation — for example adding a string to an integer.",
        "IndexError": "You indexed past the end of a sequence. Remember the last valid index is len(x) - 1.",
        "KeyError": "You looked up a dictionary key that does not exist. Use .get() or check membership first.",
        "ZeroDivisionError": "Something divided by zero. Guard the denominator before dividing.",
        "AttributeError": "You called a method the object does not have. Check the type of the value at that point.",
        "ValueError": "A function got a value of the right type but an unacceptable value.",
        "MemoryError": "Your code allocated more memory than the sandbox allows — usually an unbounded list or loop.",
    }
    for name, explanation in table.items():
        if name in text:
            return f"{explanation}\n\n{tail}" if tail else explanation
    if returncode and returncode < 0:
        return ("The process was killed, most often by hitting the memory or CPU "
                "limit. Look for an unbounded loop or a very large allocation.")
    return tail or f"The program exited with code {returncode}."


def environment_notes() -> dict[str, Any]:
    """Shown in Settings so the safety limits in force are never a guess."""
    return {
        "platform": platform.system(),
        "python": sys.version.split()[0],
        "wall_clock_timeout_seconds": LEARNING.sandbox_timeout_seconds,
        "cpu_and_memory_limits": bool(_HAS_RESOURCE),
        "memory_limit_mb": LEARNING.sandbox_memory_mb if _HAS_RESOURCE else None,
        "note": ("CPU and memory limits are enforced." if _HAS_RESOURCE else
                 "Windows has no resource-limit API, so only the wall-clock "
                 "timeout applies. Code is still run in a separate process."),
    }


# ===========================================================================
# Notebooks
# ===========================================================================
# Everything above grades one function or one program. A machine learning
# lesson is neither: it is a sequence of cells that build on each other —
# load the data, split it, fit something, look at what came back — and its
# output is as often a picture as a number.
#
# So this section adds a second runner. It executes a whole lesson in ONE
# process, keeping the namespace alive between cells, capturing each cell's
# printed output and any figure it drew.
#
# One process rather than one per cell, for two reasons. Importing pandas and
# scikit-learn costs a little over two seconds, and paying that six times per
# lesson is the difference between a page that feels alive and one that does
# not. More importantly, cell 4 cannot fit the model that cell 2 split the data
# for if cell 2's variables died with its process — and a lesson written as
# independent, self-contained cells would have to re-load and re-split in every
# one, which is not how anybody writes this code in real life.

import base64                                          # noqa: E402
import re                                              # noqa: E402

MAX_FIGURES = 8
MAX_FIGURE_BYTES = 600_000

# Applied before the first cell runs, so every chart in the course looks like
# it belongs to the same application rather than to matplotlib's defaults.
PLOT_STYLE = """
import matplotlib as _mpl
_mpl.rcParams.update({
    "figure.figsize": (7.2, 4.0),
    "figure.dpi": 120,
    "figure.facecolor": "#0e1117",
    "axes.facecolor": "#131722",
    "axes.edgecolor": "#2a3142",
    "axes.labelcolor": "#c7d0e0",
    "axes.titlecolor": "#eef2f8",
    "axes.titlesize": 12,
    "axes.titleweight": "600",
    "axes.labelsize": 10,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": "#222838",
    "grid.linewidth": 0.8,
    "text.color": "#c7d0e0",
    "xtick.color": "#8b95a9",
    "ytick.color": "#8b95a9",
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.frameon": False,
    "legend.fontsize": 9,
    "lines.linewidth": 2.0,
    "lines.markersize": 5,
    "patch.edgecolor": "none",
    "savefig.facecolor": "#0e1117",
    "savefig.bbox": "tight",
})
_mpl.rcParams["axes.prop_cycle"] = _mpl.cycler(color=[
    "#6ea8fe", "#7ee0b8", "#ffb86b", "#ff7a9c", "#c792ea",
    "#5ad1e6", "#f5d76e", "#9aa7bd",
])
"""


@dataclass
class CellOutput:
    index: int
    ok: bool = False
    stdout: str = ""
    error: str = ""              # one sentence, aimed at the reader
    traceback: str = ""          # the real thing, for the generator to fix
    figures: list[str] = field(default_factory=list)   # base64 PNG
    seconds: float = 0.0
    skipped: bool = False        # an earlier cell failed, so this never ran

    @property
    def has_output(self) -> bool:
        return bool(self.stdout.strip() or self.figures)


@dataclass
class NotebookResult:
    ok: bool
    cells: list[CellOutput] = field(default_factory=list)
    seconds: float = 0.0
    message: str = ""
    verdict: str = "pass"        # pass | fail | error | timeout | blocked

    @property
    def failed_index(self) -> int:
        for cell in self.cells:
            if not cell.ok and not cell.skipped:
                return cell.index
        return -1

    @property
    def first_error(self) -> str:
        index = self.failed_index
        return self.cells[index].error if index >= 0 else self.message

    def all_stdout(self) -> str:
        return "\n".join(c.stdout for c in self.cells if c.stdout.strip())

    def figure_count(self) -> int:
        return sum(len(c.figures) for c in self.cells)


_NOTEBOOK_HARNESS = r'''
import base64 as _b64, contextlib as _ctx, io as _io, json as _json
import os as _os, sys as _sys, time as _time, traceback as _tb, warnings

_sys.path.insert(0, _sys.argv[1])
_os.environ.setdefault("MPLBACKEND", "Agg")
_MAX_FIGURES = int(_sys.argv[3])
_MAX_BYTES = int(_sys.argv[4])

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as _plt

__STYLE__

# A convergence warning is information, not failure, and a lesson on an
# under-fitting model is entitled to produce one. They are captured into the
# cell's output rather than raised or hidden.
warnings.simplefilter("always")

_cells = _json.loads(open(_sys.argv[2], encoding="utf-8").read())
_ns = {"__name__": "__main__"}
_results = []
_drawn = 0

for _i, _src in enumerate(_cells):
    _plt.close("all")
    _buf = _io.StringIO()
    _start = _time.perf_counter()
    _error = ""
    _trace = ""
    try:
        with warnings.catch_warnings(record=True) as _caught:
            warnings.simplefilter("always")
            with _ctx.redirect_stdout(_buf), _ctx.redirect_stderr(_buf):
                exec(compile(_src, "cell %d" % (_i + 1), "exec"), _ns)
        for _w in _caught:
            _buf.write("\n[%s] %s\n" % (_w.category.__name__, _w.message))
    except BaseException as _exc:
        _error = type(_exc).__name__ + ": " + str(_exc)
        _trace = _tb.format_exc(limit=8)

    _figs = []
    for _num in _plt.get_fignums():
        if _drawn >= _MAX_FIGURES:
            break
        _png = _io.BytesIO()
        try:
            _plt.figure(_num).savefig(_png, format="png")
        except Exception:
            continue
        _raw = _png.getvalue()
        if len(_raw) <= _MAX_BYTES:
            _figs.append(_b64.b64encode(_raw).decode("ascii"))
            _drawn += 1
    _plt.close("all")

    _results.append({
        "index": _i,
        "ok": not _error,
        "stdout": _buf.getvalue(),
        "error": _error,
        "traceback": _trace,
        "figures": _figs,
        "seconds": round(_time.perf_counter() - _start, 4),
    })
    if _error:
        # Later cells depend on this one's variables, so running them would
        # only produce a cascade of NameErrors that hide the real fault.
        for _j in range(_i + 1, len(_cells)):
            _results.append({"index": _j, "ok": False, "stdout": "",
                             "error": "", "traceback": "", "figures": [],
                             "seconds": 0.0, "skipped": True})
        break

print("__NOTEBOOK__" + _json.dumps(_results))
'''


def run_notebook(cells: list[str], *, timeout: int | None = None,
                 style: bool = True) -> NotebookResult:
    """Run a lesson's cells in order, in one process, keeping state between them.

    Returns per-cell output and figures. A cell that raises stops the run: the
    cells after it are marked skipped rather than executed, because they would
    only fail on the variables the broken cell never created.
    """
    timeout = timeout or LEARNING.sandbox_timeout_seconds
    cells = [c for c in cells if isinstance(c, str) and c.strip()]
    if not cells:
        return NotebookResult(ok=True, verdict="pass", message="Nothing to run.")

    for index, source in enumerate(cells):
        allowed, reason = static_check(source)
        if not allowed:
            return NotebookResult(
                ok=False, verdict="blocked",
                message=f"Cell {index + 1} was blocked: {reason}",
                cells=[CellOutput(index=index, ok=False, error=reason)],
            )

    harness = _NOTEBOOK_HARNESS.replace("__STYLE__", PLOT_STYLE if style else "")
    tmp_dir = Path(tempfile.mkdtemp(prefix="mlm_nb_"))
    runner = tmp_dir / "runner.py"
    payload = tmp_dir / "cells.json"
    runner.write_text(harness, encoding="utf-8")
    payload.write_text(json.dumps(cells), encoding="utf-8")

    env = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONIOENCODING": "utf-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "MPLBACKEND": "Agg",
        # matplotlib and numba both want a writable cache directory, and the
        # hosted container's home is not always one. Pointing them at the temp
        # directory avoids a warning on every single run.
        "MPLCONFIGDIR": str(tmp_dir),
        "HOME": str(tmp_dir),
        "OMP_NUM_THREADS": "2",
        "OPENBLAS_NUM_THREADS": "2",
    }
    if IS_WINDOWS:
        for key in ("SYSTEMROOT", "TEMP", "TMP", "PATHEXT", "COMSPEC",
                    "USERPROFILE", "APPDATA", "LOCALAPPDATA"):
            if key in os.environ:
                env[key] = os.environ[key]

    kwargs: dict[str, Any] = {"cwd": str(tmp_dir), "env": env, "text": True,
                              "stdout": subprocess.PIPE,
                              "stderr": subprocess.PIPE}
    limiter = _limits(timeout)
    if limiter is not None:
        kwargs["preexec_fn"] = limiter

    started = time.perf_counter()
    try:
        # Not -I here: the lesson must be able to `from core.datasets import
        # load`, and isolated mode strips the path that makes it importable.
        # The path is passed explicitly as argv[1] instead of through
        # PYTHONPATH, so exactly one directory is added and it is this app's.
        proc = subprocess.run(
            [sys.executable, "-B", str(runner), str(APP_ROOT), str(payload),
             str(MAX_FIGURES), str(MAX_FIGURE_BYTES)],
            timeout=timeout, **kwargs,
        )
        stdout, stderr, returncode = proc.stdout, proc.stderr, proc.returncode
    except subprocess.TimeoutExpired:
        return NotebookResult(
            ok=False, verdict="timeout", seconds=time.perf_counter() - started,
            message=(f"The code did not finish within {timeout} seconds. On "
                     "this data that usually means a grid search with too many "
                     "combinations, a model being fitted inside a loop, or an "
                     "unbounded loop."),
        )
    except Exception as exc:                         # noqa: BLE001
        return NotebookResult(ok=False, verdict="error",
                              message=f"Could not start the sandbox: {exc}")
    finally:
        _cleanup(tmp_dir)

    elapsed = time.perf_counter() - started
    marker = "__NOTEBOOK__"
    line = next((ln for ln in stdout.splitlines() if ln.startswith(marker)), None)
    if line is None:
        return NotebookResult(
            ok=False, verdict="error", seconds=elapsed,
            message=_explain_crash(stderr, returncode),
            cells=[CellOutput(index=0, ok=False,
                              error=_explain_crash(stderr, returncode),
                              traceback=(stderr or "")[-4000:])],
        )

    outputs: list[CellOutput] = []
    for raw in json.loads(line[len(marker):]):
        outputs.append(CellOutput(
            index=int(raw["index"]),
            ok=bool(raw["ok"]),
            stdout=str(raw.get("stdout", ""))[: LEARNING.max_output_chars],
            error=friendly_error(raw.get("error", ""), raw.get("traceback", "")),
            traceback=str(raw.get("traceback", ""))[-4000:],
            figures=list(raw.get("figures", [])),
            seconds=float(raw.get("seconds", 0.0)),
            skipped=bool(raw.get("skipped")),
        ))

    ok = all(c.ok or c.skipped for c in outputs) and not any(
        c.skipped for c in outputs)
    return NotebookResult(ok=ok, cells=outputs, seconds=elapsed,
                          verdict="pass" if ok else "fail")


def run_cell(code: str, *, timeout: int | None = None) -> CellOutput:
    """One cell, for the scratchpad and the Code Lab."""
    result = run_notebook([code], timeout=timeout)
    if result.cells:
        return result.cells[0]
    return CellOutput(index=0, ok=result.ok, error=result.message)


def figure_data_uri(encoded: str) -> str:
    return f"data:image/png;base64,{encoded}"


def figure_bytes(encoded: str) -> bytes:
    return base64.b64decode(encoded)


# ---------------------------------------------------------------------------
# Errors a learner will actually hit
# ---------------------------------------------------------------------------
# The generic table above covers Python's own mistakes. These are the ones
# scikit-learn and pandas produce, which say what went wrong without ever
# saying what to do about it. Each pattern maps to the fix, in the order they
# are most often the answer.

_ML_ERRORS: list[tuple[str, str]] = [
    (r"Input (?:X )?contains NaN",
     "The model was given missing values. Every estimator in scikit-learn "
     "refuses them, so impute or drop them before fitting — inside the "
     "pipeline, not before the split."),
    (r"could not convert string to float",
     "A text column reached a model that only takes numbers. Encode the "
     "categorical columns first."),
    (r"Found input variables with inconsistent numbers of samples",
     "X and y have different lengths. Something dropped rows from one and not "
     "the other — usually a dropna applied to X alone."),
    (r"(?:is not fitted|NotFittedError)",
     "The estimator was used before .fit() was called on it."),
    (r"Expected 2D array, got 1D array",
     "A single feature needs reshaping: X[['column']] rather than "
     "X['column'], or .values.reshape(-1, 1)."),
    (r"Unknown label type",
     "A regression target was handed to a classifier, or the labels are "
     "floats where they should be classes."),
    (r"The number of classes has to be greater than one",
     "The training split contains only one class — usually a stratify that "
     "was left off, or a filter that removed every positive case."),
    (r"Found unknown categories",
     "The encoder met a category at test time that was absent at train time. "
     "handle_unknown='ignore' is the usual answer, and the rare 'satellite' "
     "level in the churn data is why this lesson exists."),
    (r"X has \d+ features, but .* is expecting \d+ features",
     "The columns at predict time are not the columns at fit time. Fitting "
     "the preprocessing inside a pipeline keeps the two in step."),
    (r"KeyError: '([^']+)'",
     "There is no column by that name. Check the spelling against "
     "df.columns — the bundled datasets use lower_snake_case."),
    (r"(?:MemoryError|Unable to allocate)",
     "The code asked for more memory than the sandbox allows. A pairwise "
     "distance matrix or a one-hot encoding of a high-cardinality column is "
     "usually the cause."),
    (r"ConvergenceWarning",
     "The solver stopped before converging. Scaling the features, or raising "
     "max_iter, usually fixes it."),
    (r"(?:No module named|ModuleNotFoundError)",
     "That library is not available here. This course uses numpy, pandas, "
     "scikit-learn, matplotlib and seaborn only."),
    (r"No dataset called",
     "That dataset name does not exist. Call core.datasets.available() to see "
     "the bundled ones."),
]


def friendly_error(error: str, traceback_text: str = "") -> str:
    """One sentence a learner can act on, with the original kept underneath."""
    if not error:
        return ""
    haystack = f"{error}\n{traceback_text}"
    for pattern, advice in _ML_ERRORS:
        if re.search(pattern, haystack):
            return f"{advice}\n\n{error}"
    generic = _explain_crash(haystack, 1)
    if generic and generic.strip() != error.strip():
        return generic
    return error
