"""Repairing and splitting generated text before it is displayed.

None of this touches Streamlit, so it can be tested anywhere the app's code
runs — which matters, because the bug it exists for was a display bug and
display bugs are the easiest kind to leave untested.

The bug: a Practice question rendered as

    Which of the following will not produce output when running the file?\\n\\n```

followed by a giant `print("Hello")` headline. Two faults at once. The model
escaped its line breaks twice, so they arrived as the two characters backslash
and n rather than actual breaks, leaving a code fence that spanned nothing. And
the page wrapped the whole string — fence included — in a markdown heading, so
the code became a headline.
"""
from __future__ import annotations

import ast
import re

# A fenced block, with or without a language tag.
_FENCED = re.compile(r"```[ \t]*(?:python|py|text|output)?[ \t]*\n?(.*?)```",
                     re.DOTALL)

# A line break that arrived as two characters. A backslash-n that is genuinely
# part of a code sample arrives as \\n and is deliberately not matched.
_ESCAPED_NEWLINE = re.compile(r"(?<!\\)\\n")
_ESCAPED_TAB = re.compile(r"(?<!\\)\\t")


def unescape_newlines(text: str) -> str:
    """Repair prose whose line breaks arrived as the characters \\ and n.

    Only applied where a code fence is present. A fence spanning no lines is
    proof the escaping is wrong, whereas a lone \\n in a sentence may well be
    what the question is about — "what does the \\n escape do?" must survive
    untouched.
    """
    if "```" not in (text or ""):
        return text
    return _ESCAPED_TAB.sub("\t", _ESCAPED_NEWLINE.sub("\n", text))


def repair_code(code: str) -> str:
    """Undo escaped line breaks in a code snippet, but only when they are wrong.

    The ambiguity is real: `print("a\\nb")` is a correct one-line program whose
    backslash-n belongs to the string, while a four-line snippet flattened onto
    one line by double escaping is broken. Guessing from the text is
    unreliable, so this asks Python instead. A snippet that already parses is
    left exactly as written; it is rewritten only when unescaping is what makes
    it parse, and left alone when neither form parses.
    """
    code = code or ""
    if "\\n" not in code:
        return code
    try:
        ast.parse(code)
        return code                      # valid as-is; the \n is intentional
    except SyntaxError:
        pass
    repaired = _ESCAPED_TAB.sub("\t", _ESCAPED_NEWLINE.sub("\n", code))
    try:
        ast.parse(repaired)
        return repaired
    except SyntaxError:
        return code                      # neither parses; do not make it worse


def split_blocks(text: str) -> list[tuple[str, str]]:
    """Break generated text into ("prose"|"code", content) segments.

    The caller renders each segment for what it is. Doing the split here rather
    than handing the whole string to a markdown renderer is what stops a code
    block becoming a headline when the string is given a heading prefix.
    """
    text = unescape_newlines(text or "")
    segments: list[tuple[str, str]] = []
    position = 0

    for match in _FENCED.finditer(text):
        prose = text[position:match.start()].strip()
        if prose:
            segments.append(("prose", prose))
        code = match.group(1).strip("\n")
        if code.strip():
            segments.append(("code", code))
        position = match.end()

    tail = text[position:].strip()
    if tail:
        segments.append(("prose", tail))
    return segments
