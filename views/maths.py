"""The Math Helper.

You said maths was the weak point and asked to be able to ask directly, about
any symbol or formula, at any moment. This is that, as a page of its own.

Three layers, in this order. Seventy-odd terms are written into the app with a
formula, a worked example using small numbers, and a sentence on why the course
needs them — those answer instantly, cost nothing, need no API key, and cannot
be invented. A symbol glossary names the squiggles, because most of the time
the blocker is that nobody ever said what the shape is called. Anything else
goes to the model under a prompt that forbids explaining one unfamiliar thing
with another, and the answer is cached so a term is only ever paid for once.
"""
from __future__ import annotations

import streamlit as st

from core import concepts, db, gamify, mathpad, scheduler, ui

settings = ui.page("Maths")

coverage = mathpad.coverage()
sizes = mathpad.bank_size()

ui.hero("Ask about any of it",
        "A symbol, a formula, a word in a lesson you did not follow. There is "
        "no such thing as too basic a question here — the whole reason this "
        "page exists is that most maths teaching assumes a step it never "
        "showed you.",
        eyebrow=f"{sizes['entries']} terms written in · "
                f"{sizes['symbols']} symbols named")
st.write("")

current = scheduler.next_topic()
day = current["day"] if current else 91

question = st.text_input(
    "What do you want explained?",
    placeholder="sigma · argmin · why we log the target · R squared · "
                "what does the hat on y-hat mean",
    key="maths-page-ask")

asked = None
ask, clear = st.columns([3, 1])
with ask:
    if st.button("Explain it", type="primary", use_container_width=True) \
            and question.strip():
        asked = question
with clear:
    if st.button("Clear", use_container_width=True):
        st.session_state.pop("maths-page-ask", None)
        st.rerun()


# ---------------------------------------------------------------------------
# Today's terms
# ---------------------------------------------------------------------------
if current and not asked:
    st.write("")
    st.markdown(f"**Today is day {day}: {current['title']}.** These are the "
                "terms it leans on.")
    picked = ui.maths_chips(current, key_prefix="maths-page-")
    if picked:
        asked = picked


# ---------------------------------------------------------------------------
# The answer
# ---------------------------------------------------------------------------
if asked:
    st.write("")
    with st.spinner("Looking it up…"):
        explanation = mathpad.explain(asked, topic=current, settings=settings)

    if not explanation.ok:
        st.warning(explanation.error
                   or "That is not in the written glossary, and no AI provider "
                      "is configured to explain it. Open Settings to add a "
                      "free key.")
    else:
        st.markdown(f"## {explanation.title}")
        st.markdown(f'<p style="font-size:1.06rem">{explanation.plain}</p>',
                    unsafe_allow_html=True)

        if explanation.formula:
            st.latex(explanation.formula)
        if explanation.reads_as:
            st.markdown(
                f'<div style="color:var(--ink-3);font-size:.92rem;'
                f'margin:-6px 0 14px">Said out loud: '
                f'{explanation.reads_as}</div>', unsafe_allow_html=True)

        if explanation.symbols:
            st.markdown("**The symbols in it**")
            columns = st.columns(min(4, len(explanation.symbols)))
            for column, symbol in zip(columns, explanation.symbols):
                with column:
                    st.markdown(
                        f'<div class="card"><h4 style="font-size:1.3rem">'
                        f'{symbol.get("glyph", "")}</h4>'
                        f'<p style="font-size:.84rem"><b>'
                        f'{symbol.get("name", "")}</b><br>'
                        f'{symbol.get("says", "")}</p></div>',
                        unsafe_allow_html=True)
            st.write("")

        if explanation.worked:
            st.markdown(
                f'<div class="says" style="font-size:1rem"><b>Worked through, '
                f'with numbers you can check</b><br><br>'
                f'{explanation.worked}</div>', unsafe_allow_html=True)

        if explanation.why:
            st.write("")
            st.markdown(f"**Why this course needs it.** {explanation.why}")

        ladder = explanation.ladder
        if ladder:
            st.write("")
            st.markdown("**If that did not land, these come first**")
            taught = [step for step in ladder if step["already_taught"]][:4]
            for step in taught:
                st.markdown(f"- {step['label']} — day {step['day']}")
            if not taught:
                st.caption("Nothing this rests on has been taught yet, which "
                           "is worth knowing in itself: this term is ahead of "
                           "where you are.")

        st.write("")
        st.caption("Written into the app — instant, and the same every time."
                   if explanation.source == "bank" else
                   "Explained on demand and saved, so it is instant from now on.")
        gamify.award("maths_asked", reason=f"Asked about {asked}",
                     dedupe_key=f"maths:{mathpad._key(asked)}")


# ---------------------------------------------------------------------------
# Browse
# ---------------------------------------------------------------------------
st.write("")
st.markdown("---")

tab_symbols, tab_terms, tab_history = st.tabs(
    ["Every symbol", "Everything written in", "What you have asked"])

with tab_symbols:
    st.caption("The squiggles, with their names. Most confusion about a "
               "formula turns out to be confusion about one of these.")
    columns = st.columns(3)
    for index, symbol in enumerate(mathpad.SYMBOLS):
        with columns[index % 3]:
            st.markdown(
                f'<div class="card" style="margin-bottom:10px">'
                f'<h4 style="font-size:1.35rem;font-family:ui-monospace">'
                f'{symbol.glyph}</h4>'
                f'<p style="font-size:.85rem"><b>{symbol.name}</b><br>'
                f'{symbol.says}</p></div>', unsafe_allow_html=True)

with tab_terms:
    st.caption(f"{sizes['entries']} terms are answered from the app itself, "
               f"covering {coverage['share'] * 100:.0f}% of the mathematics "
               "this course uses. The rest are explained on demand.")
    entries = {entry.key: entry for entry in mathpad.BANK.values()}
    ordered = sorted(entries.values(),
                     key=lambda e: concepts.introduced_on(e.key) or 99)
    for entry in ordered:
        arrives = concepts.introduced_on(entry.key)
        when = f"day {arrives}" if arrives else "background"
        with st.expander(f"{entry.title}  ·  {when}"):
            st.markdown(entry.plain)
            if entry.formula:
                st.latex(entry.formula)
                st.caption(f"Said out loud: {entry.reads_as}")
            st.markdown(f'<div class="says">{entry.worked}</div>',
                        unsafe_allow_html=True)
            st.caption(entry.why)

with tab_history:
    history = db.math_history()
    if not history:
        st.caption("Nothing asked yet.")
    else:
        st.caption("A term you have looked up several times is a term the "
                   "course did not explain well enough. That is a fact about "
                   "the material, not about you.")
        for row in history:
            st.markdown(f"- **{row['term']}** — asked {row['times_shown']} "
                        f"time(s), first on day {row['day']}")
