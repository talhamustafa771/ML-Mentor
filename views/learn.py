"""The lesson.

Theory, then the picture, then code that has already been run against the real
dataset — with the output it actually produced sitting underneath it. You can
change any cell and run it again, which is the point: the code is not an
illustration of the lesson, it is the lesson.

The Math Helper sits beside it, offering every term the day uses before you
have to go looking for it.
"""
from __future__ import annotations

import streamlit as st

from core import db, gamify, generators, mathpad, sandbox, scheduler, ui
from core.config import LEARNING

settings = ui.page("Learn")
topic = ui.topic_picker("learn_topic")

ui.topic_header(topic)
st.write("")

stored = db.latest_lesson(topic["slug"])
lesson = stored["body"] if stored else None

# What the last generation on this page produced wins over what is in the
# database, because they can disagree: a lesson that failed verification is
# still shown, and a re-read would quietly hand back the previous one instead.
fresh_key = f"fresh_{topic['slug']}"
if st.session_state.get(fresh_key) is not None:
    lesson = st.session_state[fresh_key]

# ---------------------------------------------------------------------------
# Generate
# ---------------------------------------------------------------------------
left, right = st.columns([1, 1])
with left:
    wants = st.button(
        "Write today's lesson" if lesson is None else "Write a fresh lesson",
        type="primary" if lesson is None else "secondary",
        use_container_width=True)
with right:
    if lesson is not None and st.button("Re-run all the code",
                                        use_container_width=True):
        st.session_state.pop(f"run_{topic['slug']}", None)
        st.session_state[f"rerun_{topic['slug']}"] = True

if wants:
    if not ui.require_provider(settings):
        st.stop()
    progress = st.empty()
    with st.spinner("Working — this can pause partway through on a free "
                    "API tier, which is normal…"):
        lesson = generators.teach(
            topic, settings=settings, force=stored is not None,
            on_progress=lambda message: progress.info(message))
    progress.empty()
    if lesson.get("generated") is False:
        ui.generation_error(lesson.get("failure"), what="the lesson")
    else:
        st.session_state[fresh_key] = lesson
        st.session_state.pop(f"edits_{topic['slug']}", None)
        st.session_state.pop(f"run_{topic['slug']}", None)
        gamify.award("lesson_read", reason=f"Read {topic['title']}",
                     topic_slug=topic["slug"],
                     dedupe_key=f"lesson:{topic['slug']}")
        st.rerun()

if lesson is None:
    ui.objectives_block(topic)
    st.write("")
    st.info("No lesson has been written for this day yet. Press **Write "
            "today's lesson** above.")
    st.caption("It takes about twenty seconds: the model writes it, then every "
               "code cell is executed against the real dataset and the numbers "
               "in the commentary are checked against what the code printed. "
               "A lesson that fails either check is rewritten before you see "
               "it.")
    st.stop()


# ---------------------------------------------------------------------------
# The lesson
# ---------------------------------------------------------------------------
ui.verification_note(lesson)
st.write("")

if lesson.get("hook"):
    st.markdown(
        f'<div class="glass rise" style="font-size:1.04rem;line-height:1.7">'
        f'{lesson["hook"]}</div>',
        unsafe_allow_html=True)
    st.write("")

tab_lesson, tab_code, tab_traps, tab_check = st.tabs(
    ["The idea", "The code", "Traps", "Check yourself"])

with tab_lesson:
    ui.rich_text(lesson.get("theory", ""))
    if lesson.get("intuition"):
        st.markdown(
            f'<div class="says" style="margin-top:18px"><b>The picture to '
            f'keep</b><br>{lesson["intuition"]}</div>', unsafe_allow_html=True)
    if lesson.get("summary"):
        st.write("")
        st.markdown("### In short")
        st.markdown(lesson["summary"])


# ---------------------------------------------------------------------------
# The code, with its output
# ---------------------------------------------------------------------------
with tab_code:
    cells = lesson.get("cells") or []
    if not cells:
        st.info("This lesson has no code cells.")
    else:
        dataset = lesson.get("dataset") or topic.get("dataset")
        if dataset:
            st.caption(f"Everything below runs against the bundled "
                       f"**{dataset}** dataset. Change any cell and press Run "
                       "to see what happens.")

        run_key = f"run_{topic['slug']}"
        edit_key = f"edits_{topic['slug']}"
        edits = st.session_state.setdefault(
            edit_key, [str(c.get("code", "") if isinstance(c, dict) else c)
                       for c in cells])

        if st.session_state.pop(f"rerun_{topic['slug']}", False):
            with st.spinner("Running every cell…"):
                st.session_state[run_key] = sandbox.run_notebook(
                    edits, timeout=LEARNING.sandbox_timeout_seconds)

        live = st.session_state.get(run_key)
        stored_outputs = lesson.get("outputs") or []

        for index, cell in enumerate(cells):
            code = edits[index] if index < len(edits) else ""
            explain = cell.get("explain", "") if isinstance(cell, dict) else ""
            says = cell.get("says", "") if isinstance(cell, dict) else ""

            st.markdown(
                f'<div class="lesson-cell"><span class="cell-number">'
                f'{index + 1}</span><span style="color:var(--ink-2)">'
                f'{explain}</span></div>', unsafe_allow_html=True)

            edited = st.text_area(f"Cell {index + 1}", value=code,
                                  height=max(90, 24 * (code.count("\n") + 2)),
                                  key=f"cell-{topic['slug']}-{index}",
                                  label_visibility="collapsed")
            if edited != code:
                edits[index] = edited
                st.session_state[edit_key] = edits

            # Live output wins over the stored output, because if you have just
            # run it yourself that is the truth about your version of the code.
            output = None
            if live is not None and index < len(live.cells):
                output = live.cells[index]
                if output.skipped:
                    st.caption("Not run — an earlier cell failed.")
                    output = None
            elif index < len(stored_outputs):
                stored_cell = stored_outputs[index]
                output = sandbox.CellOutput(
                    index=index, ok=True,
                    stdout=stored_cell.get("stdout", ""),
                    figures=stored_cell.get("figures", []))

            if output is not None:
                if output.stdout.strip():
                    st.code(output.stdout.rstrip(), language="text")
                for figure in output.figures:
                    ui.figure(figure)
                if output.error:
                    st.error(output.error)

            if says:
                st.markdown(f'<div class="says">{says}</div>',
                            unsafe_allow_html=True)
            st.write("")

        run_now, reset = st.columns([2, 1])
        with run_now:
            if st.button("Run all cells", type="primary",
                         use_container_width=True, key="run_all"):
                with st.spinner("Running…"):
                    result = sandbox.run_notebook(
                        edits, timeout=LEARNING.sandbox_timeout_seconds)
                st.session_state[run_key] = result
                gamify.award("cell_run", reason=f"Ran the code for "
                                                f"{topic['title']}",
                             topic_slug=topic["slug"],
                             dedupe_key=f"cells:{topic['slug']}")
                db.log_attempt(topic_slug=topic["slug"], kind="cell",
                               correct=1 if result.ok else 0,
                               score=1.0 if result.ok else 0.0)
                st.rerun()
        with reset:
            if st.button("Undo my edits", use_container_width=True):
                st.session_state.pop(edit_key, None)
                st.session_state.pop(run_key, None)
                st.rerun()

        if live is not None:
            if live.ok:
                st.success(f"All {len(live.cells)} cells ran "
                           f"({live.seconds:.1f}s).")
            else:
                st.error(f"Stopped at cell {live.failed_index + 1}.")


with tab_traps:
    traps = lesson.get("traps") or []
    if not traps:
        st.caption("No traps were recorded for this day.")
    for trap in traps:
        if not isinstance(trap, dict):
            continue
        st.markdown(
            f'<div class="trap"><b>{trap.get("trap", "")}</b><br>'
            f'{trap.get("why", "")}<br><br>'
            f'<span style="color:var(--good)">Instead:</span> '
            f'{trap.get("fix", "")}</div>', unsafe_allow_html=True)


with tab_check:
    questions = lesson.get("check_yourself") or topic["objectives"]
    st.markdown("Answer these without scrolling back up. If one of them is "
                "hard, that is the part to re-read.")
    for question in questions:
        st.markdown(f"- {question}")
    st.write("")
    if st.button("I could answer all of those", use_container_width=True):
        scheduler.record_outcome(topic["slug"], 1.0)
        st.success("Recorded. Practice on the next page will tell you whether "
                   "that was true.")


# ---------------------------------------------------------------------------
# The Math Helper
# ---------------------------------------------------------------------------
st.write("")
st.markdown("---")
ui.section("Maths used today",
           "Click any term. The written ones answer instantly and cost "
           "nothing; the rest are explained on demand and then cached.")

asked = ui.maths_chips(topic, key_prefix="learn-")
typed = st.text_input("Or ask about any symbol, formula or word",
                      placeholder="sigma, argmin, why log the target, R squared…",
                      key="learn-math-ask")
if typed and st.button("Explain that", key="learn-math-go"):
    asked = typed

if asked:
    with st.spinner("Looking it up…"):
        explanation = mathpad.explain(asked, topic=topic, settings=settings)

    if not explanation.ok:
        st.warning(explanation.error or "Could not explain that.")
    else:
        st.markdown(f"### {explanation.title}")
        st.markdown(explanation.plain)
        if explanation.formula:
            st.latex(explanation.formula)
        if explanation.reads_as:
            st.caption(f"Read aloud: {explanation.reads_as}")
        if explanation.symbols:
            columns = st.columns(min(4, len(explanation.symbols)))
            for column, symbol in zip(columns, explanation.symbols):
                with column:
                    st.markdown(
                        f'<div class="card"><h4>{symbol.get("glyph", "")}</h4>'
                        f'<p style="font-size:.82rem">'
                        f'<b>{symbol.get("name", "")}</b><br>'
                        f'{symbol.get("says", "")}</p></div>',
                        unsafe_allow_html=True)
        if explanation.worked:
            st.markdown(f'<div class="says"><b>Worked through</b><br>'
                        f'{explanation.worked}</div>', unsafe_allow_html=True)
        if explanation.why:
            st.caption(explanation.why)

        ladder = [step for step in explanation.ladder if step["already_taught"]]
        if ladder:
            st.caption("This rests on: "
                       + ", ".join(step["label"] for step in ladder[:5]))
        st.caption("Written into the app" if explanation.source == "bank"
                   else "Explained on demand and saved, so it is instant next "
                        "time")
        gamify.award("maths_asked", reason=f"Asked about {asked}",
                     dedupe_key=f"maths:{asked}")
