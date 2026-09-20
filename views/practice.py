"""Questions and flashcards.

The questions here test judgement rather than recall, because that is what this
subject is. "Given this confusion matrix, what would you change" is a real
question; "what does the C parameter stand for" is trivia.

When you get one wrong, the explanation is aimed at the belief that made your
answer look right, not at the correct answer restated more loudly.
"""
from __future__ import annotations

import streamlit as st

from core import db, gamify, generators, scheduler, ui
from core.config import LEARNING

settings = ui.page("Practice")
topic = ui.topic_picker("practice_topic")
ui.topic_header(topic)
st.write("")

session_key = f"practice_{topic['slug']}"
state = st.session_state.setdefault(session_key, {
    "items": None, "index": 0, "answers": {}, "explanations": {},
})

top_left, top_right = st.columns([1, 1])
with top_left:
    if st.button("Start a practice set", type="primary",
                 use_container_width=True):
        if not ui.require_provider(settings):
            st.stop()
        with st.spinner("Preparing questions…"):
            built = generators.build_session(topic, settings=settings)
        state.update({"items": built["mcqs"], "cards": built["cards"],
                      "index": 0, "answers": {}, "explanations": {}})
        ui.reset_timer(session_key)
        st.rerun()
with top_right:
    if state.get("items") and st.button("Start over", use_container_width=True):
        st.session_state.pop(session_key, None)
        st.rerun()

items = state.get("items")
if not items:
    have = db.content_count(topic["slug"], "mcq")
    st.info(f"{have} question(s) are stored for this day. Press **Start a "
            "practice set** to work through them." if have else
            "No questions have been written for this day yet. Press **Start a "
            "practice set** — they are generated once and then reused.")
    st.stop()


tab_questions, tab_cards = st.tabs(["Questions", "Flashcards"])

# ---------------------------------------------------------------------------
# Questions
# ---------------------------------------------------------------------------
with tab_questions:
    answered = len(state["answers"])
    st.progress(answered / len(items) if items else 0.0,
                text=f"{answered} of {len(items)} answered")

    for index, item in enumerate(items):
        st.markdown(f"#### {index + 1}. {item['question']}")
        if item.get("code"):
            st.code(ui.repair_code(str(item["code"])), language="python")

        chosen_key = f"{session_key}-q{index}"
        already = state["answers"].get(index)

        if already is None:
            choice = st.radio("Choose one", item["options"], index=None,
                              key=chosen_key, label_visibility="collapsed")
            if choice is not None:
                picked = item["options"].index(choice)
                correct = picked == item["correct_index"]
                state["answers"][index] = picked
                scheduler.record_outcome(topic["slug"], 1.0 if correct else 0.0,
                                         seconds=10)
                db.log_attempt(topic_slug=topic["slug"], kind="mcq",
                               correct=correct, score=1.0 if correct else 0.0,
                               item_id=item.get("id"))
                if correct:
                    gamify.award("mcq_correct", topic_slug=topic["slug"],
                                 dedupe_key=f"mcq:{item.get('id')}")
                st.rerun()
        else:
            right = item["correct_index"]
            for position, option in enumerate(item["options"]):
                if position == right:
                    st.markdown(f"- **{option}** ✓")
                elif position == already:
                    st.markdown(f"- ~~{option}~~ — what you chose")
                else:
                    st.markdown(f"- {option}")

            if already == right:
                st.success(item["explanation"])
            else:
                st.error(item["explanation"])
                explanation_key = f"why-{index}"
                if explanation_key not in state["explanations"]:
                    if st.button("Why did my answer look right?",
                                 key=f"{session_key}-why{index}"):
                        with st.spinner("Thinking about it…"):
                            state["explanations"][explanation_key] = \
                                generators.explain_mistake(
                                    topic, item["question"],
                                    item["options"][already],
                                    item["options"][right], settings=settings)
                        st.rerun()
                else:
                    st.markdown(
                        f'<div class="says">'
                        f'{state["explanations"][explanation_key]}</div>',
                        unsafe_allow_html=True)
        st.write("")

    if len(state["answers"]) == len(items):
        correct_count = sum(1 for index, picked in state["answers"].items()
                            if picked == items[index]["correct_index"])
        share = correct_count / len(items)
        st.markdown("---")
        ui.stat_row([
            {"label": "Score", "value": f"{correct_count}/{len(items)}",
             "grad": True},
            {"label": "Share", "value": f"{share * 100:.0f}%"},
            {"label": "Minutes", "value": f"{ui.session_timer(session_key) // 60}"},
        ])
        if share >= 0.999:
            st.balloons()
            db.end_session(db.start_session(topic["slug"], "practice"),
                           ui.session_timer(session_key), notes="perfect")
        elif share < LEARNING.mastery_threshold:
            st.warning("Below the mastery bar. Re-read the lesson's traps "
                       "before moving on — that is usually where the gap is.")


# ---------------------------------------------------------------------------
# Flashcards
# ---------------------------------------------------------------------------
with tab_cards:
    cards = state.get("cards") or []
    if not cards:
        st.caption("No flashcards for this day yet.")
    else:
        position = st.session_state.setdefault(f"{session_key}-card", 0)
        position = min(position, len(cards) - 1)
        card = cards[position]

        st.caption(f"Card {position + 1} of {len(cards)}")
        st.markdown(
            f'<div class="glass rise" style="min-height:120px;display:grid;'
            f'place-items:center;text-align:center;font-size:1.1rem">'
            f'{card["front"]}</div>', unsafe_allow_html=True)
        st.write("")

        show_key = f"{session_key}-show{position}"
        if st.session_state.get(show_key):
            st.markdown(f'<div class="says">{card["back"]}</div>',
                        unsafe_allow_html=True)
            st.write("")
            grades = st.columns(4)
            labels = [("Again", 0), ("Hard", 3), ("Good", 4), ("Easy", 5)]
            for column, (label, grade) in zip(grades, labels):
                with column:
                    if st.button(label, key=f"{session_key}-g{position}-{grade}",
                                 use_container_width=True):
                        scheduler.schedule_review(card.get("id", ""),
                                                  topic["slug"], grade)
                        db.log_attempt(topic_slug=topic["slug"], kind="card",
                                       correct=grade >= 3,
                                       score=grade / 5, item_id=card.get("id"))
                        gamify.award("card_reviewed", topic_slug=topic["slug"],
                                     dedupe_key=f"card:{card.get('id')}:{grade}")
                        st.session_state[f"{session_key}-card"] = \
                            (position + 1) % len(cards)
                        st.session_state[show_key] = False
                        st.rerun()
        else:
            if st.button("Show the answer", type="primary",
                         use_container_width=True,
                         key=f"{session_key}-reveal{position}"):
                st.session_state[show_key] = True
                st.rerun()
