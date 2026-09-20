"""The challenge bank.

Each one is a task against a bundled dataset. Your code runs, then a hidden
check runs in the same namespace and asserts against what you left behind.

The checks grade the method, not only the answer. Several of them assert an
upper bound on your score, because on this data a near-perfect result means a
column leaked, and a grader that congratulated you for that would be teaching
the opposite of the lesson.
"""
from __future__ import annotations

import streamlit as st

from core import db, gamify, scheduler, ui, verify

settings = ui.page("Challenges")

day_now = (scheduler.next_topic() or {"day": 91})["day"]
progress = db.bank_progress()

ui.hero("Challenges",
        "Write the code. A hidden grader checks what you found out, and tells "
        "you what went wrong rather than only that something did.",
        eyebrow=f"{progress['solved']} of {progress['total']} solved",
        ring=(progress["solved"] / max(1, progress["total"]), "solved"))
st.write("")

show_all = st.toggle("Show challenges from days I have not reached",
                     value=False,
                     help="They are hidden by default because their solutions "
                          "need tools the course has not taught you yet.")
unlocked = db.bank_challenges(max_day=None if show_all else day_now)

if not unlocked:
    st.info("No challenges are unlocked yet. They open as the course reaches "
            "the day each one needs.")
    st.stop()

left, right = st.columns([2, 1])
with left:
    difficulty = st.segmented_control(
        "Difficulty", ["All", "beginner", "intermediate", "advanced"],
        default="All") if hasattr(st, "segmented_control") else st.selectbox(
        "Difficulty", ["All", "beginner", "intermediate", "advanced"])
with right:
    hide_solved = st.toggle("Hide the ones I have solved", value=False)

shown = [c for c in unlocked
         if (difficulty in ("All", None) or c["difficulty"] == difficulty)
         and not (hide_solved and c["solved"])]

st.caption(f"{len(shown)} shown · {len(unlocked)} unlocked · "
           f"{progress['total']} in the bank")
st.write("")

labels = {c["id"]: f"{'✓ ' if c['solved'] else ''}Day {c['topic_day']:>2} · "
                   f"{c['task'].splitlines()[0][:70]}" for c in shown}
if not shown:
    st.info("Nothing matches those filters.")
    st.stop()

chosen_id = st.selectbox("Pick one", [c["id"] for c in shown],
                         format_func=lambda cid: labels[cid])
challenge = next(c for c in shown if c["id"] == chosen_id)

st.write("")
ui.pills([(challenge["difficulty"].title(),
           {"beginner": "good", "intermediate": "accent",
            "advanced": "warn"}.get(challenge["difficulty"], "")),
          (challenge["dataset"], "accent"),
          (f"unlocks day {challenge.get('unlock_day', challenge['topic_day'])}", ""),
          ("Solved", "good") if challenge["solved"] else ("Unsolved", "")])

st.write("")
st.markdown(f'<div class="glass rise" style="font-size:1.02rem;'
            f'line-height:1.7;white-space:pre-wrap">{challenge["task"]}</div>',
            unsafe_allow_html=True)

if challenge.get("setup"):
    with st.expander("This runs before your code"):
        st.code(challenge["setup"], language="python")

st.write("")
answer_key = f"answer-{challenge['id']}"
answer = st.text_area("Your code", height=260, key=answer_key,
                      placeholder="# leave the variables the task asks for\n")

hints_key = f"hints-{challenge['id']}"
hints_used = st.session_state.get(hints_key, 0)

run, hint, solution = st.columns([2, 1, 1])
with run:
    attempt = st.button("Run and check", type="primary",
                        use_container_width=True)
with hint:
    if hints_used < len(challenge["hints"]):
        if st.button(f"Hint {hints_used + 1}", use_container_width=True):
            st.session_state[hints_key] = hints_used + 1
            st.rerun()
    else:
        st.button("No hints left", disabled=True, use_container_width=True)
with solution:
    reveal = st.button("Show the answer", use_container_width=True)

for index in range(st.session_state.get(hints_key, 0)):
    st.markdown(f'<div class="says">Hint {index + 1}: '
                f'{challenge["hints"][index]}</div>', unsafe_allow_html=True)

if attempt:
    if not answer.strip():
        st.warning("Write something first.")
    else:
        with st.spinner("Running your code, then the grader…"):
            result = verify.verify_challenge(challenge, answer, timeout=120)

        for cell in result.cells:
            if cell.stdout.strip():
                st.code(cell.stdout.rstrip(), language="text")
            for figure in cell.figures:
                ui.figure(figure)

        first_solve = not db.is_solved(challenge["id"])
        db.save_submission(content_id=challenge["id"],
                           topic_slug=challenge["topic_slug"],
                           code=answer,
                           verdict="pass" if result.ok else "fail",
                           passed=1 if result.ok else 0, total=1,
                           stderr=result.first_error[:2000])
        db.log_attempt(topic_slug=challenge["topic_slug"], kind="code",
                       correct=result.ok, score=1.0 if result.ok else 0.0,
                       item_id=challenge["id"],
                       hints_used=st.session_state.get(hints_key, 0))
        scheduler.record_outcome(challenge["topic_slug"],
                                 1.0 if result.ok else 0.0, seconds=60)

        if result.ok:
            earned = gamify.award_challenge(
                challenge, hints_used=st.session_state.get(hints_key, 0),
                first_solve=first_solve)
            st.success("Solved." + (f" +{earned} XP." if earned else
                                    " (Already counted.)"))
            st.balloons()
        else:
            # The grader's own message is the teaching. It is shown whole
            # rather than summarised, because the sentence after the comma is
            # usually the entire point of the challenge.
            st.error(result.first_error or "The check did not pass.")

if reveal:
    st.markdown("**One way to do it.** Yours may differ and still be right — "
                "the grader checks what you found, not how you wrote it.")
    st.code(challenge["solution"], language="python")


# ---------------------------------------------------------------------------
st.write("")
st.markdown("---")
ui.section("Where you are strong and weak")
skills = gamify.skill_map()
if not skills:
    st.caption("Solve a few and this fills in.")
else:
    columns = st.columns(4)
    for index, row in enumerate(skills[:12]):
        with columns[index % 4]:
            tone = {"strong": "good", "working": "accent",
                    "started": "", "untouched": "warn"}[row["state"]]
            ui.card(row["skill"].replace("_", " "),
                    meta=f"{row['solved']}/{row['total']} · {row['state']}",
                    tone=tone)
