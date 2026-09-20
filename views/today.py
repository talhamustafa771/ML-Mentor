"""ML Mentor — the dashboard.

The first thing you see each day, and the only page whose job is to answer one
question: what should I do now. Everything else is one click from here.
"""
from __future__ import annotations

import streamlit as st

from core import analytics, challenges, curriculum, datasets, db, gamify
from core import scheduler, ui

settings = ui.page("Today")

status = scheduler.plan_status()
summary = gamify.summary()
next_topic = scheduler.next_topic()
day = next_topic["day"] if next_topic else 1

ui.hero(
    "Machine learning, ninety-one days",
    "Theory, a picture of the theory, and code that has been run against real "
    "data before you see it. Every day builds on the one before it and nothing "
    "arrives before you are ready for it.",
    eyebrow=f"Day {day} of 91 · Week {next_topic['week'] if next_topic else 1}",
    art="network" if status["mastered"] == 0 else "",
    ring=((status["percent_complete"] / 100, "of the plan")
          if status["mastered"] else None),
)

st.write("")
ui.level_bar()
st.write("")


# ---------------------------------------------------------------------------
# What to do now
# ---------------------------------------------------------------------------
ui.section("What to do now")

if next_topic is None:
    st.success("You have finished the plan. Every day is mastered.")
else:
    left, right = st.columns([3, 2])
    with left:
        st.markdown(f"### Day {next_topic['day']} · {next_topic['title']}")
        st.markdown(next_topic["summary"])
        chips = [(f"Week {next_topic['week']}", ""),
                 (f"{next_topic['minutes']} min", ""),
                 (next_topic["difficulty"].title(), "")]
        dataset = next_topic.get("dataset")
        if dataset and dataset in datasets.CATALOGUE:
            chips.append((f"on {dataset}", "accent"))
        if next_topic.get("kind") != "lesson":
            chips.append((next_topic["kind"].title(), "warn"))
        ui.pills(chips)
        st.write("")
        if st.button("Open today's lesson", type="primary",
                     use_container_width=True):
            st.session_state["current_topic"] = next_topic["slug"]
            st.switch_page("views/learn.py")

    with right:
        st.markdown("**Today you will be able to**")
        for objective in next_topic["objectives"][:3]:
            st.markdown(f"- {objective}")

st.write("")

ui.stat_row([
    {"label": "Days done", "value": f"{status['mastered']}/91", "grad": True,
     "sub": f"{status['percent_complete']:.0f}% of the plan"},
    {"label": "Hours in", "value": f"{status['hours_spent']:.1f}",
     "sub": f"of about {sum(t['minutes'] for t in curriculum.TOPICS) / 60:.0f} planned"},
    {"label": "Challenges", "value": f"{summary['challenges_solved']}/"
                                     f"{summary['challenges_total']}",
     "sub": f"{len(challenges.unlocked_by(day))} unlocked so far"},
    {"label": "Streak", "value": f"{summary['streak']} d",
     "sub": "days in a row"},
])


# ---------------------------------------------------------------------------
# The plan, as a board
# ---------------------------------------------------------------------------
st.write("")
ui.section("Thirteen weeks",
           "A tile lifts off the board as its week is finished. Hover to "
           "straighten it.")

topics = db.topics_with_mastery()
by_week: dict[int, list[dict]] = {}
for topic in topics:
    by_week.setdefault(topic["week"], []).append(topic)

shares = {}
for week, group in by_week.items():
    done = sum(1 for t in group if scheduler.is_mastered(t["slug"]))
    shares[week] = done / len(group) if group else 0.0

ui.lattice(shares, current_week=next_topic["week"] if next_topic else 0)

WEEK_TITLES = {
    1: "What learning from data means", 2: "Evaluation, before any model",
    3: "Cleaning and pipelines", 4: "Regression and gradient descent",
    5: "Classification", 6: "Trees, forests and boosting",
    7: "Finding structure without labels", 8: "Text, time and leakage",
    9: "Neural networks from scratch", 10: "Shipping and keeping it working",
    11: "The statistics underneath", 12: "Interviews and consolidation",
    13: "The capstone",
}
columns = st.columns(4)
for index, week in enumerate(sorted(by_week)):
    with columns[index % 4]:
        share = shares[week]
        tone = "good" if share >= 0.999 else ("accent" if share > 0 else "")
        ui.card(f"Week {week}", WEEK_TITLES.get(week, ""),
                meta=f"{round(share * 100)}% · {len(by_week[week])} days",
                tone=tone)


# ---------------------------------------------------------------------------
# Where you are weakest
# ---------------------------------------------------------------------------
weakest = scheduler.weakest_topics(3)
if weakest:
    st.write("")
    ui.section("Worth going back to",
               "Scored lowest relative to how much you have attempted.")
    for topic in weakest:
        ui.card(f"Day {topic['day']} · {topic['title']}",
                topic["summary"],
                meta=f"{round((topic.get('score') or 0) * 100)}% mastery · "
                     f"{topic.get('attempts', 0)} attempts",
                tone="warn")


# ---------------------------------------------------------------------------
# Recent
# ---------------------------------------------------------------------------
series = analytics.daily_series(14)
if any(d["minutes"] for d in series):
    st.write("")
    ui.section("The last fortnight")
    st.bar_chart({"minutes": [d["minutes"] for d in series]},
                 height=180, color="#6ea8fe")

st.write("")
st.caption(f"{len(curriculum.TOPICS)} days · "
           f"{len(datasets.available())} bundled datasets · "
           f"{len(challenges.CHALLENGES)} challenges · "
           "every lesson's code is executed before it is shown.")
