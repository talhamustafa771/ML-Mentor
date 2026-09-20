"""Progress.

Honest numbers, all derived from what actually happened rather than stored, so
nothing here can drift away from the record underneath it.

The badges are worth a look even before you have earned any: half of them
reward the judgement this subject is about — noticing a score that is too good
to be true, beating a baseline honestly, choosing a threshold by what a mistake
costs — rather than counting days finished.
"""
from __future__ import annotations

import streamlit as st

from core import analytics, curriculum, db, gamify, scheduler, ui

settings = ui.page("Progress")

status = scheduler.plan_status()
summary = gamify.summary()

ui.hero("Where you are",
        "Everything on this page is computed from your attempts, not stored, "
        "so it cannot quietly disagree with what you did.",
        eyebrow=f"Level {summary['level']['level']} · "
                f"{summary['level']['name']}",
        ring=(status["percent_complete"] / 100, "of the plan"))
st.write("")
ui.level_bar()
st.write("")

ui.stat_row([
    {"label": "Days mastered", "value": f"{status['mastered']}/91", "grad": True},
    {"label": "Hours", "value": f"{status['hours_spent']:.1f}"},
    {"label": "Challenges", "value": f"{summary['challenges_solved']}/"
                                     f"{summary['challenges_total']}"},
    {"label": "Streak", "value": f"{summary['streak']} d"},
])

st.write("")
if status["started_on"]:
    pace = status["day_index"]
    st.caption(f"You started {status['started_on'][:10]}, which makes today "
               f"day {pace} of the plan's {status['plan_days']}. "
               + ("You are keeping up." if status["on_track"] else
                  f"You have {status['mastered']} days mastered against "
                  f"{pace} elapsed — the plan is a schedule, not a deadline, "
                  "but it is worth knowing."))


tab_plan, tab_badges, tab_skills, tab_time, tab_mistakes = st.tabs(
    ["The plan", "Badges", "Skills", "Time", "What you got wrong"])


# ---------------------------------------------------------------------------
with tab_plan:
    topics = db.topics_with_mastery()
    by_week: dict[int, list[dict]] = {}
    for topic in topics:
        by_week.setdefault(topic["week"], []).append(topic)

    shares = {}
    for week, group in by_week.items():
        mastered = sum(1 for t in group if scheduler.is_mastered(t["slug"]))
        shares[week] = mastered / len(group) if group else 0.0

    current = scheduler.next_topic()
    ui.lattice(shares, current_week=current["week"] if current else 0)

    for week in sorted(by_week):
        group = by_week[week]
        mastered = sum(1 for t in group if scheduler.is_mastered(t["slug"]))
        with st.expander(f"Week {week} — {mastered} of {len(group)} mastered",
                         expanded=(current is not None
                                   and current["week"] == week)):
            for topic in group:
                score = round((topic.get("score") or 0) * 100)
                attempts = topic.get("attempts") or 0
                bar = "█" * (score // 10) + "░" * (10 - score // 10)
                st.markdown(
                    f"`{bar}` **Day {topic['day']}** · {topic['title']} — "
                    f"{score}% over {attempts} attempt(s)")


# ---------------------------------------------------------------------------
with tab_badges:
    badges = gamify.earned_badges()
    st.caption(f"{sum(1 for b in badges if b['earned'])} of {len(badges)} "
               "earned.")
    GROUPS = {
        "judgement": ("Judgement",
                      "The ones that matter most. Each is earned by noticing "
                      "something, not by finishing something."),
        "craft": ("Craft", "Built it properly."),
        "progress": ("Progress", "Distance covered."),
        "habit": ("Habit", "Turning up, and asking."),
    }
    for group, (title, note) in GROUPS.items():
        subset = [b for b in badges if b["group"] == group]
        if not subset:
            continue
        st.markdown(f"### {title}")
        st.caption(note)
        ui.badge_wall(subset)
        st.write("")


# ---------------------------------------------------------------------------
with tab_skills:
    skills = gamify.skill_map()
    if not skills:
        st.caption("Solve some challenges and this fills in. It answers a "
                   "question topic mastery cannot: you can have read the "
                   "leakage lesson and still ship a leaking model.")
    else:
        for row in skills:
            tone = {"strong": "good", "working": "accent",
                    "started": "", "untouched": "warn"}[row["state"]]
            st.markdown(
                ui.pill(f"{row['skill'].replace('_', ' ')} — "
                        f"{row['solved']}/{row['total']}", tone),
                unsafe_allow_html=True)


# ---------------------------------------------------------------------------
with tab_time:
    series = analytics.daily_series(30)
    st.markdown("### Minutes a day, last thirty")
    st.bar_chart({"minutes": [d["minutes"] for d in series]},
                 height=200, color="#6ea8fe")

    st.markdown("### Attempts a day")
    st.bar_chart({"attempts": [d["attempts"] for d in series]},
                 height=160, color="#7ee0b8")

    total_planned = sum(t["minutes"] for t in curriculum.TOPICS)
    st.caption(f"The plan is about {total_planned / 60:.0f} hours across 91 "
               f"days. You have logged {status['hours_spent']:.1f}.")


# ---------------------------------------------------------------------------
with tab_mistakes:
    mistakes = db.recent_mistakes(25)
    if not mistakes:
        st.caption("Nothing wrong yet — or nothing attempted yet.")
    else:
        st.caption("The most recent things you got wrong. This is the most "
                   "useful list on the page.")
        for row in mistakes:
            payload = row.get("payload") or {}
            question = (payload.get("question") if isinstance(payload, dict)
                        else None)
            st.markdown(
                f'<div class="card" style="margin-bottom:8px">'
                f'<h4>{row.get("topic_title") or row.get("topic_slug", "")}</h4>'
                f'<p>{question or row.get("kind", "")}</p>'
                f'<div class="meta">{str(row.get("created_at", ""))[:16]}</div>'
                f'</div>', unsafe_allow_html=True)
