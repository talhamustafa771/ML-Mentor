"""Projects, milestones and the capstone.

Seven weekly projects, five milestones at the three-week marks, and a five-day
capstone at the end. Each one asks you to do a whole thing rather than a step
of one — which is the only way to find out whether the week actually landed.

A brief is generated once and kept, so you can come back to it. Your notes and
code are saved against the project, so the work survives closing the tab.
"""
from __future__ import annotations

import streamlit as st

from core import db, gamify, llm, sandbox, ui
from core.config import LEARNING

settings = ui.page("Projects")

topics = [t for t in db.topics_with_mastery()
          if t.get("kind") in ("project", "milestone", "capstone")]
statuses = db.project_statuses()
done = sum(1 for t in topics if statuses.get(t["slug"]) == "done")

ui.hero("Build something",
        "A project at the end of every week, a milestone every three, and a "
        "capstone at the end. Reading about a pipeline and building one that "
        "does not leak are different skills.",
        eyebrow=f"{done} of {len(topics)} finished",
        ring=(done / max(1, len(topics)), "finished"))
st.write("")

if not topics:
    st.info("The plan has not been seeded yet.")
    st.stop()


def label(slug: str) -> str:
    topic = next(t for t in topics if t["slug"] == slug)
    mark = {"done": "✓ ", "in_progress": "· "}.get(statuses.get(slug), "")
    return f"{mark}Day {topic['day']:>2} · {topic['title']}"


chosen = st.selectbox("Which one", [t["slug"] for t in topics],
                      format_func=label)
topic = next(t for t in topics if t["slug"] == chosen)
work = db.get_project(topic["slug"])

ui.topic_header(topic)
st.write("")

KIND_NOTE = {
    "project": "A week's work, drawn together.",
    "milestone": "Three weeks' work. Longer, and less scaffolded.",
    "capstone": "The end of the course. This is the one you show people.",
}
st.caption(KIND_NOTE.get(topic.get("kind", ""), ""))

BRIEF_SYSTEM = (
    "You set applied machine learning projects for a student who has reached "
    "this point in a 91-day course. Your briefs are specific, achievable in "
    "the time given, and judged by something checkable. You never ask for a "
    "dataset the student does not have."
)


def brief_prompt() -> str:
    from core import concepts, datasets                 # noqa: PLC0415

    return f"""{topic['title']} — {topic['summary']}

This is a {topic.get('kind')} on day {topic['day']} of 91, worth about
{topic['minutes']} minutes. The student must be able to finish it with the
bundled datasets: {', '.join(datasets.available())}, loaded with
`from core.datasets import load`. Nothing may be downloaded.

What they are meant to demonstrate:
{chr(10).join('  - ' + o for o in topic['objectives'])}

{concepts.horizon(topic)}

Write the brief as JSON:
{{
  "goal": "one paragraph: what they are building and for whom",
  "dataset": "which bundled dataset",
  "steps": ["five to nine concrete steps, in order"],
  "deliverable": "what exists at the end, precisely",
  "done_when": ["three to five checkable conditions — a number reached, a
                 plot produced, a claim demonstrated. Each must be something
                 they can verify themselves rather than a matter of taste."],
  "stretch": ["two optional extensions for anyone with time left"],
  "watch_out_for": ["two or three specific ways this goes wrong"]
}}"""


stored = db.list_content(topic["slug"], "project", limit=1)
brief = stored[0]["body"] if stored else None

if brief is None:
    st.info("No brief has been written for this project yet.")
    if st.button("Write the brief", type="primary", use_container_width=True):
        if not ui.require_provider(settings):
            st.stop()
        with st.spinner("Writing…"):
            response = llm.complete_json(BRIEF_SYSTEM, brief_prompt(),
                                         settings=settings, tier="strong",
                                         max_tokens=2500)
        if not response.ok or not isinstance(response.data, dict):
            ui.generation_error(response.error, what="the brief")
        else:
            db.save_content(f"project-{topic['slug']}", topic["slug"],
                            "project", topic["difficulty"], response.data,
                            response.provider, response.model, replace=True)
            st.rerun()
    st.stop()

# ---------------------------------------------------------------------------
st.markdown(f'<div class="glass rise" style="font-size:1.04rem;'
            f'line-height:1.7">{brief.get("goal", "")}</div>',
            unsafe_allow_html=True)
st.write("")

left, right = st.columns([3, 2])
with left:
    st.markdown("### Steps")
    for index, step in enumerate(brief.get("steps", []), start=1):
        st.markdown(f"{index}. {step}")

    st.markdown("### Done when")
    for condition in brief.get("done_when", []):
        st.markdown(f"- {condition}")

with right:
    if brief.get("dataset"):
        ui.card("Dataset", str(brief["dataset"]), tone="accent")
    st.markdown("**Deliverable**")
    st.markdown(brief.get("deliverable", ""))

    watch = brief.get("watch_out_for", [])
    if watch:
        st.markdown("**Watch out for**")
        for item in watch:
            st.markdown(f'<div class="trap">{item}</div>',
                        unsafe_allow_html=True)

    stretch = brief.get("stretch", [])
    if stretch:
        with st.expander("If you have time left"):
            for item in stretch:
                st.markdown(f"- {item}")


# ---------------------------------------------------------------------------
st.write("")
st.markdown("---")
ui.section("Your work", "Saved against this project, so it survives closing "
                        "the tab.")

code = st.text_area("Code", value=work.get("code") or "", height=320,
                    key=f"project-code-{topic['slug']}")
notes = st.text_area("Notes — what you tried, what surprised you",
                     value=work.get("notes") or "", height=140,
                     key=f"project-notes-{topic['slug']}")

save, run, finish = st.columns([1, 1, 1])
with save:
    if st.button("Save", use_container_width=True):
        db.save_project(topic["slug"], code=code, notes=notes,
                        status=work.get("status") or "in_progress")
        st.success("Saved.")
with run:
    if st.button("Run it", type="primary", use_container_width=True):
        if not code.strip():
            st.warning("Nothing to run yet.")
        else:
            db.save_project(topic["slug"], code=code, notes=notes,
                            status="in_progress")
            with st.spinner("Running…"):
                result = sandbox.run_notebook(
                    [code], timeout=LEARNING.sandbox_timeout_seconds * 2)
            st.session_state[f"project-run-{topic['slug']}"] = result
            st.rerun()
with finish:
    already = statuses.get(topic["slug"]) == "done"
    if st.button("Mark it finished", use_container_width=True,
                 disabled=already):
        db.save_project(topic["slug"], code=code, notes=notes, status="done")
        kind = topic.get("kind", "project")
        earned = gamify.award(kind, reason=f"Finished {topic['title']}",
                              topic_slug=topic["slug"],
                              dedupe_key=f"{kind}:{topic['slug']}")
        st.success(f"Finished." + (f" +{earned} XP." if earned else ""))
        st.balloons()
        st.rerun()

result = st.session_state.get(f"project-run-{topic['slug']}")
if result is not None:
    for cell in result.cells:
        if cell.stdout.strip():
            st.code(cell.stdout.rstrip(), language="text")
        for figure in cell.figures:
            ui.figure(figure)
        if cell.error:
            st.error(cell.error)
    if result.ok:
        st.success(f"Ran in {result.seconds:.1f}s.")


# ---------------------------------------------------------------------------
st.write("")
st.markdown("---")
ui.section("All of them")
columns = st.columns(3)
for index, item in enumerate(topics):
    with columns[index % 3]:
        status = statuses.get(item["slug"], "not_started")
        tone = {"done": "good", "in_progress": "accent"}.get(status, "")
        ui.card(f"Day {item['day']} · {item['title']}",
                item["summary"],
                meta=f"{item.get('kind', 'project').title()} · "
                     f"{status.replace('_', ' ')}",
                tone=tone)
