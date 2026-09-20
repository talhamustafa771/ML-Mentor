"""ML Mentor — the entry point.

Navigation is declared here rather than inferred from filenames, which buys
three things worth having: the pages get real titles instead of whatever the
file was called, they group into sections that say what each part of the app is
for, and the order is the order you would actually use them in rather than
alphabetical.

The first two statements below are in that order deliberately. Streamlit
requires `set_page_config()` to be the first Streamlit command a script runs,
and *any* Streamlit output counts — including a warning printed by a module
during import. This app once failed to start because importing `core` read
`st.secrets`, found no secrets file, and Streamlit rendered a message about it
onto the page. So the page is configured before `core` is imported at all, and
nothing here may be reordered for tidiness.
"""
from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="ML Mentor", page_icon="◈", layout="wide",
    # "auto" rather than "expanded": on a phone an expanded sidebar covers the
    # page you came to read, and this is studied on a train as much as at a
    # desk.
    initial_sidebar_state="auto",
)

from core import ui                                      # noqa: E402

ui.shell()

navigation = st.navigation({
    "Study": [
        st.Page("views/today.py", title="Today", icon=":material/today:",
                default=True),
        st.Page("views/learn.py", title="Learn", icon=":material/school:"),
        st.Page("views/practice.py", title="Practice",
                icon=":material/quiz:"),
        st.Page("views/challenges.py", title="Challenges",
                icon=":material/flag:"),
    ],
    "Work it out": [
        st.Page("views/maths.py", title="Math Helper",
                icon=":material/functions:"),
        st.Page("views/visualise.py", title="Visualise",
                icon=":material/animation:"),
        st.Page("views/lab.py", title="Lab", icon=":material/science:"),
        st.Page("views/projects.py", title="Projects",
                icon=":material/build:"),
    ],
    "You": [
        st.Page("views/progress.py", title="Progress",
                icon=":material/trending_up:"),
        st.Page("views/settings.py", title="Settings",
                icon=":material/settings:"),
    ],
})

navigation.run()
