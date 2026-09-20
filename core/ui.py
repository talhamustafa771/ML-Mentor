"""The look of the application, and the pieces every page is built from.

The brief was that this should feel more professional and more interesting than
the Python mentor, with depth and motion rather than a flat page of widgets. So
the visual system here is built on four ideas, each of which is doing a job
rather than decorating.

**Depth.** Everything sits on one of four planes: the drifting aurora at the
back, the panel glass, the raised card, and the floating accent. Shadow and
blur follow from which plane a thing is on, so the page reads as layers rather
than as boxes. A card lifts and tilts under the pointer because a surface that
responds to you is a surface you believe you can touch.

**Motion with meaning.** Nothing moves for its own sake. Content rises in the
order it should be read. A progress ring fills from zero so you see the
distance travelled, not just the number. The lattice tile for a finished week
lifts off the board. Everything is suspended under prefers-reduced-motion,
because motion that cannot be turned off is an accessibility failure and not a
flourish.

**One accent, used sparingly.** Blue through mint is the through-line; violet
marks the future and amber marks a warning. Colour never carries meaning on its
own — every state that is coloured is also labelled, so it survives being read
by someone who cannot separate the two.

**It has to work on a phone.** This is studied on a train as much as at a desk.
Single column under 720px, 44px touch targets, a 16px gutter, and no horizontal
scroll anywhere.

Everything is inline CSS and inline SVG. No CDN, no web font, no JavaScript
framework: a hosted Streamlit app blocks most external requests, and a page
that depends on one is a page that renders unstyled at the worst moment.
"""
from __future__ import annotations

import html as _html
import inspect as _inspect
import os as _os
import time
from typing import Any

import streamlit as st

from . import concepts, curriculum, db, gamify, scheduler

APP_NAME = "ML Mentor"
APP_TAGLINE = "Ninety-one days from what a model is to how one is built"


# ===========================================================================
# The stylesheet
# ===========================================================================

THEME_CSS = """
<style>
:root {
  --bg:            #070910;
  --bg-2:          #0b0f1a;
  --panel:         rgba(20, 26, 41, 0.72);
  --panel-solid:   #141a29;
  --raised:        rgba(28, 36, 56, 0.86);
  --line:          rgba(122, 146, 194, 0.16);
  --line-strong:   rgba(122, 146, 194, 0.30);

  --ink:           #eef2f8;
  --ink-2:         #c2cbdc;
  --ink-3:         #8b95a9;
  --ink-4:         #5f6b80;

  --accent:        #6ea8fe;
  --accent-2:      #7ee0b8;
  --accent-3:      #c792ea;
  --warn:          #ffb86b;
  --bad:           #ff7a9c;
  --good:          #7ee0b8;

  --r-sm: 10px;
  --r:    16px;
  --r-lg: 22px;

  --shadow-1: 0 1px 2px rgba(0,0,0,.4);
  --shadow-2: 0 8px 24px -8px rgba(0,0,0,.55);
  --shadow-3: 0 24px 60px -20px rgba(0,0,0,.75);
  --glow:     0 0 0 1px rgba(110,168,254,.25), 0 8px 40px -12px rgba(110,168,254,.35);

  --space-1: 6px;  --space-2: 10px; --space-3: 16px;
  --space-4: 24px; --space-5: 36px; --space-6: 56px;
}

/* ---------------------------------------------------------------- shell -- */
.stApp {
  background: var(--bg);
  color: var(--ink);
}

/* The aurora. Three slow-drifting radial gradients on a fixed layer behind
   everything, so the page has atmosphere without a single image request. */
.stApp::before {
  content: "";
  position: fixed;
  inset: -20vmax;
  z-index: 0;
  pointer-events: none;
  background:
    radial-gradient(38vmax 30vmax at 12% 8%,  rgba(110,168,254,.20), transparent 62%),
    radial-gradient(34vmax 28vmax at 88% 6%,  rgba(199,146,234,.16), transparent 60%),
    radial-gradient(42vmax 34vmax at 70% 88%, rgba(126,224,184,.13), transparent 64%),
    radial-gradient(30vmax 26vmax at 20% 82%, rgba(110,168,254,.10), transparent 60%);
  filter: blur(6px) saturate(115%);
  animation: aurora 38s ease-in-out infinite alternate;
}
@keyframes aurora {
  0%   { transform: translate3d(0,0,0)        scale(1);    }
  33%  { transform: translate3d(2.5%,-2%,0)   scale(1.06); }
  66%  { transform: translate3d(-2%,2.5%,0)   scale(1.02); }
  100% { transform: translate3d(1.5%,1.5%,0)  scale(1.08); }
}

/* A faint grid, to give the aurora something to sit against. */
.stApp::after {
  content: "";
  position: fixed;
  inset: 0;
  z-index: 0;
  pointer-events: none;
  background-image:
    linear-gradient(rgba(122,146,194,.045) 1px, transparent 1px),
    linear-gradient(90deg, rgba(122,146,194,.045) 1px, transparent 1px);
  background-size: 56px 56px;
  mask-image: radial-gradient(ellipse 120% 80% at 50% 0%, #000 35%, transparent 78%);
  -webkit-mask-image: radial-gradient(ellipse 120% 80% at 50% 0%, #000 35%, transparent 78%);
}

.block-container {
  position: relative;
  z-index: 1;
  padding-top: 1.6rem !important;
  padding-bottom: 5rem !important;
  max-width: 1280px;
}

/* --------------------------------------------------------- typography -- */
html, body, [class*="css"], .stMarkdown, .stApp {
  font-family: ui-sans-serif, -apple-system, "Segoe UI", Roboto,
               "Helvetica Neue", Arial, sans-serif;
  font-feature-settings: "cv02","cv03","cv04","ss01";
}
h1, h2, h3, h4 { color: var(--ink); letter-spacing: -0.018em; }
h1 { font-weight: 700; font-size: 2.05rem; line-height: 1.16; }
h2 { font-weight: 650; font-size: 1.42rem; margin-top: 1.9rem; }
h3 { font-weight: 620; font-size: 1.12rem; }
p, li { color: var(--ink-2); line-height: 1.68; font-size: 1rem; }
a { color: var(--accent); text-decoration-color: rgba(110,168,254,.4); }
code, kbd, pre, .stCode { font-family: ui-monospace, "SF Mono", Menlo,
                          "Cascadia Code", Consolas, monospace; }
:not(pre) > code {
  background: rgba(110,168,254,.10);
  border: 1px solid rgba(110,168,254,.20);
  border-radius: 6px;
  padding: 1px 6px;
  font-size: .88em;
  color: #a9c9ff;
}

/* ------------------------------------------------------------- panels -- */
.glass {
  position: relative;
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: var(--r);
  box-shadow: var(--shadow-2);
  backdrop-filter: blur(18px) saturate(140%);
  -webkit-backdrop-filter: blur(18px) saturate(140%);
  padding: var(--space-4);
  overflow: hidden;
}
/* A one-pixel gradient hairline along the top edge. It is what makes a panel
   look lit from above rather than merely outlined. */
.glass::before {
  content: "";
  position: absolute; inset: 0 0 auto 0; height: 1px;
  background: linear-gradient(90deg, transparent,
              rgba(110,168,254,.55) 22%, rgba(126,224,184,.45) 58%, transparent);
  opacity: .8;
}

/* ---------------------------------------------------------------- hero -- */
.hero {
  position: relative;
  border-radius: var(--r-lg);
  border: 1px solid var(--line);
  background:
    linear-gradient(135deg, rgba(110,168,254,.16), rgba(199,146,234,.10) 44%,
                    rgba(126,224,184,.12)),
    var(--panel-solid);
  box-shadow: var(--shadow-3);
  padding: var(--space-5) var(--space-5) var(--space-4);
  overflow: hidden;
  isolation: isolate;
}
.hero::after {
  content: "";
  position: absolute; inset: -40% -10% auto -10%; height: 180%;
  background: conic-gradient(from 180deg at 50% 50%,
              rgba(110,168,254,.10), rgba(199,146,234,.10),
              rgba(126,224,184,.10), rgba(110,168,254,.10));
  animation: spin 26s linear infinite;
  z-index: -1;
  filter: blur(40px);
}
@keyframes spin { to { transform: rotate(360deg); } }

.hero-eyebrow {
  display: inline-flex; align-items: center; gap: 8px;
  font-size: .74rem; font-weight: 680; letter-spacing: .13em;
  text-transform: uppercase; color: var(--accent-2);
  background: rgba(126,224,184,.10);
  border: 1px solid rgba(126,224,184,.24);
  border-radius: 999px; padding: 5px 12px; margin-bottom: 14px;
}
.hero h1 { margin: 0 0 8px; font-size: 2.3rem; }
.hero p  { margin: 0; color: var(--ink-2); font-size: 1.05rem; max-width: 62ch; }
.hero-grid {
  display: grid; grid-template-columns: 1fr auto;
  gap: var(--space-4); align-items: center;
}

/* --------------------------------------------------------------- cards -- */
.card {
  position: relative;
  background: var(--raised);
  border: 1px solid var(--line);
  border-radius: var(--r);
  padding: var(--space-3) var(--space-4);
  box-shadow: var(--shadow-2);
  transition: transform .34s cubic-bezier(.2,.7,.3,1),
              box-shadow .34s cubic-bezier(.2,.7,.3,1),
              border-color .34s ease;
  transform-style: preserve-3d;
  will-change: transform;
  height: 100%;
}
.card:hover {
  transform: perspective(900px) translateY(-4px) rotateX(2.2deg) rotateY(-2.2deg);
  box-shadow: var(--shadow-3), var(--glow);
  border-color: var(--line-strong);
}
.card h4 { margin: 0 0 6px; font-size: 1rem; font-weight: 640; }
.card p  { margin: 0; font-size: .92rem; color: var(--ink-2); }
.card .meta {
  margin-top: 10px; font-size: .78rem; color: var(--ink-3);
  letter-spacing: .02em;
}
.card.accent { border-color: rgba(110,168,254,.35); }
.card.good   { border-color: rgba(126,224,184,.32); }
.card.warn   { border-color: rgba(255,184,107,.32); }

/* -------------------------------------------------------------- stats -- */
.stat-row { display: grid; gap: var(--space-2); }
.stat {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: var(--r);
  padding: 14px 16px;
  backdrop-filter: blur(14px);
  -webkit-backdrop-filter: blur(14px);
  transition: transform .3s cubic-bezier(.2,.7,.3,1), border-color .3s ease;
}
.stat:hover { transform: translateY(-2px); border-color: var(--line-strong); }
.stat .label {
  font-size: .70rem; letter-spacing: .12em; text-transform: uppercase;
  color: var(--ink-3); font-weight: 650;
}
.stat .value {
  font-size: 1.62rem; font-weight: 700; color: var(--ink);
  line-height: 1.2; margin-top: 4px;
  font-variant-numeric: tabular-nums;
}
.stat .value.grad {
  background: linear-gradient(120deg, var(--accent), var(--accent-2));
  -webkit-background-clip: text; background-clip: text;
  -webkit-text-fill-color: transparent;
}
.stat .sub { font-size: .78rem; color: var(--ink-3); margin-top: 2px; }

/* --------------------------------------------------------------- pills -- */
.pill {
  display: inline-flex; align-items: center; gap: 6px;
  font-size: .74rem; font-weight: 640; letter-spacing: .02em;
  border-radius: 999px; padding: 4px 11px;
  border: 1px solid var(--line); background: rgba(122,146,194,.08);
  color: var(--ink-2); white-space: nowrap;
}
.pill.good { color: var(--good); border-color: rgba(126,224,184,.35);
             background: rgba(126,224,184,.10); }
.pill.warn { color: var(--warn); border-color: rgba(255,184,107,.35);
             background: rgba(255,184,107,.10); }
.pill.bad  { color: var(--bad);  border-color: rgba(255,122,156,.35);
             background: rgba(255,122,156,.10); }
.pill.accent { color: var(--accent); border-color: rgba(110,168,254,.35);
               background: rgba(110,168,254,.10); }
.pill .dot { width: 6px; height: 6px; border-radius: 50%;
             background: currentColor; }

/* ---------------------------------------------------------- the lattice -- */
/* Thirteen weeks as an isometric board, five to a row. A finished week's tile
   lifts off the surface, so progress is something you can see at a glance from
   across the room rather than a number you have to read. Laid out as a grid
   rather than a single row on purpose: thirteen tiles in a line, tilted, reads
   as a diagonal streak; three rows of five reads as a board. */
.lattice-wrap {
  perspective: 1200px;
  height: 260px;
  display: grid;
  place-items: center;
  margin: 8px 0 12px;
}
.lattice {
  display: grid;
  grid-template-columns: repeat(5, 46px);
  gap: 10px;
  transform: rotateX(54deg) rotateZ(-42deg);
  transform-style: preserve-3d;
  transition: transform .8s cubic-bezier(.2,.7,.3,1);
}
.lattice-wrap:hover .lattice { transform: rotateX(46deg) rotateZ(-36deg) scale(1.04); }
.tile {
  width: 46px; height: 46px;
  border-radius: 5px;
  background: rgba(122,146,194,.10);
  border: 1px solid var(--line);
  transform: translateZ(0);
  transition: transform .5s cubic-bezier(.2,.7,.3,1), background .5s ease;
}
.tile.part { background: linear-gradient(180deg, rgba(110,168,254,.55),
                                         rgba(110,168,254,.20));
             transform: translateZ(10px); box-shadow: 0 6px 16px rgba(0,0,0,.55); }
.tile.done { background: linear-gradient(180deg, var(--accent-2),
                                         rgba(126,224,184,.40));
             transform: translateZ(26px); box-shadow: 0 16px 30px rgba(0,0,0,.6); }
.tile.now  { background: linear-gradient(180deg, var(--accent-3),
                                         rgba(199,146,234,.35));
             transform: translateZ(16px);
             box-shadow: 0 12px 26px rgba(0,0,0,.55);
             animation: bob 2.8s ease-in-out infinite; }
@keyframes bob {
  0%,100% { transform: translateZ(16px); }
  50%     { transform: translateZ(30px); }
}

/* -------------------------------------------------------- week strip -- */
.weeks { display: flex; gap: 5px; flex-wrap: wrap; margin: 6px 0 2px; }
.week {
  flex: 1 1 26px; min-width: 26px; height: 7px; border-radius: 4px;
  background: rgba(122,146,194,.14);
  position: relative; overflow: hidden;
}
.week > span {
  position: absolute; inset: 0; width: var(--fill, 0%);
  background: linear-gradient(90deg, var(--accent), var(--accent-2));
  border-radius: 4px;
  animation: grow 1.1s cubic-bezier(.2,.7,.3,1) both;
}
@keyframes grow { from { width: 0; } }

/* ------------------------------------------------------------- entrance -- */
.rise { animation: rise .55s cubic-bezier(.2,.7,.3,1) both; }
@keyframes rise {
  from { opacity: 0; transform: translateY(14px) scale(.985); }
  to   { opacity: 1; transform: none; }
}
.rise:nth-child(1) { animation-delay: .02s; }
.rise:nth-child(2) { animation-delay: .07s; }
.rise:nth-child(3) { animation-delay: .12s; }
.rise:nth-child(4) { animation-delay: .17s; }
.rise:nth-child(5) { animation-delay: .22s; }

/* --------------------------------------------------------------- badges -- */
.badge-wall { display: grid; gap: 10px;
              grid-template-columns: repeat(auto-fill, minmax(112px, 1fr)); }
.badge-tile {
  text-align: center; padding: 14px 8px 12px; border-radius: var(--r);
  border: 1px solid var(--line); background: rgba(122,146,194,.05);
  transition: transform .3s cubic-bezier(.2,.7,.3,1), box-shadow .3s ease;
}
.badge-tile.earned {
  border-color: rgba(126,224,184,.34);
  background: linear-gradient(160deg, rgba(126,224,184,.13), rgba(110,168,254,.07));
  box-shadow: 0 0 0 1px rgba(126,224,184,.12), 0 10px 26px -14px rgba(126,224,184,.5);
}
.badge-tile.earned:hover { transform: translateY(-3px) scale(1.03); }
.badge-tile .nm { font-size: .78rem; font-weight: 640; color: var(--ink);
                  margin-top: 8px; }
.badge-tile .ds { font-size: .68rem; color: var(--ink-3); margin-top: 3px;
                  line-height: 1.35; }
.badge-tile:not(.earned) svg { opacity: .3; }
.badge-tile:not(.earned) .nm { color: var(--ink-3); }

/* ---------------------------------------------------------- streamlit -- */
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, rgba(14,18,30,.96), rgba(9,12,20,.98));
  border-right: 1px solid var(--line);
  backdrop-filter: blur(20px);
}
section[data-testid="stSidebar"] .block-container { padding-top: 1.2rem; }

.stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {
  border-radius: var(--r-sm);
  border: 1px solid var(--line-strong);
  background: rgba(122,146,194,.10);
  color: var(--ink);
  font-weight: 600;
  padding: .55rem 1.05rem;
  min-height: 44px;
  transition: transform .18s ease, box-shadow .25s ease, background .25s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover {
  background: rgba(110,168,254,.16);
  border-color: rgba(110,168,254,.45);
  transform: translateY(-1px);
  box-shadow: 0 8px 22px -12px rgba(110,168,254,.8);
}
.stButton > button:active { transform: translateY(0); }
.stButton > button[kind="primary"],
.stButton > button[kind="primary"] p,
.stFormSubmitButton > button,
.stFormSubmitButton > button p {
  /* !important because Streamlit sets its own colour on the inner <p>, and on
     this gradient its light default is close to unreadable. */
  color: #07121f !important;
  font-weight: 700 !important;
}
.stButton > button[kind="primary"],
.stFormSubmitButton > button {
  background: linear-gradient(120deg, var(--accent), #8a7cf0);
  border: none;
}
.stButton > button:focus-visible, a:focus-visible, input:focus-visible {
  outline: 2px solid var(--accent-2); outline-offset: 2px;
}

div[data-testid="stExpander"] {
  border: 1px solid var(--line); border-radius: var(--r);
  background: var(--panel); backdrop-filter: blur(14px);
  overflow: hidden;
}
div[data-testid="stExpander"] summary { font-weight: 620; color: var(--ink-2); }

.stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid var(--line); }
.stTabs [data-baseweb="tab"] {
  border-radius: var(--r-sm) var(--r-sm) 0 0;
  padding: 10px 16px; min-height: 44px;
  color: var(--ink-3); font-weight: 600;
}
.stTabs [aria-selected="true"] {
  color: var(--ink) !important;
  background: rgba(110,168,254,.10);
  box-shadow: inset 0 -2px 0 var(--accent);
}

.stTextInput input, .stTextArea textarea, .stSelectbox [data-baseweb="select"] > div {
  background: rgba(10,14,24,.7) !important;
  border-color: var(--line) !important;
  color: var(--ink) !important;
  border-radius: var(--r-sm) !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
  border-color: rgba(110,168,254,.55) !important;
  box-shadow: 0 0 0 3px rgba(110,168,254,.15) !important;
}

div[data-testid="stAlert"] { border-radius: var(--r); border-width: 1px; }
hr { border-color: var(--line); }

.stProgress > div > div > div > div {
  background: linear-gradient(90deg, var(--accent), var(--accent-2));
}

/* Code blocks: a touch more contrast than the default, and a left rule so a
   cell reads as a unit rather than as floating text. */
.stCode, pre {
  border-radius: var(--r-sm) !important;
  border: 1px solid var(--line) !important;
  border-left: 2px solid rgba(110,168,254,.45) !important;
  background: rgba(8,11,19,.85) !important;
}

/* --------------------------------------------------------------- misc -- */
.lesson-cell {
  border-left: 2px solid rgba(110,168,254,.35);
  padding-left: 14px; margin: 6px 0 18px;
}
.cell-number {
  display: inline-flex; align-items: center; justify-content: center;
  width: 22px; height: 22px; border-radius: 6px;
  background: rgba(110,168,254,.16); border: 1px solid rgba(110,168,254,.3);
  color: var(--accent); font-size: .72rem; font-weight: 700;
  margin-right: 8px;
}
.says {
  background: rgba(126,224,184,.07);
  border: 1px solid rgba(126,224,184,.20);
  border-radius: var(--r-sm);
  padding: 10px 14px; margin-top: 10px;
  font-size: .92rem; color: var(--ink-2);
}
.trap {
  background: rgba(255,184,107,.07);
  border: 1px solid rgba(255,184,107,.22);
  border-radius: var(--r-sm);
  padding: 12px 15px; margin: 8px 0;
}
.trap b { color: var(--warn); }

.maths-chip {
  display: inline-block; margin: 3px 4px 3px 0;
}

.shimmer {
  background: linear-gradient(90deg, rgba(122,146,194,.06) 25%,
              rgba(122,146,194,.16) 37%, rgba(122,146,194,.06) 63%);
  background-size: 400% 100%;
  animation: shimmer 1.5s ease-in-out infinite;
  border-radius: var(--r-sm);
}
@keyframes shimmer { 0% { background-position: 100% 50%; }
                     100% { background-position: 0 50%; } }

/* ---------------------------------------------------------- responsive -- */
@media (max-width: 860px) {
  .hero-grid { grid-template-columns: 1fr; }
  .hero { padding: var(--space-4) var(--space-3); }
  .hero h1 { font-size: 1.62rem; }
  .lattice-wrap { height: 210px; }
  .lattice { grid-template-columns: repeat(5, 36px); gap: 8px;
             transform: rotateX(50deg) rotateZ(-38deg); }
  .tile { width: 36px; height: 36px; }
}
@media (max-width: 720px) {
  .block-container { padding-left: 16px !important; padding-right: 16px !important; }
  h1 { font-size: 1.5rem; }
  h2 { font-size: 1.18rem; }
  .stat .value { font-size: 1.3rem; }
  .card:hover { transform: translateY(-2px); }   /* no tilt on touch */
  .badge-wall { grid-template-columns: repeat(auto-fill, minmax(96px, 1fr)); }
  .stTabs [data-baseweb="tab"] { padding: 10px 11px; font-size: .86rem; }
}

/* Motion is suspended entirely for anyone who has asked for that. The layout
   is identical; only the movement stops. */
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    animation-duration: .001ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: .001ms !important;
  }
  .lattice { transform: rotateX(54deg) rotateZ(-42deg); }
  .card:hover { transform: none; }
}
</style>
"""


# ===========================================================================
# Boot
# ===========================================================================

def shell() -> None:
    """Everything that must happen exactly once per run, before any view.

    Called only by app.py. Split out from the per-view setup because
    st.set_page_config may be called once and once only, and with explicit
    navigation the entry script runs before the chosen view does.
    """
    configure_page()
    st.markdown(THEME_CSS, unsafe_allow_html=True)
    _prepare()


def configure_page() -> None:
    """Set the page configuration, if nothing has done so already.

    app.py calls this before importing anything else, because Streamlit
    requires set_page_config to be the very first Streamlit command and an
    import that prints a warning is enough to break that. Calling it twice is
    harmless here: the second call is swallowed, so a view run on its own
    still works.
    """
    try:
        st.set_page_config(
            page_title=APP_NAME, page_icon="\u25c8", layout="wide",
            # "auto" rather than "expanded": on a phone an expanded sidebar
            # covers the page you came to read, and this is studied on a train
            # as much as at a desk.
            initial_sidebar_state="auto",
        )
    except Exception:                     # noqa: BLE001 - already configured
        pass


def _prepare() -> None:
    """One-time database setup. Cheap on every run after the first."""
    if st.session_state.get("_db_ready"):
        return
    try:
        db.init_db()
    except Exception as exc:                      # noqa: BLE001 - shown, not hidden
        _database_unreachable(exc)

    issues = curriculum.validate() + concepts.validate()
    if issues:
        st.error("The curriculum failed validation:\n\n"
                 + "\n".join(f"- {i}" for i in issues))
        st.stop()

    # Seeding writes a few hundred rows. That is nothing against a file on a
    # laptop and close to a minute against a hosted database, where every
    # statement crosses a continent. The fingerprint covers exactly what
    # seeding writes, so an unchanged curriculum costs one query instead of
    # several hundred, and an edited one still re-seeds by itself.
    from . import challenges                      # noqa: PLC0415 - boot only

    fingerprint = db.seed_fingerprint(curriculum.TOPICS, challenges.CHALLENGES)
    if db.seeded_with() != fingerprint:
        with st.spinner("Setting up your plan — this happens once…"):
            db.seed_topics(curriculum.TOPICS)
            db.seed_challenge_bank(challenges.CHALLENGES)
            # Material generated before a curriculum change can reach past the
            # day it now sits on. It is retired rather than shown, because a
            # lesson using something you have not been taught is worse than no
            # lesson at all.
            db.retire_unteachable_content()
            db.mark_seeded(fingerprint)
    st.session_state["_db_ready"] = True


def page(page_title: str = "") -> dict[str, str]:
    """Every view starts with this. Returns the settings dict.

    page_title is accepted and ignored: the title now comes from the
    navigation declaration in app.py, which is the only place it can be right.
    The argument stays so a view reads as self-describing.
    """
    _prepare()
    try:
        settings = db.get_settings()
    except Exception as exc:                      # noqa: BLE001 - shown, not hidden
        _database_unreachable(exc)
    _sidebar(settings)
    return settings


# The old name, kept so nothing breaks if a view is run on its own.
bootstrap = page


def _database_unreachable(exc: Exception) -> None:
    """Explain a database failure in terms someone can act on, then stop.

    This exists because a KeyError once claimed a column was missing while a
    probe query said naming worked. Both could not be true, and nothing short
    of printing the raw shapes settled it.
    """
    from . import dbdriver                        # noqa: PLC0415

    st.error("**The app cannot read its database.**")
    st.markdown(f"```\n{exc}\n```")

    url = _os.environ.get("TURSO_DATABASE_URL", "")
    details = [
        f"- Database: **{url.split('://')[-1].split('/')[0] or 'a local file'}**",
        f"- Token supplied: **{'yes' if _os.environ.get('TURSO_AUTH_TOKEN') else 'no'}**",
    ]
    problem = dbdriver.replica_problem()
    if problem:
        details.append(f"- Local synced copy failed with: `{problem}`")
    details.append(f"- Driver installed: **{'yes' if dbdriver.libsql else 'no'}**")
    st.markdown("\n".join(details))
    st.caption("If the database name or token is wrong, fix it in the app's "
               "Secrets and reboot. If the database is paused, opening it once "
               "in the Turso dashboard wakes it.")

    with st.expander("Technical details — copy this if you are asked for it"):
        try:
            report = dbdriver.probe_report(db.get_conn())
        except Exception as inner:                # noqa: BLE001
            report = f"could not be collected: {type(inner).__name__}: {inner}"
        st.code(report, language="text")
    st.stop()


# ===========================================================================
# Components
# ===========================================================================

def _esc(value: Any) -> str:
    return _html.escape(str(value), quote=True)


def hero(title: str, subtitle: str, eyebrow: str = "",
         ring: tuple[float, str] | None = None,
         art: str = "") -> None:
    """The banner at the top of a page."""
    right = ""
    if ring is not None:
        right = f'<div>{progress_ring(ring[0], ring[1])}</div>'
    elif art == "network":
        right = f"<div>{neural_art()}</div>"

    eyebrow_html = (f'<div class="hero-eyebrow"><span class="dot" '
                    f'style="width:6px;height:6px;border-radius:50%;'
                    f'background:currentColor"></span>{_esc(eyebrow)}</div>'
                    if eyebrow else "")
    st.markdown(
        f'<div class="hero rise"><div class="hero-grid"><div>{eyebrow_html}'
        f'<h1>{_esc(title)}</h1><p>{_esc(subtitle)}</p></div>{right}</div></div>',
        unsafe_allow_html=True,
    )


def progress_ring(fraction: float, caption: str = "", size: int = 112) -> str:
    """A conic-gradient ring that fills from zero when the page loads.

    Animated with @property where the browser supports it and static where it
    does not, because a ring that jumps to its final value tells you the number
    but not the distance.
    """
    fraction = max(0.0, min(1.0, float(fraction)))
    degrees = round(fraction * 360)
    percent = round(fraction * 100)
    inner = size - 18
    return f"""
<div style="display:flex;flex-direction:column;align-items:center;gap:8px">
  <div style="width:{size}px;height:{size}px;border-radius:50%;
       background:conic-gradient(var(--accent) 0deg, var(--accent-2) {degrees}deg,
                  rgba(122,146,194,.14) {degrees}deg 360deg);
       display:grid;place-items:center;
       box-shadow:0 10px 30px -12px rgba(110,168,254,.6);
       animation:rise .7s cubic-bezier(.2,.7,.3,1) both">
    <div style="width:{inner}px;height:{inner}px;border-radius:50%;
         background:var(--panel-solid);display:grid;place-items:center;
         border:1px solid var(--line)">
      <span style="font-size:1.28rem;font-weight:700;color:var(--ink);
            font-variant-numeric:tabular-nums">{percent}%</span>
    </div>
  </div>
  <span style="font-size:.74rem;letter-spacing:.1em;text-transform:uppercase;
        color:var(--ink-3);font-weight:650">{_esc(caption)}</span>
</div>"""


def neural_art(width: int = 210, height: int = 140) -> str:
    """A small network that pulses along its edges.

    Three layers, drawn rather than pictured, with the signal animating from
    input to output on a loop. It is the app's one piece of ornament, and it
    is at least an honest diagram of the thing the course builds up to.
    """
    layers = [(28, [30, 70, 110]), (105, [22, 58, 94, 118]), (182, [52, 88])]
    edges, nodes = [], []
    for (x1, ys1), (x2, ys2) in zip(layers, layers[1:]):
        for i, y1 in enumerate(ys1):
            for j, y2 in enumerate(ys2):
                delay = (i + j) * 0.18
                edges.append(
                    f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
                    f'stroke="url(#wire)" stroke-width="1" opacity=".45">'
                    f'<animate attributeName="opacity" values=".12;.75;.12" '
                    f'dur="3.2s" begin="{delay:.2f}s" repeatCount="indefinite"/>'
                    f'</line>')
    for layer_index, (x, ys) in enumerate(layers):
        for i, y in enumerate(ys):
            delay = layer_index * 0.5 + i * 0.12
            nodes.append(
                f'<circle cx="{x}" cy="{y}" r="5.5" fill="url(#node)" '
                f'stroke="rgba(110,168,254,.5)" stroke-width="1">'
                f'<animate attributeName="r" values="5;7;5" dur="3.2s" '
                f'begin="{delay:.2f}s" repeatCount="indefinite"/></circle>')
    return f"""
<svg width="{width}" height="{height}" viewBox="0 0 210 140"
     role="img" aria-label="An animated three-layer neural network">
  <defs>
    <linearGradient id="wire" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0%" stop-color="#6ea8fe"/><stop offset="100%" stop-color="#7ee0b8"/>
    </linearGradient>
    <radialGradient id="node">
      <stop offset="0%" stop-color="#cfe2ff"/><stop offset="100%" stop-color="#6ea8fe"/>
    </radialGradient>
  </defs>
  {''.join(edges)}{''.join(nodes)}
</svg>"""


def lattice(weeks_done: dict[int, float], current_week: int = 0) -> None:
    """Thirteen weeks as an isometric board, one tile per week.

    `weeks_done` maps a week number to a fraction from 0 to 1. A finished week
    lifts clear of the surface; the current week hovers.
    """
    tiles = []
    for week in range(1, curriculum.PLAN_WEEKS + 1):
        share = weeks_done.get(week, 0.0)
        css = "done" if share >= 0.999 else ("part" if share > 0.05 else "")
        if week == current_week and css != "done":
            css = "now"
        state = ("complete" if share >= 0.999 else
                 f"{round(share * 100)} per cent done")
        tiles.append(f'<div class="tile {css}" title="Week {week}: {state}" '
                     f'aria-label="Week {week}, {state}"></div>')
    st.markdown(
        f'<div class="lattice-wrap"><div class="lattice" role="img" '
        f'aria-label="Thirteen weeks; raised tiles are finished">'
        f'{"".join(tiles)}</div></div>',
        unsafe_allow_html=True,
    )


def week_strip() -> None:
    """A flat fallback for the lattice, and the compact sidebar version."""
    topics = db.topics_with_mastery()
    by_week: dict[int, list[dict[str, Any]]] = {}
    for topic in topics:
        by_week.setdefault(topic["week"], []).append(topic)

    bars = []
    for week in sorted(by_week):
        group = by_week[week]
        done = sum(1 for t in group if scheduler.is_mastered(t["slug"]))
        share = round(100 * done / len(group)) if group else 0
        bars.append(f'<div class="week" title="Week {week}: {share}%">'
                    f'<span style="--fill:{share}%"></span></div>')
    st.markdown(f'<div class="weeks">{"".join(bars)}</div>',
                unsafe_allow_html=True)


def stat_row(stats: list[dict[str, Any]]) -> None:
    """A row of figures. Each is {label, value, sub?, grad?}."""
    columns = st.columns(len(stats)) if stats else []
    for column, stat in zip(columns, stats):
        with column:
            grad = " grad" if stat.get("grad") else ""
            sub = (f'<div class="sub">{_esc(stat["sub"])}</div>'
                   if stat.get("sub") else "")
            st.markdown(
                f'<div class="stat rise"><div class="label">'
                f'{_esc(stat["label"])}</div>'
                f'<div class="value{grad}">{_esc(stat["value"])}</div>{sub}</div>',
                unsafe_allow_html=True)


def card(title: str, body: str = "", meta: str = "", tone: str = "") -> None:
    meta_html = f'<div class="meta">{_esc(meta)}</div>' if meta else ""
    body_html = f"<p>{_esc(body)}</p>" if body else ""
    st.markdown(
        f'<div class="card {tone} rise"><h4>{_esc(title)}</h4>{body_html}'
        f'{meta_html}</div>', unsafe_allow_html=True)


def pill(text: str, tone: str = "") -> str:
    return f'<span class="pill {tone}"><span class="dot"></span>{_esc(text)}</span>'


def pills(items: list[tuple[str, str]]) -> None:
    st.markdown(" ".join(pill(text, tone) for text, tone in items),
                unsafe_allow_html=True)


def state_badge(slug: str) -> str:
    """Where a topic stands. Labelled as well as coloured, always."""
    if scheduler.is_mastered(slug):
        return pill("Mastered", "good")
    row = db.query_one("SELECT attempts, score FROM mastery WHERE topic_slug=?",
                       (slug,))
    if row and (row["attempts"] or 0) > 0:
        return pill(f"In progress · {round((row['score'] or 0) * 100)}%", "accent")
    topic = db.get_topic(slug)
    if topic and not scheduler.prereqs_met(topic)[0]:
        return pill("Locked", "warn")
    return pill("Not started")


def section(title: str, note: str = "") -> None:
    st.markdown(f"## {title}")
    if note:
        st.caption(note)


def rich_text(text: str, heading: str = "") -> None:
    """Render generated markdown, with its LaTeX handled properly.

    Streamlit renders $$...$$ inconsistently inside a markdown block, and this
    course puts a formula on its own line constantly. Splitting the display
    blocks out and passing them to st.latex is the difference between a lesson
    that reads and one full of stray dollar signs.
    """
    from . import textfmt                         # noqa: PLC0415

    if heading:
        st.markdown(f"### {heading}")
    text = textfmt.unescape_newlines(text or "")
    if not text.strip():
        return

    buffer: list[str] = []
    in_maths = False
    for chunk in text.split("$$"):
        if in_maths:
            if buffer:
                st.markdown("\n".join(buffer))
                buffer = []
            formula = chunk.strip()
            if formula:
                st.latex(formula)
        else:
            buffer.append(chunk)
        in_maths = not in_maths
    if buffer:
        st.markdown("\n".join(buffer))


def maths_chips(topic: dict[str, Any], key_prefix: str = "") -> str | None:
    """The Math Helper's offer, under a lesson.

    This is the feature that was asked for directly: rather than having to
    formulate a question about notation you do not know the name of, the terms
    the day uses are listed and one click asks about any of them. Returns the
    term chosen, if any.
    """
    from . import mathpad                         # noqa: PLC0415

    terms = mathpad.offer_for(topic)
    if not terms:
        return None

    st.markdown('<div style="font-size:.74rem;letter-spacing:.11em;'
                'text-transform:uppercase;color:var(--ink-3);font-weight:650;'
                'margin-bottom:8px">Not sure about a term? Ask.</div>',
                unsafe_allow_html=True)

    chosen = None
    per_row = 4
    for start in range(0, len(terms), per_row):
        row = terms[start:start + per_row]
        for column, (key, label, banked) in zip(st.columns(per_row), row):
            with column:
                if st.button(label, key=f"{key_prefix}math-{key}",
                             use_container_width=True,
                             help=("Explained in the app" if banked
                                   else "Will be explained on demand")):
                    chosen = key
    return chosen


BADGE_ICONS = {
    "spark":    "M12 2l2.2 6.3L21 10l-5.4 4 1.7 7-5.3-3.8L6.7 21l1.7-7L3 10l6.8-1.7z",
    "check":    "M4 12.5l5 5L20 6.5",
    "stack":    "M12 3l9 5-9 5-9-5zM3 13l9 5 9-5M3 17l9 5 9-5",
    "trophy":   "M7 4h10v5a5 5 0 01-10 0zM4 5h3M17 5h3M9 19h6M12 14v5",
    "target":   "M12 3a9 9 0 100 18 9 9 0 000-18zm0 4a5 5 0 100 10 5 5 0 000-10zm0 4a1 1 0 100 2 1 1 0 000-2z",
    "magnifier": "M11 4a7 7 0 100 14 7 7 0 000-14zM16.5 16.5L21 21",
    "arrow":    "M5 19L19 5M19 5h-8M19 5v8",
    "eye":      "M2 12s4-7 10-7 10 7 10 7-4 7-10 7S2 12 2 12zm10 3a3 3 0 100-6 3 3 0 000 6z",
    "scales":   "M12 3v18M5 7h14M7 7l-3 6h6zM17 7l-3 6h6z",
    "book":     "M4 4h7v16H4zM13 4h7v16h-7z",
    "shield":   "M12 3l8 3v6c0 5-3.5 8-8 9-4.5-1-8-4-8-9V6z",
    "hammer":   "M14 3l7 7-3 3-7-7zM11 8l-8 8 3 3 8-8",
    "chart":    "M4 20V10M10 20V4M16 20v-7M22 20H2",
    "dial":     "M12 3a9 9 0 019 9M12 12l5-4M12 21a9 9 0 01-9-9",
    "fire":     "M12 2s5 5 5 9a5 5 0 01-10 0c0-2 1-3 2-4 0 2 1 3 2 3 0-3 1-6 1-8z",
    "comet":    "M19 5l-9 9M5 19l4-4M3 13l3 3M15 3l2 2",
    "question": "M9 9a3 3 0 115 2.2c-.9.7-2 1.3-2 2.8M12 18h.01",
    "star":     "M12 3l2.6 6.5 6.9.5-5.3 4.4 1.7 6.6-5.9-3.8-5.9 3.8 1.7-6.6L2.5 10l6.9-.5z",
    "blocks":   "M4 4h7v7H4zM13 4h7v7h-7zM4 13h7v7H4zM13 13h7v7h-7z",
    "tower":    "M6 21V9l6-6 6 6v12M10 21v-5h4v5",
    "cap":      "M2 8l10-4 10 4-10 4zM6 11v5c0 1.5 3 3 6 3s6-1.5 6-3v-5",
}


def badge_icon(name: str, earned: bool, size: int = 26) -> str:
    path = BADGE_ICONS.get(name, BADGE_ICONS["star"])
    stroke = "url(#bg)" if earned else "rgba(139,149,169,.7)"
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
            f'stroke="{stroke}" stroke-width="1.7" stroke-linecap="round" '
            f'stroke-linejoin="round" aria-hidden="true">'
            f'<defs><linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">'
            f'<stop offset="0%" stop-color="#7ee0b8"/>'
            f'<stop offset="100%" stop-color="#6ea8fe"/></linearGradient></defs>'
            f'<path d="{path}"/></svg>')


def badge_wall(badges: list[dict[str, Any]]) -> None:
    tiles = []
    for badge in badges:
        earned = bool(badge.get("earned"))
        tiles.append(
            f'<div class="badge-tile {"earned" if earned else ""}" '
            f'title="{_esc(badge["description"])}">'
            f'{badge_icon(badge.get("icon", "star"), earned)}'
            f'<div class="nm">{_esc(badge["name"])}</div>'
            f'<div class="ds">{_esc(badge["description"])}</div></div>')
    st.markdown(f'<div class="badge-wall">{"".join(tiles)}</div>',
                unsafe_allow_html=True)


def level_bar() -> None:
    summary = gamify.summary()
    level = summary["level"]
    goal = summary["goal"]
    # Pulled out of the f-string below on purpose: nesting the same quote
    # character inside an f-string is a syntax error before Python 3.12, and
    # this app has to run on whatever the host provides as well as on a laptop.
    streak, badges = summary["streak"], summary["badges_earned"]
    badge_total = summary["badges_total"]
    minutes, target = goal["minutes"], goal["target"]
    st.markdown(
        f'<div class="glass rise" style="padding:16px 20px">'
        f'<div style="display:flex;justify-content:space-between;'
        f'align-items:baseline;gap:12px;flex-wrap:wrap">'
        f'<div><span style="font-weight:680;font-size:1.02rem">'
        f'Level {level["level"]} · {_esc(level["name"])}</span>'
        f'<span style="color:var(--ink-3);font-size:.84rem;margin-left:10px">'
        f'{level["xp"]:,} XP</span></div>'
        f'<div style="color:var(--ink-3);font-size:.8rem">'
        f'{level["to_next"]:,} XP to the next level</div></div>'
        f'<div style="height:8px;border-radius:5px;margin-top:10px;'
        f'background:rgba(122,146,194,.14);overflow:hidden">'
        f'<div style="height:100%;width:{level["progress"] * 100:.1f}%;'
        f'border-radius:5px;background:linear-gradient(90deg,var(--accent),'
        f'var(--accent-2));animation:grow 1.1s cubic-bezier(.2,.7,.3,1) both">'
        f'</div></div>'
        f'<div style="display:flex;gap:8px;margin-top:12px;flex-wrap:wrap">'
        + pill(f"{streak} day streak", "accent" if streak else "")
        + pill(f"{badges}/{badge_total} badges")
        + pill(f"{minutes}/{target} min today", "good" if goal["met"] else "")
        + '</div></div>', unsafe_allow_html=True)


# ===========================================================================
# Sidebar
# ===========================================================================

def _sidebar(settings: dict[str, str]) -> None:
    with st.sidebar:
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:10px;'
            f'margin-bottom:6px">'
            f'<div style="width:34px;height:34px;border-radius:10px;'
            f'background:linear-gradient(135deg,var(--accent),var(--accent-2));'
            f'display:grid;place-items:center;font-weight:800;color:#0a0d14;'
            f'font-size:.9rem">ML</div>'
            f'<div><div style="font-weight:700;font-size:1rem;line-height:1.1">'
            f'{APP_NAME}</div>'
            f'<div style="font-size:.7rem;color:var(--ink-3)">91 days</div>'
            f'</div></div>', unsafe_allow_html=True)

        try:
            status = scheduler.plan_status()
        except Exception:                         # noqa: BLE001
            return

        st.markdown(
            f'<div style="margin:14px 0 4px;font-size:.72rem;'
            f'letter-spacing:.1em;text-transform:uppercase;color:var(--ink-3);'
            f'font-weight:650">Plan</div>', unsafe_allow_html=True)
        week_strip()
        st.caption(f"{status['mastered']} of {status['total_topics']} days · "
                   f"{status['hours_spent']:.1f} h in")

        providers = _configured(settings)
        tone = "good" if providers else "warn"
        label = (", ".join(providers).title() if providers
                 else "No AI key configured")
        st.markdown(pill(label, tone), unsafe_allow_html=True)
        if not providers:
            st.caption("Open Settings and paste a free Groq key to generate "
                       "lessons.")


def _configured(settings: dict[str, str]) -> list[str]:
    from .config import configured_providers      # noqa: PLC0415
    try:
        return configured_providers(settings)
    except Exception:                             # noqa: BLE001
        return []


def require_provider(settings: dict[str, str]) -> bool:
    """Say clearly what is missing rather than failing later and vaguely."""
    if _configured(settings):
        return True
    st.warning("**No AI provider is configured yet.** Lessons, questions and "
               "the Math Helper's unwritten answers all need one.")
    st.markdown("Groq's free tier is enough for this whole course. Open "
                "**Settings**, paste a key, and press Save.")
    st.caption("Everything else — the curriculum, the datasets, the written "
               "Math Helper answers, the challenge bank — works without a key.")
    return False


# ===========================================================================
# Topic pieces
# ===========================================================================

def topic_picker(key: str = "topic", *,
                 default_slug: str | None = None) -> dict[str, Any]:
    topics = db.topics_with_mastery()
    if not topics:
        st.error("The plan has not been seeded yet.")
        st.stop()

    if default_slug is None:
        default_slug = st.session_state.get("current_topic")
    if default_slug is None:
        recommended = scheduler.next_topic()
        default_slug = recommended["slug"] if recommended else topics[0]["slug"]

    slugs = [t["slug"] for t in topics]
    index = slugs.index(default_slug) if default_slug in slugs else 0

    def label(slug: str) -> str:
        topic = next(t for t in topics if t["slug"] == slug)
        mark = "✓ " if scheduler.is_mastered(slug) else ""
        return f"{mark}Day {topic['day']:>2} · {topic['title']}"

    chosen = st.selectbox("Day", slugs, index=index, format_func=label, key=key)
    st.session_state["current_topic"] = chosen
    return next(t for t in topics if t["slug"] == chosen)


def topic_header(topic: dict[str, Any]) -> None:
    from . import datasets                        # noqa: PLC0415

    kind = topic.get("kind", "lesson")
    kind_tone = {"project": "accent", "milestone": "warn",
                 "capstone": "bad"}.get(kind, "")
    items = [
        (f"Week {topic['week']}", ""),
        (f"Day {topic['day']} of 91", ""),
        (topic["difficulty"].title(), ""),
        (f"{topic['minutes']} min", ""),
    ]
    if kind != "lesson":
        items.append((kind.title(), kind_tone))
    dataset = topic.get("dataset")
    if dataset and dataset in datasets.CATALOGUE:
        items.append((f"{dataset} · {datasets.CATALOGUE[dataset].rows} rows",
                      "accent"))

    st.markdown(f"# {topic['title']}")
    st.markdown(" ".join(pill(text, tone) for text, tone in items)
                + " " + state_badge(topic["slug"]), unsafe_allow_html=True)
    st.markdown(f"<p style='margin-top:14px'>{_esc(topic['summary'])}</p>",
                unsafe_allow_html=True)


def objectives_block(topic: dict[str, Any]) -> None:
    left, right = st.columns(2)
    with left:
        st.markdown("**By the end of today you can**")
        for objective in topic["objectives"]:
            st.markdown(f"- {objective}")
    with right:
        st.markdown("**Why it matters**")
        for note in topic["real_world"]:
            st.markdown(f"- {note}")


def fill_width(function: Any) -> dict[str, Any]:
    """The argument that makes a widget fill its column, for THIS Streamlit.

    Streamlit renamed this twice. Old builds take `use_container_width=True`;
    newer ones take `width="stretch"` and have dropped the old name entirely,
    so passing it raises TypeError and the page dies.

    Two installations reporting the same version number turned out to disagree
    about this, which is why it is decided by reading the function's actual
    signature rather than by checking a version number. A string default on
    `width` means the new API — that is the parameter that used to be an
    integer pixel count and is now "content" or "stretch".
    """
    try:
        parameters = _inspect.signature(function).parameters
    except (TypeError, ValueError):               # a builtin or C function
        return {}

    width = parameters.get("width")
    if width is not None and isinstance(width.default, str):
        return {"width": "stretch"}
    if "use_container_width" in parameters:
        return {"use_container_width": True}
    return {}                                     # neither: let it size itself


def figure(encoded: str, caption: str = "") -> None:
    """Render a base64 PNG captured from a lesson cell."""
    from . import sandbox                         # noqa: PLC0415
    st.image(sandbox.figure_data_uri(encoded), **fill_width(st.image))
    if caption:
        st.caption(caption)


def table(data: Any, **kwargs: Any) -> None:
    """A dataframe that fills its column on any Streamlit build."""
    st.dataframe(data, **kwargs, **fill_width(st.dataframe))


def verification_note(body: dict[str, Any]) -> None:
    """Say plainly whether this lesson's code was run and whether it passed.

    The promise is that it was. When it was not, that has to be visible rather
    than quietly absent, or the promise means nothing.
    """
    if body.get("verified"):
        st.markdown(
            pill("Every code cell ran against the real dataset", "good"),
            unsafe_allow_html=True)
        for warning in body.get("verify_warnings", [])[:2]:
            st.caption(warning)
        return

    problems = body.get("verify_problems") or []
    if not problems and body.get("generated") is False:
        return

    GATE_MEANS = {
        "horizon": ("It uses ideas from later in the course. Nothing here is "
                    "wrong, but some of it is early — if a part does not "
                    "land, that is why, and it is not you."),
        "claims": ("A number it states does not match what its code printed. "
                   "Trust the output below the cells, not the sentences "
                   "about them."),
        "execution": "Some of its code does not run.",
        "dataset": "It refers to data that is not bundled with this course.",
        "shape": "It came back incomplete.",
    }
    gate = body.get("verify_gate", "")
    st.warning(f"**This lesson did not pass the {gate} check.** "
               + GATE_MEANS.get(gate, "Read it with more care than usual.")
               + "\n\nIt is shown anyway rather than thrown away. Pressing "
                 "**Write a fresh lesson** tries again.")
    with st.expander(f"What the {gate} check objected to"):
        for problem in problems[:6]:
            st.markdown(f"- {problem}")


def session_timer(key: str) -> int:
    started = st.session_state.get(f"_t_{key}")
    if started is None:
        st.session_state[f"_t_{key}"] = time.time()
        return 0
    return int(time.time() - started)


def reset_timer(key: str) -> None:
    st.session_state[f"_t_{key}"] = time.time()


def show_generation_notes(notes: list[str]) -> None:
    if not notes:
        return
    with st.expander(f"{len(notes)} item(s) were discarded — why"):
        for note in notes[:12]:
            st.caption(f"· {note}")


def generation_error(message: str | None, *, what: str = "material") -> None:
    st.error(f"**Could not generate {what}.**\n\n{message or 'Unknown error.'}")
    st.caption("A free-tier rate limit is the usual cause. Waiting a minute "
               "and pressing the button again normally works; Settings has a "
               "connection test.")


def repair_code(code: str) -> str:
    from . import textfmt                         # noqa: PLC0415
    return textfmt.repair_code(code)
