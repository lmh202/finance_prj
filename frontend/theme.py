"""AURORA visual theme — ports the ui_design ("Prism") look into Streamlit.

`apply_theme()` injects the parts a config.toml can't express: web fonts
(Space Grotesk / IBM Plex Mono), the radial-glow background, film grain,
card surfaces, and polished metric / button / input / dataframe styling.

Call it once per page, right AFTER st.set_page_config(...). It only touches
presentation — no page logic, no data flow — so every engine page keeps
working exactly as before.

Design tokens (mirrors ui_design/src/app/globals.css):
    bg0 #06090c · panel #0b1014 · ink #ecf3ee · mut #86938a
    line rgba(255,255,255,.08) · accent #b3f34c · gain #3fd9a4 · loss #ff7a7a
"""

import streamlit as st

# ui_design allocation/benchmark palette, exposed so charts can match the UI.
CHART = {
    "accent": "#b3f34c",   # portfolio / primary series
    "spy": "#8da2fb",      # S&P 500 benchmark
    "qqq": "#cfa3ff",      # NASDAQ 100 benchmark
    "gain": "#3fd9a4",
    "loss": "#ff7a7a",
    "ink": "#ecf3ee",
    "mut": "#86938a",
    "cash": "#5b6560",
    "grid": "rgba(255,255,255,0.06)",
}

# fractal-noise film grain, identical to the ui_design overlay
_GRAIN = (
    "data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'"
    "%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.85'"
    " numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25'"
    " height='100%25' filter='url(%23n)'/%3E%3C/svg%3E"
)

_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root {
  --a-bg0:#06090c; --a-panel:#0b1014;
  --a-ink:#ecf3ee; --a-mut:#86938a;
  --a-line:rgba(255,255,255,0.08);
  --a-accent:#b3f34c; --a-accent-dim:rgba(179,243,76,0.14);
  --a-gain:#3fd9a4; --a-loss:#ff7a7a;
  --a-font:'Space Grotesk',ui-sans-serif,system-ui,-apple-system,'Segoe UI',sans-serif;
  --a-mono:'IBM Plex Mono',ui-monospace,SFMono-Regular,monospace;
}

/* ---------- canvas: radial glows over near-black ---------- */
.stApp {
  background:
    radial-gradient(1200px 700px at 85% -10%, rgba(179,243,76,0.06), transparent 60%),
    radial-gradient(1000px 800px at -10% 20%, rgba(77,163,255,0.05), transparent 55%),
    radial-gradient(900px 900px at 50% 120%, rgba(179,243,76,0.035), transparent 60%),
    var(--a-bg0);
  background-attachment: fixed;
  color: var(--a-ink);
}
/* film grain (visual only — never intercepts clicks) */
.stApp::before {
  content:""; position:fixed; inset:0; z-index:60; pointer-events:none;
  opacity:0.032; background-image:url("__GRAIN__");
}

/* ---------- typography (leave icon fonts untouched) ---------- */
html, body, .stApp,
.stApp p, .stApp li, .stApp label, .stApp a, .stApp small,
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
.stApp button, .stApp input, .stApp textarea, .stApp select,
[data-testid="stMarkdownContainer"], [data-testid="stWidgetLabel"] {
  font-family: var(--a-font);
}
[data-testid="stIconMaterial"], .material-icons, .material-icons-outlined,
.material-symbols-outlined, .material-symbols-rounded {
  font-family:'Material Symbols Rounded','Material Symbols Outlined','Material Icons' !important;
}
code, pre, kbd, [data-testid="stMetricValue"], [data-testid="stMetricDelta"] {
  font-family: var(--a-mono); font-variant-numeric: tabular-nums;
}

.stApp h1 { font-weight:700; letter-spacing:-0.02em; color:var(--a-ink); }
.stApp h2, .stApp h3 { font-weight:600; letter-spacing:-0.01em; color:var(--a-ink); }
[data-testid="stMarkdownContainer"] a { color:var(--a-accent); text-decoration:none; }
[data-testid="stMarkdownContainer"] a:hover { text-decoration:underline; }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] * { color:var(--a-mut); }

/* ---------- layout width + spacing ---------- */
.stMainBlockContainer, .block-container {
  max-width:1400px; padding-top:2.6rem; padding-bottom:4rem;
}

/* ---------- top chrome ---------- */
[data-testid="stHeader"] { background:rgba(6,9,12,0.55); backdrop-filter:blur(12px); }
[data-testid="stDecoration"] { display:none; }
[data-testid="stToolbar"] { color:var(--a-mut); }

/* ---------- sidebar ---------- */
[data-testid="stSidebar"] {
  background:linear-gradient(180deg, rgba(255,255,255,0.025), rgba(255,255,255,0.006));
  border-right:1px solid var(--a-line);
}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3, [data-testid="stSidebar"] label { color:var(--a-ink); }

/* ---------- metric cards ---------- */
[data-testid="stMetric"] {
  background:linear-gradient(180deg, rgba(255,255,255,0.045), rgba(255,255,255,0.018));
  border:1px solid var(--a-line); border-radius:18px; padding:16px 18px;
  box-shadow:inset 0 1px 0 rgba(255,255,255,0.05), 0 24px 60px -32px rgba(0,0,0,0.7);
}
[data-testid="stMetricLabel"] {
  text-transform:uppercase; letter-spacing:0.14em; font-size:0.68rem; color:var(--a-mut);
}
[data-testid="stMetricValue"] { color:var(--a-ink); font-weight:600; }

/* ---------- cards: forms, bordered containers, expanders ---------- */
[data-testid="stForm"],
[data-testid="stExpander"] details,
[data-testid="stVerticalBlockBorderWrapper"]:has(> div > [data-testid="stVerticalBlock"]) > div > [data-testid="stVerticalBlock"] {
  border-radius:20px;
}
[data-testid="stForm"], [data-testid="stExpander"] details {
  background:linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.015));
  border:1px solid var(--a-line); border-radius:20px; overflow:hidden;
  box-shadow:inset 0 1px 0 rgba(255,255,255,0.05), 0 24px 60px -32px rgba(0,0,0,0.7);
}
[data-testid="stExpander"] summary { color:var(--a-ink); }

/* ---------- buttons ---------- */
.stButton>button, .stDownloadButton>button, .stFormSubmitButton>button {
  border-radius:12px; border:1px solid var(--a-line);
  background:rgba(255,255,255,0.03); color:var(--a-ink);
  font-weight:500; transition:all 0.15s ease;
}
.stButton>button:hover, .stDownloadButton>button:hover, .stFormSubmitButton>button:hover {
  border-color:rgba(179,243,76,0.5); background:var(--a-accent-dim); color:var(--a-accent);
}
button[kind="primary"], button[kind="primaryFormSubmit"],
[data-testid="stBaseButton-primary"], [data-testid="stBaseButton-primaryFormSubmit"] {
  background:var(--a-accent) !important; color:#0a0f07 !important; border:none !important; font-weight:600;
}
button[kind="primary"]:hover, button[kind="primaryFormSubmit"]:hover,
[data-testid="stBaseButton-primary"]:hover, [data-testid="stBaseButton-primaryFormSubmit"]:hover {
  filter:brightness(1.08); color:#0a0f07 !important;
}

/* ---------- inputs (BaseWeb) ---------- */
[data-baseweb="input"], [data-baseweb="base-input"],
[data-baseweb="select"]>div, [data-baseweb="textarea"] {
  background:rgba(255,255,255,0.03) !important; border-radius:12px !important;
}
[data-baseweb="input"]:focus-within, [data-baseweb="select"]>div:focus-within,
[data-baseweb="textarea"]:focus-within {
  border-color:var(--a-accent) !important; box-shadow:0 0 0 1px rgba(179,243,76,0.35) !important;
}
.stTextInput input, .stNumberInput input { color:var(--a-ink) !important; }

/* ---------- dataframes ---------- */
[data-testid="stDataFrame"] { border-radius:14px; overflow:hidden; border:1px solid var(--a-line); }

/* ---------- misc ---------- */
hr, [data-testid="stDivider"] { border-color:var(--a-line) !important; }
[data-testid="stNotification"], [data-testid="stAlert"], .stAlert { border-radius:14px; }
::selection { background:rgba(179,243,76,0.85); color:#0a0f07; }
""".replace("__GRAIN__", _GRAIN)


def apply_theme() -> None:
    """Inject the AURORA/Prism visual theme. Presentation only — call once
    per page, immediately after st.set_page_config()."""
    st.markdown(f"<style>{_CSS}</style>", unsafe_allow_html=True)
