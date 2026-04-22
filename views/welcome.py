import json
from pathlib import Path

import streamlit as st
from modules.common import inject_css

inject_css()

DOCS_DIR = Path(__file__).resolve().parents[1] / "docs"
HELP_PDF = DOCS_DIR / "LM-Stats-Suite-Technical-Documentation.pdf"
HELP_INDEX_JSON = DOCS_DIR / "help_index.json"

if "welcome_help_open" not in st.session_state:
    st.session_state["welcome_help_open"] = False

st.sidebar.markdown("---")
if st.sidebar.button("📘 Help", key="welcome_help_button", use_container_width=True):
    st.session_state["welcome_help_open"] = not st.session_state.get("welcome_help_open", False)

st.title("🏠 Welcome")
st.markdown(
    """
    This app brings together practical statistical and experimental design tools for analytical, formulation, and development work.

    **Available suites**
    - **📊 Stats Suite** for descriptive statistics, regression, shelf-life analysis, hypothesis testing, ANOVA, intervals, and PCA
    - **💊 IVIVC Suite** for dissolution similarity, in vitro Weibull fitting, deconvolution, and IVIVC workflows
    - **🧪 DoE Studio** for process, mixture, mixture-process, and co-solvent experimental design work

    Use the sidebar to move between pages.
    """
)

if st.session_state.get("welcome_help_open", False):
    st.markdown("---")
    st.subheader("📘 Help index")
    st.caption("Search by page, method, equation, or keyword. The full technical manual can also be downloaded as a PDF.")
    items = []
    if HELP_INDEX_JSON.exists():
        try:
            items = json.loads(HELP_INDEX_JSON.read_text(encoding="utf-8"))
        except Exception:
            items = []
    query = st.text_input("Search help topics", placeholder="Example: IVIVC, deconvolution, Tukey, mixture, export")
    q = query.strip().lower()
    matches = []
    for item in items:
        hay = " ".join([item.get("topic", ""), item.get("summary", ""), " ".join(item.get("keywords", []))]).lower()
        if not q or q in hay:
            matches.append(item)
    if matches:
        for item in matches:
            kw = ", ".join(item.get("keywords", []))
            with st.expander(item.get("topic", "Help topic"), expanded=bool(q)):
                st.write(item.get("summary", ""))
                if kw:
                    st.caption(f"Keywords: {kw}")
    else:
        st.info("No help topics matched the current search. Try a broader keyword.")

    c1, c2 = st.columns([1, 2])
    with c1:
        if HELP_PDF.exists():
            st.download_button(
                "Download full PDF manual",
                data=HELP_PDF.read_bytes(),
                file_name="LM-Stats-Suite-Technical-Documentation.pdf",
                mime="application/pdf",
                use_container_width=True,
                key="welcome_help_download_pdf",
            )
    with c2:
        st.caption("The bundled manual documents the pages, equations, numerical engines, key Python packages, and export/reporting flow used in the current app.")
