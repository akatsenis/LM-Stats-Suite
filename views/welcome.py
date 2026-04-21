import streamlit as st
from modules.common import inject_css

inject_css()

st.title("🏠 Welcome")
st.markdown(
    """
    This app brings together practical statistical and experimental design tools for analytical, formulation, and development work.

    **Available suites**
    - **📊 Stats Suite** for descriptive statistics, regression, shelf-life analysis, hypothesis testing, ANOVA, intervals, and PCA
    - **💊 IVIVC Suite** for dissolution similarity (f2) and additional IVIVC tools
    - **🧪 DoE Studio** for process, mixture, mixture-process, and co-solvent experimental design work

    Use the sidebar to move between pages.
    """
)
