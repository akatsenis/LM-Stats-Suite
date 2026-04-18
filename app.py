from modules.common import init_page
import streamlit as st
from modules.stats_suite import render as render_stats_suite
from modules.ivivc_suite import render as render_ivivc_suite
from modules.doe_studio import render as render_doe_studio

init_page("lm Stats Suite")


def home_page():
    st.title("Welcome")
    st.markdown(
        """
        This app brings together practical statistical and experimental design tools for analytical, formulation, and development work.

        **Available suites**
        - **📊 Stats Suite** for descriptive statistics, regression, shelf-life analysis, hypothesis testing, ANOVA, intervals, and PCA
        - **💊 IVIVC Suite** for dissolution similarity today, with room for additional IVIVC tools later
        - **🧪 DoE Studio** for process, mixture, mixture-process, and co-solvent experimental design work
        """
    )


pg = st.navigation(
    {
        "Apps": [
            st.Page(home_page, title="Welcome", icon="🏠"),
            st.Page(render_stats_suite, title="Stats Suite", icon="📊"),
            st.Page(render_ivivc_suite, title="IVIVC Suite", icon="💊"),
            st.Page(render_doe_studio, title="DoE Studio", icon="🧪"),
        ]
    }
)
pg.run()
