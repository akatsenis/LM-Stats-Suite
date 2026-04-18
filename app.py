import streamlit as st
from modules.common import inject_css

st.set_page_config(page_title="Labomed Projects", page_icon="🔬", layout="wide")
inject_css()

pg = st.navigation(
    [
        st.Page("views/welcome.py", title="Welcome", icon="🏠", default=True),
        st.Page("views/stats_suite.py", title="Stats Suite", icon="📊"),
        st.Page("views/ivivc_suite.py", title="IVIVC Suite", icon="💊"),
        st.Page("views/doe_studio.py", title="DoE Studio", icon="🧪"),
    ]
)
pg.run()
