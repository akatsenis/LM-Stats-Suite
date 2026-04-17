from modules.common import init_page, app_header, info_box
import streamlit as st

init_page("lm Stats")
app_header("🔬 lm Stats", "One Streamlit app with a multipage structure: Home, Stats Suite, and DoE Studio.")
st.markdown("""
### Welcome
Use the page selector in the left sidebar to move between:

- **Home**
- **Stats Suite**
- **DoE Studio**

This structure keeps everything under one Streamlit link while splitting the code into separate modules for easier maintenance.
""")
info_box("Tip: the same display/export settings are available on each page, and your graph style preferences persist in session state.")
