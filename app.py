from modules.common import init_page
import streamlit as st

init_page("lm Stats Suite")

st.title("Welcome")
st.markdown(
    """
    This app brings together practical statistical and experimental design tools for analytical, formulation, and development work, including 
    - descriptive statistics
    - regression
    - shelf-life analysis
    - dissolution comparison
    - hypothesis testing
    - ANOVA
    - PCA
    - DoE
    
    The pages are organized into Stats Suite and DoE Studio so all analyses stay under one app while remaining easy to navigate and maintain.
    """
)