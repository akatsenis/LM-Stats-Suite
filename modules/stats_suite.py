import modules.common as common
from modules.common import *

st = common.st
pd = common.pd
np = common.np
plt = common.plt
stats = common.stats
smf = common.smf
anova_lm = common.anova_lm
PCA = common.PCA

app_header = common.app_header
info_box = common.info_box

DEFAULT_DECIMALS = common.DEFAULT_DECIMALS


def regression_anova_and_coefficients_local(x, y, alpha=0.05):
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()

    n = len(x)
    if n < 3:
        raise ValueError("At least 3 points are required.")

    X = np.column_stack([np.ones(n), x])
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)

    intercept, slope = beta
    fitted = X @ beta
    resid = y - fitted

    df_reg = 1
    df_err = n - 2
    df_tot = n - 1

    ss_reg = float(np.sum((fitted - np.mean(y)) ** 2))
    ss_err = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))

    ms_reg = ss_reg / df_reg
    ms_err = ss_err / df_err if df_err > 0 else np.nan

    f_stat = ms_reg / ms_err if ms_err > 0 else np.nan
    p_reg = 1 - stats.f.cdf(f_stat, df_reg, df_err) if np.isfinite(f_stat) else np.nan

    se_beta = np.sqrt(np.diag(XtX_inv) * ms_err)
    t_vals = beta / se_beta
    p_vals = 2 * (1 - stats.t.cdf(np.abs(t_vals), df_err))

    tcrit = stats.t.ppf(1 - alpha / 2, df_err)

    coef_df = pd.DataFrame({
        "Term": ["Intercept", "Slope"],
        "Coefficient": [intercept, slope],
        "SE Coefficient": se_beta,
        "t Value": t_vals,
        "p Value": p_vals,
        "Lower CI": beta - tcrit * se_beta,
        "Upper CI": beta + tcrit * se_beta,
    })

    anova_df = pd.DataFrame({
        "Source": ["Regression", "Error", "Total"],
        "DF": [df_reg, df_err, df_tot],
        "SS": [ss_reg, ss_err, ss_tot],
        "MS": [ms_reg, ms_err, np.nan],
        "F": [f_stat, np.nan, np.nan],
        "P Value": [p_reg, np.nan, np.nan],
    })

    trend_text = (
        f"Significant linear trend detected (slope p = {p_vals[1]:.4g} < {alpha:.4g})."
        if p_vals[1] < alpha
        else f"No significant linear trend detected (slope p = {p_vals[1]:.4g} ≥ {alpha:.4g})."
    )

    return {
        "anova": anova_df,
        "coefficients": coef_df,
        "slope_p_value": float(p_vals[1]),
        "regression_p_value": float(p_reg),
        "f_stat": float(f_stat),
        "trend_text": trend_text,
    }


TOOLS = ['01 - Descriptive Statistics', '02 - Regression Intervals', '03 - Shelf Life Estimator', '04 - Dissolution Comparison (f2)', '05 - Two-Sample Tests', '06 - Two-Way ANOVA', '07 - Tolerance & Confidence Intervals', '08 - PCA Analysis']


def render():
    render_display_settings()
    st.sidebar.title("🔬 lm Stats")
    st.sidebar.markdown("Stats Suite")
    tool = st.sidebar.radio("Stats tool", TOOLS, key="stats_tool")
    st.sidebar.caption("Use the pages menu to switch to DoE Studio.")
    if tool == "01 - Descriptive Statistics":
        app_header("📊 App 01 - Descriptive Statistics", "Paste one or more numeric columns with headers. For one column, get a graphical summary. For multiple columns, choose a reference and a test column to compare.")
        data_input = st.text_area("Data (paste with headers)", height=220)
        decimals = st.slider("Decimals", 1, 8, DEFAULT_DECIMALS, key="desc_dec")
        alpha = st.slider("Significance level α", 0.001, 0.100, 0.050, 0.001, key="desc_alpha")
        mean_ci_conf = st.slider("Mean CI confidence (%)", 80, 99, 95, 1, key="desc_mean_ci")
        tol_cov = st.slider("Tolerance interval coverage (%)", 80, 99, 99, 1, key="desc_tol_cov")
        tol_conf = st.slider("Tolerance interval confidence (%)", 80, 99, 95, 1, key="desc_tol_conf")
        st.info("This is the file version created from the code you pasted. It is intended as a downloadable handoff file.")
