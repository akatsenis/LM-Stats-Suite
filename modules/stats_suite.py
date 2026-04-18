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
        "p Value": [p_reg, np.nan, np.nan],
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
        st.info("This file includes the PCA ellipse checkbox update. Keep your current Descriptive Statistics block here.")

    if tool == "02 - Regression Intervals":
        app_header("📈 App 02 - Regression Intervals", "Linear regression with CI / PI / both, one-sided or two-sided bands, prediction points, and spec-limit crossing.")
        st.info("Keep your current Regression block here.")

    if tool == "03 - Shelf Life Estimator":
        app_header("⏳ App 03 - Shelf Life Estimator", "Paste stability data, choose lower or upper specification, and estimate shelf life from fit, CI, or PI crossing.")
        st.info("Keep your current Shelf Life block here.")

    if tool == "04 - Dissolution Comparison (f2)":
        app_header("💊 App 04 - Dissolution Comparison (f2)", "FDA-style point selection, conventional f2 checks, and optional bootstrap / BCa confidence intervals.")
        st.info("Keep your current Dissolution block here.")

    if tool == "05 - Two-Sample Tests":
        app_header("⚖️ App 05 - Two-Sample Tests", "Paste one table with headers, then choose any two sample columns to compare.")
        st.info("Keep your current Two-Sample block here.")

    if tool == "06 - Two-Way ANOVA":
        app_header("📐 App 06 - Two-Way ANOVA", "Analyze two categorical factors and their interaction for a selected numeric response.")
        st.info("Keep your current Two-Way ANOVA block here.")

    if tool == "07 - Tolerance & Confidence Intervals":
        app_header("🎯 App 07 - Tolerance & Confidence Intervals", "Generate confidence intervals and normal-theory tolerance intervals for one or two samples.")
        st.info("Keep your current Tolerance/CI block here.")

    if tool == "08 - PCA Analysis":
        app_header("🌐 App 08 - PCA Analysis", "Reduce multivariate data to principal components and visualize scores and loadings.")
        data_input = st.text_area("Paste data with headers", height=240)
        decimals = st.slider("Decimals", 1, 8, DEFAULT_DECIMALS, key="pca_dec")
        if data_input:
            try:
                df = parse_pasted_table(data_input, header=True)
                num_cols = get_numeric_columns(df)
                all_cols = list(df.columns)
                c1, c2, c3, c4, c5 = st.columns([1.2, 1, 1, 0.9, 1.1])
                with c1:
                    vars_sel = st.multiselect("Numeric variables", num_cols, default=num_cols)
                with c2:
                    label_col = st.selectbox("Label column (optional)", ["(None)"] + all_cols)
                with c3:
                    group_col = st.selectbox("Group column (optional)", ["(None)"] + [c for c in all_cols if c != label_col])
                with c4:
                    show_ellipses = st.checkbox("Show ellipses", value=True)
                with c5:
                    ellipse_mode = st.selectbox("Ellipse mode", ["Overall", "By group", "Both"], disabled=not show_ellipses)

                if len(vars_sel) >= 2:
                    X = df[vars_sel].apply(to_numeric).dropna()
                    Z = (X - X.mean()) / X.std(ddof=1)
                    pca = PCA(n_components=2)
                    scores = pca.fit_transform(Z)
                    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
                    exp = pca.explained_variance_ratio_ * 100
                    eig = pd.DataFrame({
                        "Principal Component": ["PC1", "PC2"],
                        "Eigenvalue": pca.explained_variance_,
                        "Variance Explained (%)": exp,
                        "Cumulative Variance (%)": np.cumsum(exp),
                    })
                    load_df = pd.DataFrame({"Variable": vars_sel, "PC1": loadings[:, 0], "PC2": loadings[:, 1]})
                    report_table(eig, "Eigenvalues and explained variance", decimals)
                    report_table(load_df, "Loading matrix", decimals)

                    scores_df = pd.DataFrame({"PC1": scores[:, 0], "PC2": scores[:, 1]}, index=X.index)
                    if label_col != "(None)":
                        scores_df["Label"] = df.loc[X.index, label_col].astype(str).values
                    if group_col != "(None)":
                        scores_df["Group"] = df.loc[X.index, group_col].astype(str).values

                    score_cfg = common.safe_get_plot_cfg("PCA score plot")
                    fig_scores, ax = plt.subplots(figsize=(score_cfg["fig_w"], score_cfg["fig_h"]))
                    color_cycle = [score_cfg["primary_color"], score_cfg["secondary_color"], score_cfg["tertiary_color"], "#9467bd", "#8c564b", "#e377c2"]

                    if group_col != "(None)":
                        unique_groups = list(scores_df["Group"].unique())
                        for i, grp in enumerate(unique_groups):
                            col = color_cycle[i % len(color_cycle)]
                            m = scores_df["Group"] == grp
                            ax.scatter(scores_df.loc[m, "PC1"], scores_df.loc[m, "PC2"], s=score_cfg["marker_size"], color=col, label=str(grp))
                            if show_ellipses and ellipse_mode in ["By group", "Both"]:
                                draw_conf_ellipse(scores_df.loc[m, ["PC1", "PC2"]].to_numpy(), ax, edgecolor=col, facecolor=col, plot_key="PCA score plot")
                        if show_ellipses and ellipse_mode in ["Overall", "Both"]:
                            draw_conf_ellipse(scores_df[["PC1", "PC2"]].to_numpy(), ax, edgecolor="#111827", facecolor="#111827", plot_key="PCA score plot")
                    else:
                        col = score_cfg["primary_color"]
                        ax.scatter(scores_df["PC1"], scores_df["PC2"], s=score_cfg["marker_size"], color=col, label="Scores")
                        if show_ellipses:
                            draw_conf_ellipse(scores_df[["PC1", "PC2"]].to_numpy(), ax, edgecolor=col, facecolor=col, plot_key="PCA score plot")

                    if label_col != "(None)":
                        for _, row in scores_df.iterrows():
                            ax.text(row["PC1"], row["PC2"], str(row["Label"]), fontsize=8)

                    ax.axhline(0, color="#64748b", lw=score_cfg["aux_line_width"], ls=score_cfg["aux_line_style"])
                    ax.axvline(0, color="#64748b", lw=score_cfg["aux_line_width"], ls=score_cfg["aux_line_style"])
                    apply_ax_style(ax, "PCA score plot", f"PC1 ({exp[0]:.1f}% var)", f"PC2 ({exp[1]:.1f}% var)", legend=(group_col != "(None)"), plot_key="PCA score plot")
                    st.pyplot(fig_scores)

                    load_cfg = common.safe_get_plot_cfg("PCA loading plot")
                    fig_load, ax2 = plt.subplots(figsize=(load_cfg["fig_w"], load_cfg["fig_h"]))
                    ax2.axhline(0, color="#64748b", lw=load_cfg["aux_line_width"], ls=load_cfg["aux_line_style"])
                    ax2.axvline(0, color="#64748b", lw=load_cfg["aux_line_width"], ls=load_cfg["aux_line_style"])
                    for i, var in enumerate(vars_sel):
                        ax2.arrow(0, 0, loadings[i, 0], loadings[i, 1], head_width=load_cfg["arrow_size"], length_includes_head=True, color=load_cfg["primary_color"], lw=load_cfg["line_width"], ls=load_cfg["line_style"])
                        ax2.text(loadings[i, 0], loadings[i, 1], var)
                    lim = max(1.1, np.max(np.abs(loadings)) * 1.2)
                    ax2.set_xlim(-lim, lim)
                    ax2.set_ylim(-lim, lim)
                    apply_ax_style(ax2, "PCA loading plot", "PC1", "PC2", plot_key="PCA loading plot")
                    st.pyplot(fig_load)

                    export_results(
                        prefix="pca_analysis",
                        report_title="Statistical Analysis Report",
                        module_name="PCA Analysis",
                        statistical_analysis="Principal component analysis was performed on the selected numeric variables after standardization to zero mean and unit variance. Eigenvalues, explained variance, component scores, and variable loadings were calculated, and optional pasted header columns were used as point labels or groups on the score plot.",
                        offer_text="This analysis offers a way to reduce dimensionality, detect clustering or separation patterns, identify influential variables, and visualize multivariate relationships while retaining labels or grouping information from the pasted Excel headers.",
                        python_tools="Python tools used here include pandas and numpy for data handling and standardization, sklearn.decomposition.PCA for principal component analysis, matplotlib for score and loading plots, openpyxl for Excel export, and reportlab for the PDF-style report.",
                        table_map={"Explained Variance": eig, "Loadings": load_df, "Scores": scores_df.reset_index(drop=True)},
                        figure_map={"PCA score plot": fig_to_png_bytes(fig_scores), "PCA loading plot": fig_to_png_bytes(fig_load)},
                        conclusion="PCA transforms correlated variables into orthogonal components that summarize the major variation structure in the data and help reveal clustering, separation, and variable influence patterns.",
                        decimals=decimals,
                    )
            except Exception as e:
                st.error(str(e))
