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
normal_ad = common.normal_ad

app_header = common.app_header
info_box = common.info_box
show_figure = common.show_figure
DEFAULT_DECIMALS = common.DEFAULT_DECIMALS

TOOLS = [
    '📊 Descriptive Statistics',
    '📈 Regression Analysis',
    '⏳ Shelf Life Estimator',
    '⚖️ Two-Sample Tests',
    '📐 Two-Way ANOVA / GLM',
    '🎯 Tolerance & Confidence Intervals',
    '🌐 PCA Analysis',
]

SAMPLE_DATA = {
    "desc": "LotA\tLotB\n98.2\t97.5\n99.1\t98.1\n100.4\t99.0\n97.8\t98.4\n98.9\t97.9\n99.5\t98.8\n100.1\t99.2\n98.7\t98.0\n",
    "reg": "Month\tAssay\n0\t100.0\n3\t99.1\n6\t98.5\n9\t97.8\n12\t97.2\n18\t96.0\n24\t94.8\n",
    "shelf": "Month\tAssay\n0\t100.0\n3\t99.4\n6\t98.8\n9\t98.1\n12\t97.6\n18\t96.3\n24\t95.1\n36\t92.4\n",
    "f2_ref": "Time\tU1\tU2\tU3\tU4\tU5\tU6\tU7\tU8\tU9\tU10\tU11\tU12\n5\t22\t24\t23\t25\t21\t24\t23\t22\t24\t25\t23\t22\n10\t45\t47\t46\t48\t44\t46\t45\t47\t46\t48\t45\t46\n15\t63\t65\t64\t66\t62\t64\t63\t65\t64\t66\t63\t64\n20\t78\t80\t79\t81\t77\t79\t78\t80\t79\t81\t78\t79\n30\t91\t92\t93\t92\t90\t91\t92\t93\t92\t91\t92\t91\n45\t97\t98\t98\t99\t97\t98\t98\t99\t98\t98\t97\t98\n",
    "f2_test": "Time\tU1\tU2\tU3\tU4\tU5\tU6\tU7\tU8\tU9\tU10\tU11\tU12\n5\t20\t22\t21\t23\t20\t22\t21\t22\t23\t22\t21\t22\n10\t42\t44\t43\t45\t41\t43\t42\t44\t43\t44\t42\t43\n15\t60\t62\t61\t63\t59\t61\t60\t62\t61\t62\t60\t61\n20\t75\t77\t76\t78\t74\t76\t75\t77\t76\t77\t75\t76\n30\t88\t89\t90\t90\t87\t88\t89\t90\t89\t89\t88\t89\n45\t95\t96\t97\t97\t94\t95\t96\t97\t96\t96\t95\t96\n",
    "two_sample": "Reference\tTest\tPaired_A\tPaired_B\n101.2\t99.8\t10.2\t10.0\n100.8\t98.9\t10.5\t10.1\n99.7\t100.1\t9.9\t9.7\n100.4\t99.2\t10.3\t10.0\n101.0\t99.5\t10.1\t9.8\n99.9\t98.7\t10.4\t10.2\n100.6\t99.1\t10.0\t9.9\n",
    "anova": "Operator\tMachine\tShift\tTemp\tResponse\nA\tM1\tDay\t24.8\t98.1\nA\tM1\tNight\t25.4\t97.4\nA\tM2\tDay\t24.9\t99.0\nA\tM2\tNight\t25.7\t98.0\nB\tM1\tDay\t25.2\t97.6\nB\tM1\tNight\t25.8\t96.8\nB\tM2\tDay\t25.0\t98.5\nB\tM2\tNight\t26.1\t97.2\nC\tM1\tDay\t24.7\t98.8\nC\tM1\tNight\t25.3\t97.9\nC\tM2\tDay\t24.8\t99.3\nC\tM2\tNight\t25.9\t98.1\nA\tM1\tDay\t24.6\t98.4\nA\tM2\tDay\t25.1\t98.7\nB\tM1\tNight\t25.9\t96.9\nB\tM2\tNight\t26.0\t97.5\nC\tM1\tDay\t24.9\t98.6\nC\tM2\tDay\t25.0\t99.1\n",
    "ti": "SampleA\tSampleB\n98.1\t97.4\n99.2\t98.0\n100.0\t98.8\n97.9\t97.1\n98.7\t97.9\n99.5\t98.4\n100.2\t99.0\n",
    "pca": "Batch\tSite\tAssay\tImpurity\tWater\tHardness\nB1\tNorth\t99.1\t0.12\t1.8\t7.2\nB2\tNorth\t98.7\t0.18\t2.0\t7.0\nB3\tSouth\t97.9\t0.31\t2.8\t6.1\nB4\tSouth\t98.2\t0.27\t2.5\t6.4\nB5\tEast\t99.4\t0.10\t1.7\t7.5\nB6\tEast\t99.0\t0.14\t1.9\t7.3\nB7\tWest\t97.6\t0.35\t3.0\t5.9\nB8\tWest\t97.8\t0.33\t2.9\t6.0\n",
}


def load_sample_text(state_key, sample_key):
    st.session_state[state_key] = SAMPLE_DATA[sample_key]


def load_dual_sample_text(state_key_a, sample_key_a, state_key_b, sample_key_b):
    st.session_state[state_key_a] = SAMPLE_DATA[sample_key_a]
    st.session_state[state_key_b] = SAMPLE_DATA[sample_key_b]


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
    return {
        "anova": anova_df,
        "coefficients": coef_df,
        "slope_p_value": float(p_vals[1]),
        "regression_p_value": float(p_reg),
        "f_stat": float(f_stat),
    }


def _one_sample_summary(arr, label, ci_conf=0.95, tol_p=0.99, tol_confidence=0.95):
    arr = np.asarray(arr, dtype=float)
    n = len(arr)
    mean = np.mean(arr)
    sd = np.std(arr, ddof=1) if n > 1 else np.nan
    se = sd / np.sqrt(n) if n > 1 else np.nan
    tcrit = t.ppf(1 - (1 - ci_conf) / 2, n - 1) if n > 1 else np.nan
    ci_half = tcrit * se if n > 1 else np.nan
    _, tol_lower, tol_upper = tolerance_interval_normal(arr, p=tol_p, conf=tol_confidence, two_sided=True)
    ad_stat, ad_p = normal_ad(arr) if n >= 8 else (np.nan, np.nan)
    try:
        sh_stat, sh_p = stats.shapiro(arr) if 3 <= n <= 5000 else (np.nan, np.nan)
    except Exception:
        sh_stat, sh_p = (np.nan, np.nan)
    q1, med, q3 = np.percentile(arr, [25, 50, 75])
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    whisker_lower = np.min(arr[arr >= lower_fence]) if np.any(arr >= lower_fence) else np.min(arr)
    whisker_upper = np.max(arr[arr <= upper_fence]) if np.any(arr <= upper_fence) else np.max(arr)
    return {
        "label": label, "n": n, "sum": np.sum(arr), "mean": mean, "sd": sd, "var": np.var(arr, ddof=1) if n > 1 else np.nan,
        "min": np.min(arr), "q1": q1, "median": med, "q3": q3, "max": np.max(arr),
        "whisker_lower": whisker_lower, "whisker_upper": whisker_upper,
        "ci_half": ci_half, "ci_lower": mean - ci_half if pd.notna(ci_half) else np.nan, "ci_upper": mean + ci_half if pd.notna(ci_half) else np.nan,
        "tol_lower": tol_lower, "tol_upper": tol_upper, "ad_stat": ad_stat, "ad_p": ad_p, "shapiro_stat": sh_stat, "shapiro_p": sh_p,
    }


def _f_test_equal_var(a, b):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    v1 = np.var(a, ddof=1); v2 = np.var(b, ddof=1)
    if np.isnan(v1) or np.isnan(v2) or v1 == 0 or v2 == 0:
        return np.nan, np.nan
    if v1 >= v2:
        fstat = v1 / v2; dfn, dfd = len(a) - 1, len(b) - 1
    else:
        fstat = v2 / v1; dfn, dfd = len(b) - 1, len(a) - 1
    p = 2 * min(stats.f.cdf(fstat, dfn, dfd), 1 - stats.f.cdf(fstat, dfn, dfd))
    return fstat, min(p, 1.0)


def _anova_two_groups(a, b):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    n1, n2 = len(a), len(b); allv = np.concatenate([a, b]); grand = np.mean(allv); m1, m2 = np.mean(a), np.mean(b)
    ss_between = n1 * (m1 - grand) ** 2 + n2 * (m2 - grand) ** 2
    ss_within = np.sum((a - m1) ** 2) + np.sum((b - m2) ** 2)
    ss_total = np.sum((allv - grand) ** 2)
    df_between = 1; df_within = n1 + n2 - 2; df_total = n1 + n2 - 1
    ms_between = ss_between / df_between; ms_within = ss_within / df_within if df_within > 0 else np.nan
    f_stat = ms_between / ms_within if ms_within and ms_within > 0 else np.nan
    p = 1 - stats.f.cdf(f_stat, df_between, df_within) if pd.notna(f_stat) else np.nan
    return pd.DataFrame({
        "Source of Variation": ["Between Groups", "Within Groups", "Total"],
        "SS": [ss_between, ss_within, ss_total],
        "df": [df_between, df_within, df_total],
        "MS": [ms_between, ms_within, np.nan],
        "F": [f_stat, np.nan, np.nan],
        "P-Value": [p, np.nan, np.nan],
    }), ms_within, ss_between, ss_total


def _acceptance_band(ref, test, alpha_level=0.05):
    ref = np.asarray(ref, dtype=float); test = np.asarray(test, dtype=float)
    n1, n2 = len(ref), len(test); m1 = np.mean(ref); v1 = np.var(ref, ddof=1); v2 = np.var(test, ddof=1)
    sp2 = ((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2); se_diff = np.sqrt((1 / n1 + 1 / n2) * sp2); tcrit = t.ppf(1 - alpha_level / 2, n1 + n2 - 2)
    return m1 - tcrit * se_diff, m1 + tcrit * se_diff


def _graphical_summary_figure(stats_list, title, tol_cov, tol_conf, mean_ci_conf, shaded_range=None, shaded_label=None):
    cfg = common.safe_get_plot_cfg("Descriptive summary")
    colors = [cfg["primary_color"], cfg["secondary_color"], cfg["tertiary_color"]]
    labels = [s["label"] for s in stats_list]
    mins, maxs = [], []
    for s in stats_list:
        for key in ["min", "whisker_lower", "q1", "mean", "tol_lower", "ci_lower"]:
            if pd.notna(s.get(key, np.nan)): mins.append(s[key])
        for key in ["max", "whisker_upper", "q3", "mean", "tol_upper", "ci_upper"]:
            if pd.notna(s.get(key, np.nan)): maxs.append(s[key])
    sr = None
    if shaded_range is not None:
        sr = np.asarray(shaded_range, dtype=float).ravel()
        if sr.size == 2 and np.all(np.isfinite(sr)):
            mins += [float(np.min(sr))]; maxs += [float(np.max(sr))]
        else:
            sr = None
    x_min = min(mins) if mins else 0.0; x_max = max(maxs) if maxs else 1.0; pad = 0.08 * (x_max - x_min if x_max > x_min else 1); x_lo, x_hi = x_min - pad, x_max + pad
    fig, (ax, axr) = plt.subplots(1, 2, figsize=(max(cfg["fig_w"] * 1.95, 13), max(cfg["fig_h"] * 1.55, 7.2)), gridspec_kw={"width_ratios": [1.6, 1]})
    density_y0 = 6.35; row_centers = [5.25, 4.35, 3.45, 2.55, 1.65, 0.75]
    row_names = ["Whisker Min/Max", "Min/Max", "Mean ± 3SD", "IQR (Q1, Q3)", f"{tol_cov}%/{tol_conf}% Tol. Interval", f"{mean_ci_conf}% CI for Mean"]
    if sr is not None:
        ax.axvspan(sr[0], sr[1], color=cfg["band_color"], alpha=0.18)
        ax.axvline(sr[0], color=cfg["secondary_color"], ls=cfg["aux_line_style"], lw=cfg["aux_line_width"])
        ax.axvline(sr[1], color=cfg["secondary_color"], ls=cfg["aux_line_style"], lw=cfg["aux_line_width"])
        if shaded_label:
            ax.text(float(np.mean(sr)), 6.55, shaded_label, color=cfg["secondary_color"], ha="center", va="bottom", fontsize=9, bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=2))
    xgrid = np.linspace(x_lo, x_hi, 600)
    for i, s in enumerate(stats_list):
        arr = s["raw"]; col = colors[i]
        if len(np.unique(arr)) > 1 and len(arr) >= 3:
            try:
                dens = gaussian_kde(arr)(xgrid); dens = dens / dens.max() * 0.85
            except Exception:
                dens = np.zeros_like(xgrid)
        else:
            dens = np.zeros_like(xgrid)
        ax.plot(xgrid, density_y0 + dens, color=col, lw=cfg["line_width"], ls=cfg["line_style"])
        ax.hlines(density_y0, x_lo, x_hi, color="#111827", lw=0.8)
    offsets = [0.10, -0.10] if len(stats_list) > 1 else [0.0]
    for ridx, yc in enumerate(row_centers):
        ax.hlines(yc - 0.37, x_lo, x_hi, color="#d1d5db", lw=0.8)
        for i, s in enumerate(stats_list):
            yy = yc + offsets[i]; col = colors[i]; ms = max(4, cfg["marker_size"] / 12)
            if ridx == 0:
                ax.hlines(yy, s["whisker_lower"], s["whisker_upper"], color=col, lw=cfg["line_width"], ls=cfg["line_style"]); ax.plot(s["median"], yy, 'o', color=col, ms=ms)
            elif ridx == 1:
                ax.hlines(yy, s["min"], s["max"], color=col, lw=cfg["line_width"], ls=cfg["line_style"]); ax.plot(s["median"], yy, 'o', color=col, ms=ms)
            elif ridx == 2:
                lo = s["mean"] - 3 * s["sd"] if pd.notna(s["sd"]) else np.nan; hi = s["mean"] + 3 * s["sd"] if pd.notna(s["sd"]) else np.nan
                if pd.notna(lo) and pd.notna(hi): ax.hlines(yy, lo, hi, color=col, lw=cfg["line_width"], ls=cfg["line_style"])
                ax.plot(s["mean"], yy, 'o', color=col, ms=max(4.5, cfg["marker_size"] / 10))
            elif ridx == 3:
                ax.hlines(yy, s["q1"], s["q3"], color=col, lw=cfg["line_width"] + 0.2, ls=cfg["line_style"]); ax.plot(s["median"], yy, 'o', color=col, ms=ms)
            elif ridx == 4:
                ax.hlines(yy, s["tol_lower"], s["tol_upper"], color=col, lw=cfg["line_width"], ls=cfg["line_style"]); ax.plot(s["mean"], yy, 'o', color=col, ms=max(4.5, cfg["marker_size"] / 10))
            else:
                ax.hlines(yy, s["ci_lower"], s["ci_upper"], color=col, lw=cfg["line_width"], ls=cfg["line_style"]); ax.plot(s["mean"], yy, 'o', color=col, ms=max(4.5, cfg["marker_size"] / 10))
    ax.set_xlim(x_lo, x_hi); ax.set_ylim(0.35, 6.95); ax.set_yticks([density_y0] + row_centers); ax.set_yticklabels(["Normal distribution"] + row_names)
    apply_ax_style(ax, title, "", "", legend=False, plot_key="Descriptive summary")
    ax.grid(axis="x", alpha=cfg["grid_alpha"])
    if cfg["show_legend"] and len(labels) > 1:
        handles = [plt.Line2D([0], [0], color=colors[i], marker='o', lw=cfg["line_width"], ls=cfg["line_style"], label=labels[i]) for i in range(len(labels))]
        ax.legend(handles=handles, frameon=False, loc=cfg["legend_loc"])
    axr.axis("off")
    axr.set_title("Graphical Summary with Descriptive Statistics", fontsize=13, weight="bold", pad=10)
    x0, x1, x2 = 0.02, 0.72, 0.95; y = 0.94
    if len(stats_list) == 1:
        s = stats_list[0]
        rows = [["Normality (AD), p-value", f"{s['ad_p']:.3f}" if pd.notna(s['ad_p']) else "-"],["Normality (Shapiro), p-value", f"{s['shapiro_p']:.3f}" if pd.notna(s['shapiro_p']) else "-"],["Mean", f"{s['mean']:.3f}"],["SD", f"{s['sd']:.3f}"],["N", f"{s['n']:.0f}"],["Variance", f"{s['var']:.3f}"],["Minimum", f"{s['min']:.3f}"],["1st Quartile", f"{s['q1']:.3f}"],["Median", f"{s['median']:.3f}"],["3rd Quartile", f"{s['q3']:.3f}"],["Maximum", f"{s['max']:.3f}"],[f"{tol_cov}%/{tol_conf}% Tol. Int. Lower", f"{s['tol_lower']:.3f}"],[f"{tol_cov}%/{tol_conf}% Tol. Int. Upper", f"{s['tol_upper']:.3f}"],[f"{mean_ci_conf}% LCI for Mean", f"{s['ci_lower']:.3f}"],[f"{mean_ci_conf}% UCI for Mean", f"{s['ci_upper']:.3f}"]]
        axr.text(0.65, 0.98, s["label"], ha="center", va="top", fontsize=12, weight="bold")
        for label, val in rows:
            axr.text(0.02, y, label, ha="left", va="center", fontsize=10.5, weight="bold"); axr.text(0.84, y, val, ha="right", va="center", fontsize=10.5); y -= 0.06
    else:
        s1, s2 = stats_list[:2]
        axr.text(x1, 0.98, s1["label"], ha="center", va="top", fontsize=12, weight="bold"); axr.text(x2, 0.98, s2["label"], ha="center", va="top", fontsize=12, weight="bold")
        rows = [["Normality (AD), p-value", f"{s1['ad_p']:.3f}" if pd.notna(s1['ad_p']) else "-", f"{s2['ad_p']:.3f}" if pd.notna(s2['ad_p']) else "-"],["Mean", f"{s1['mean']:.3f}", f"{s2['mean']:.3f}"],["SD", f"{s1['sd']:.3f}", f"{s2['sd']:.3f}"],["N", f"{s1['n']:.0f}", f"{s2['n']:.0f}"],["Variance", f"{s1['var']:.3f}", f"{s2['var']:.3f}"],["Minimum", f"{s1['min']:.3f}", f"{s2['min']:.3f}"],["1st Quartile", f"{s1['q1']:.3f}", f"{s2['q1']:.3f}"],["Median", f"{s1['median']:.3f}", f"{s2['median']:.3f}"],["3rd Quartile", f"{s1['q3']:.3f}", f"{s2['q3']:.3f}"],["Maximum", f"{s1['max']:.3f}", f"{s2['max']:.3f}"],[f"{tol_cov}%/{tol_conf}% Tol. Int. Lower", f"{s1['tol_lower']:.3f}", f"{s2['tol_lower']:.3f}"],[f"{tol_cov}%/{tol_conf}% Tol. Int. Upper", f"{s1['tol_upper']:.3f}", f"{s2['tol_upper']:.3f}"],[f"{mean_ci_conf}% LCI for Mean", f"{s1['ci_lower']:.3f}", f"{s2['ci_lower']:.3f}"],[f"{mean_ci_conf}% UCI for Mean", f"{s1['ci_upper']:.3f}", f"{s2['ci_upper']:.3f}"]]
        for label, v1, v2 in rows:
            axr.text(x0, y, label, ha="left", va="center", fontsize=10.5, weight="bold"); axr.text(x1, y, v1, ha="center", va="center", fontsize=10.5); axr.text(x2, y, v2, ha="center", va="center", fontsize=10.5); y -= 0.06
    fig.tight_layout()
    return fig


def render():
    render_display_settings()
    st.sidebar.title("🔬 lm Stats")
    st.sidebar.markdown("Stats Suite")
    tool = st.sidebar.radio("Stats tool", TOOLS, key="stats_tool")
    st.sidebar.caption("Use the app navigation to switch between Stats Suite, IVIVC Suite, and DoE Studio.")

    if tool == "📊 Descriptive Statistics":
        app_header("📊 Descriptive Statistics", "Paste one or more numeric columns with headers. For one column, get a graphical summary. For multiple columns, choose a reference and a test column to compare.")
        c1, c2 = st.columns([1, 5])
        with c1:
            st.button("Sample Data", key="sample_desc", on_click=load_sample_text, args=("desc_input", "desc"))
        with c2:
            data_input = st.text_area("Data (paste with headers)", height=220, key="desc_input")
        decimals = st.slider("Decimals", 1, 8, DEFAULT_DECIMALS, key="desc_dec")
        alpha = st.slider("Significance level α", 0.001, 0.100, 0.050, 0.001, key="desc_alpha")
        mean_ci_conf = st.slider("Mean CI confidence (%)", 80, 99, 95, 1, key="desc_mean_ci")
        tol_cov = st.slider("Tolerance interval coverage (%)", 80, 99, 99, 1, key="desc_tol_cov")
        tol_conf = st.slider("Tolerance interval confidence (%)", 80, 99, 95, 1, key="desc_tol_conf")
        if data_input:
            df = parse_pasted_table(data_input, header=True)
            if df is None or df.empty:
                st.error("Could not parse the pasted data.")
            else:
                st.success(f"Loaded {df.shape[0]} rows × {df.shape[1]} columns")
                with st.expander("Preview data"):
                    st.dataframe(df, use_container_width=True)
                numeric_cols = get_numeric_columns(df)
                if len(numeric_cols) == 0:
                    st.error("No numeric columns were detected.")
                else:
                    is_single = len(numeric_cols) == 1
                    if is_single:
                        ref_col = numeric_cols[0]; test_col = None; st.info(f"Single numeric column detected: {ref_col}")
                    else:
                        c1, c2 = st.columns(2)
                        with c1:
                            ref_col = st.selectbox("Reference column", numeric_cols, index=0)
                        with c2:
                            test_candidates = [c for c in numeric_cols if c != ref_col]
                            test_col = st.selectbox("Test column", test_candidates, index=0)
                    ref = to_numeric(df[ref_col]).dropna().to_numpy()
                    if len(ref) < 3:
                        st.error("Reference column must contain at least 3 numeric values.")
                    else:
                        ref_stats = _one_sample_summary(ref, ref_col, ci_conf=mean_ci_conf / 100, tol_p=tol_cov / 100, tol_confidence=tol_conf / 100)
                        ref_stats["raw"] = ref
                        tables = {}; figs = {}
                        summary_tbl = pd.DataFrame({"Groups": [ref_col], "Count": [ref_stats["n"]], "Sum": [ref_stats["sum"]], "Average": [ref_stats["mean"]], "StDev": [ref_stats["sd"]], f"{mean_ci_conf}% CI ±": [ref_stats["ci_half"]]})
                        normality_tbl = pd.DataFrame([
                            {"Test": "Anderson-Darling", "Group": ref_col, "Statistic": ref_stats["ad_stat"], "P-Value": ref_stats["ad_p"], "Comment": "Normally distributed" if pd.notna(ref_stats["ad_p"]) and ref_stats["ad_p"] >= alpha else "Possible non-normality"},
                            {"Test": "Shapiro-Wilk", "Group": ref_col, "Statistic": ref_stats["shapiro_stat"], "P-Value": ref_stats["shapiro_p"], "Comment": "Normally distributed" if pd.notna(ref_stats["shapiro_p"]) and ref_stats["shapiro_p"] >= alpha else "Possible non-normality"},
                        ])
                        st.markdown("### Tables")
                        report_table(summary_tbl, "Summary of Means", decimals)
                        report_table(normality_tbl, "Normality Tests", decimals)
                        tables["Summary of Means"] = summary_tbl; tables["Normality Tests"] = normality_tbl
                        fig = _graphical_summary_figure([ref_stats], f"Graphical Summary: {ref_col}", tol_cov, tol_conf, mean_ci_conf)
                        st.markdown("### Graphical Summary")
                        show_figure(fig); figs["Graphical Summary"] = fig_to_png_bytes(fig); plt.close(fig)
                        if not is_single and test_col is not None:
                            test = to_numeric(df[test_col]).dropna().to_numpy()
                            if len(test) >= 3:
                                test_stats = _one_sample_summary(test, test_col, ci_conf=mean_ci_conf / 100, tol_p=tol_cov / 100, tol_confidence=tol_conf / 100)
                                test_stats["raw"] = test
                                summary_tbl = pd.DataFrame({"Groups": [ref_col, test_col], "Count": [ref_stats["n"], test_stats["n"]], "Sum": [ref_stats["sum"], test_stats["sum"]], "Average": [ref_stats["mean"], test_stats["mean"]], "StDev": [ref_stats["sd"], test_stats["sd"]], f"{mean_ci_conf}% CI ±": [ref_stats["ci_half"], test_stats["ci_half"]]})
                                normality_tbl = pd.DataFrame([
                                    {"Test": "Anderson-Darling", "Group": ref_col, "Statistic": ref_stats["ad_stat"], "P-Value": ref_stats["ad_p"], "Comment": "Normally distributed" if pd.notna(ref_stats["ad_p"]) and ref_stats["ad_p"] >= alpha else "Possible non-normality"},
                                    {"Test": "Anderson-Darling", "Group": test_col, "Statistic": test_stats["ad_stat"], "P-Value": test_stats["ad_p"], "Comment": "Normally distributed" if pd.notna(test_stats["ad_p"]) and test_stats["ad_p"] >= alpha else "Possible non-normality"},
                                    {"Test": "Shapiro-Wilk", "Group": ref_col, "Statistic": ref_stats["shapiro_stat"], "P-Value": ref_stats["shapiro_p"], "Comment": "Normally distributed" if pd.notna(ref_stats["shapiro_p"]) and ref_stats["shapiro_p"] >= alpha else "Possible non-normality"},
                                    {"Test": "Shapiro-Wilk", "Group": test_col, "Statistic": test_stats["shapiro_stat"], "P-Value": test_stats["shapiro_p"], "Comment": "Normally distributed" if pd.notna(test_stats["shapiro_p"]) and test_stats["shapiro_p"] >= alpha else "Possible non-normality"},
                                ])
                                f_stat, f_p = _f_test_equal_var(ref, test); lev_stat, lev_p = stats.levene(ref, test, center="mean")
                                eqvar_tbl = pd.DataFrame([{"Test": "F Test", "Statistic": f_stat, "P-Value": f_p, "Comment": "Equal variances" if pd.notna(f_p) and f_p >= alpha else "Unequal variances"}, {"Test": "Levene's Test (mean)", "Statistic": lev_stat, "P-Value": lev_p, "Comment": "Equal variances" if lev_p >= alpha else "Unequal variances"}])
                                t_eq = stats.ttest_ind(ref, test, equal_var=True); t_welch = stats.ttest_ind(ref, test, equal_var=False); mw = stats.mannwhitneyu(ref, test, alternative="two-sided")
                                comp_tbl = pd.DataFrame([{"Test": "Student t-test", "Statistic": t_eq.statistic, "P-Value": t_eq.pvalue, "Comment": "Difference in means" if t_eq.pvalue < alpha else "No evidence of difference in means"}, {"Test": "Welch t-test", "Statistic": t_welch.statistic, "P-Value": t_welch.pvalue, "Comment": "Difference in means" if t_welch.pvalue < alpha else "No evidence of difference in means"}, {"Test": "Mann-Whitney U", "Statistic": mw.statistic, "P-Value": mw.pvalue, "Comment": "Difference in distributions" if mw.pvalue < alpha else "No evidence of distributional difference"}])
                                anova_tbl, mse, ss_between, ss_total = _anova_two_groups(ref, test)
                                rsq = ss_between / ss_total if ss_total > 0 else np.nan; rsq_adj = 1 - (1 - rsq) * ((len(ref) + len(test) - 1) / (len(ref) + len(test) - 2)) if (len(ref) + len(test) - 2) > 0 and pd.notna(rsq) else np.nan
                                model_tbl = pd.DataFrame({"Pooled SD": [np.sqrt(mse)], "R²": [rsq], "R² (adj)": [rsq_adj]})
                                shaded = _acceptance_band(ref, test, alpha_level=alpha)
                                st.markdown("### Comparison Tables")
                                for cap, tbl in [("Summary of Means", summary_tbl), ("Normality Tests", normality_tbl), ("Equal Variances Test", eqvar_tbl), ("ANOVA", anova_tbl), ("Model Summary (ANOVA)", model_tbl), ("Mean / Distribution Comparison", comp_tbl)]:
                                    report_table(tbl, cap, decimals)
                                fig = _graphical_summary_figure([ref_stats, test_stats], f"Graphical Summary: {ref_col} vs {test_col}", tol_cov, tol_conf, mean_ci_conf, shaded_range=shaded, shaded_label=f"p > {alpha:.3f} zone around {ref_col} mean")
                                st.markdown("### Graphical Summary"); info_box(f"The shaded area is centered on the reference mean and spans the range in which the test mean would remain within the two-sided t-test acceptance zone at α = {alpha:.3f}, using the pooled within-group variance.")
                                show_figure(fig); figs = {"Graphical Summary": fig_to_png_bytes(fig)}; plt.close(fig)
                                tables = {"Summary of Means": summary_tbl, "Normality Tests": normality_tbl, "Equal Variances Test": eqvar_tbl, "ANOVA": anova_tbl, "Model Summary (ANOVA)": model_tbl, "Mean / Distribution Comparison": comp_tbl}
                        export_results(prefix="descriptive_statistics", report_title="Statistical Analysis Report", module_name="Descriptive Statistics", statistical_analysis="Descriptive and comparison statistics were calculated for the selected columns, including normality checks and confidence/tolerance intervals.", offer_text="This module summarizes one or two populations and provides a graphical summary plus common comparison tests.", python_tools="pandas, numpy, scipy.stats, statsmodels, matplotlib, openpyxl, reportlab", table_map=tables, figure_map=figs, conclusion="Review the tables and graphical summary to judge center, spread, normality, and possible differences.", decimals=decimals)

    if tool == "📈 Regression Analysis":
        app_header("📈 Regression Analysis", "Linear regression with CI / PI / both, one-sided or two-sided bands, prediction points, and optional spec-limit crossing.")
        left, right = st.columns([1.45, 1])
        with left:
            c1, c2 = st.columns([1, 5])
            with c1:
                st.button("Sample Data", key="sample_reg", on_click=load_sample_text, args=("reg_xy_input", "reg"))
            with c2:
                xy_input = st.text_area("Paste X and Y data (two Excel columns, with or without headers)", height=220, key="reg_xy_input")
        with right:
            x_pred_text = st.text_area("Predict X (optional)", height=110, placeholder="Paste X values to predict")
        if xy_input:
            try:
                data_df, x_label_detected, y_label_detected = parse_xy(xy_input)
                c1, c2, c3 = st.columns([1, 1, 1.2])
                with c1: interval_mode = st.selectbox("Interval", ["ci", "pi", "both"], format_func=lambda x: {"ci": "CI", "pi": "PI", "both": "Both"}[x])
                with c2: side_mode = st.selectbox("Side", ["upper", "lower", "two-sided"], format_func=lambda x: {"upper": "Upper", "lower": "Lower", "two-sided": "Two-sided"}[x])
                with c3: confidence = st.slider("Confidence", 0.80, 0.99, 0.95, 0.01, format="%.2f")
                c4, c5, c6, c7 = st.columns([1.2, 1.1, 1.1, 0.9])
                with c4: plot_title = st.text_input("Title", value="")
                with c5: xlabel = st.text_input("X label", value=x_label_detected or "X")
                with c6: ylabel = st.text_input("Y label", value=y_label_detected or "Y")
                with c7: point_label = st.text_input("Point label", value="Data")
                c8, c9, c10, c11, c12 = st.columns([0.8, 0.9, 0.9, 0.9, 1.1])
                with c8: y_suffix = st.text_input("Y suffix", value="%")
                with c9: x_min_txt = st.text_input("X min", value="")
                with c10:
                    pred_arr = parse_x_values(x_pred_text)
                    x_max_txt = st.text_input("X max", value="")
                with c11: decimals = st.slider("Decimals", 1, 8, DEFAULT_DECIMALS, key="reg_dec_refined")
                with c12: reg_alpha = st.number_input("Trend test α", min_value=0.0001, max_value=0.2000, value=0.0500, step=0.0050, format="%.4f", key="reg_alpha_trend")
                s1, s2, s3, s4 = st.columns([0.9, 1, 1, 1.2])
                with s1: spec_enabled = st.checkbox("Use spec limit", value=False)
                with s2: spec_value_txt = st.text_input("Spec value", value="3.0", disabled=not spec_enabled)
                with s3: spec_label = st.text_input("Spec label", value="US", disabled=not spec_enabled)
                with s4: crossing_on = st.selectbox("Crossing on", ["auto", "fit", "ci_upper", "ci_lower", "pi_upper", "pi_lower"], disabled=not spec_enabled)
                pred_x = parse_x_values(x_pred_text)
                x_all_max = max(data_df["x"].max(), np.max(pred_x)) if len(pred_x) else data_df["x"].max()
                x_min = parse_optional_float(x_min_txt); x_max = parse_optional_float(x_max_txt)
                if x_min is None: x_min = min(0.0, float(data_df["x"].min()))
                if x_max is None: x_max = x_all_max * 1.15 if x_all_max != 0 else 1.0
                if x_max <= x_min: raise ValueError("X max must be greater than X min.")
                grid_x = np.linspace(x_min, x_max, 500)
                model = reg_fit_linear_model(data_df["x"], data_df["y"])
                reg_stats = regression_anova_and_coefficients_local(data_df["x"], data_df["y"], alpha=reg_alpha)
                grid_df = reg_predict_with_intervals(model, grid_x, confidence=confidence, side=side_mode)
                fig_main, crossing_x = plot_regression_advanced(data_df, model, grid_df, confidence, interval_mode, side_mode, plot_title, xlabel, ylabel, point_label, y_suffix, spec_enabled, parse_optional_float(spec_value_txt) if spec_enabled else None, spec_label, crossing_on)
                show_figure(fig_main)
                summary_tbl = pd.DataFrame({"Intercept": [model["intercept"]], "Slope": [model["slope"]], "R²": [model["r2"]], "Residual SD (s)": [model["s"]], "Degrees of Freedom": [model["df"]]})
                if crossing_x is not None: summary_tbl["Crossing Point"] = [crossing_x]
                report_table(summary_tbl, "Regression model summary", decimals)
                report_table(reg_stats["coefficients"], "Regression coefficients", decimals)
                report_table(reg_stats["anova"], "Regression ANOVA", decimals)
                report_table(data_df.rename(columns={"x": "X Value", "y": "Actual Y"}), "Table 1: Parsed input data", decimals)
                new_pred_x = np.setdiff1d(pred_x, data_df["x"].to_numpy()) if len(pred_x) > 0 else np.array([])
                combined_pts_df = pd.concat([data_df[["x", "y"]], pd.DataFrame({"x": new_pred_x, "y": np.nan})], ignore_index=True) if len(new_pred_x) > 0 else data_df[["x", "y"]].copy()
                combined_pts_df = combined_pts_df.sort_values("x").reset_index(drop=True)
                intervals_df = reg_predict_with_intervals(model, combined_pts_df["x"].unique(), confidence=confidence, side=side_mode)
                final_table_df = pd.merge(combined_pts_df, intervals_df, on="x", how="left")
                final_table_df = final_table_df[[c for c in ["x", "y", "fit", "ci_lower", "ci_upper", "pi_lower", "pi_upper"] if c in final_table_df.columns]]
                final_table_df.columns = ["X Value", "Actual Y", "Fitted Y", "Lower CI", "Upper CI", "Lower PI", "Upper PI"]
                report_table(final_table_df, "Table 2: Fitted values and intervals", decimals)
                fig_res = residual_plot(model["fitted"], model["resid"], xlabel="Fitted values", ylabel="Residuals", title="Residuals vs fitted")
                show_figure(fig_res)
                fig_qq = qq_plot(model["resid"], title="Normal probability plot of regression residuals")
                show_figure(fig_qq)
                export_results(prefix="regression_intervals_refined", report_title="Statistical Analysis Report", module_name="Regression Analysis", statistical_analysis="A simple linear regression model was fitted and confidence/prediction intervals were calculated. A slope significance test and crossing against a specification limit were also supported.", offer_text="This module evaluates linear trends, predictions, uncertainty bands, and optional spec-limit crossing.", python_tools="pandas, numpy, scipy.stats, matplotlib, openpyxl, reportlab", table_map={"Regression Model Summary": summary_tbl, "Regression Coefficients": reg_stats["coefficients"], "Regression ANOVA": reg_stats["anova"], "Parsed Input Data": data_df.rename(columns={"x": "X Value", "y": "Actual Y"}), "Fitted Values and Intervals": final_table_df}, figure_map={"Regression plot": fig_to_png_bytes(fig_main), "Residuals vs fitted": fig_to_png_bytes(fig_res), "Normal probability plot": fig_to_png_bytes(fig_qq)}, conclusion="Review the fitted line, interval bands, regression ANOVA, and residual diagnostics before interpreting the trend.", decimals=decimals)
            except Exception as e:
                st.error(str(e))

    if tool == "⏳ Shelf Life Estimator":
        app_header("⏳ Shelf Life Estimator", "Paste stability data, choose lower or upper specification, and estimate shelf life from fit, CI, or PI crossing.")
        def sl_predict_local(model, x_values, confidence=0.95, one_sided=True):
            x_values = np.asarray(x_values, dtype=float).ravel(); Xg = np.column_stack([np.ones(len(x_values)), x_values]); beta = np.array([model["intercept"], model["slope"]]); fit = Xg @ beta
            h = np.einsum("ij,jk,ik->i", Xg, model["XtX_inv"], Xg); se_mean = model["s"] * np.sqrt(h); se_pred = model["s"] * np.sqrt(1 + h); alpha = 1 - confidence; tcrit = t.ppf(confidence, model["df"]) if one_sided else t.ppf(1 - alpha / 2, model["df"])
            return pd.DataFrame({"x": x_values, "fit": fit, "ci_lower": fit - tcrit * se_mean, "ci_upper": fit + tcrit * se_mean, "pi_lower": fit - tcrit * se_pred, "pi_upper": fit + tcrit * se_pred})
        def sl_find_crossing_local(xv, yv, limit):
            xv = np.asarray(xv, dtype=float); yv = np.asarray(yv, dtype=float); d = yv - limit
            if len(d) == 0: return None
            if d[0] == 0: return float(xv[0])
            for i in range(len(d) - 1):
                if d[i] == 0: return float(xv[i])
                if d[i] * d[i + 1] < 0:
                    x1, x2 = xv[i], xv[i + 1]; y1, y2 = yv[i], yv[i + 1]
                    return float(x1) if y2 == y1 else float(x1 + (limit - y1) * (x2 - x1) / (y2 - y1))
            return None
        def sl_get_bound_column_local(spec_side, shelf_basis):
            if shelf_basis == "fit": return "fit"
            if shelf_basis == "ci": return "ci_lower" if spec_side == "lower" else "ci_upper"
            return "pi_lower" if spec_side == "lower" else "pi_upper"
        def sl_plot_local(data_df, grid_df, spec_side, spec_limit, shelf_basis, show_ci_band, show_pi_band, title, xlabel, ylabel, point_label, y_suffix, spec_label):
            x = data_df["x"].to_numpy(); y = data_df["y"].to_numpy(); fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
            if show_pi_band:
                ax.fill_between(grid_df["x"], grid_df["pi_lower"], grid_df["pi_upper"], color=SECONDARY_COLOR, alpha=0.10, label="PI band")
                ax.plot(grid_df["x"], grid_df["pi_lower"], color=SECONDARY_COLOR, lw=1.0, ls=(0, (4, 4))); ax.plot(grid_df["x"], grid_df["pi_upper"], color=SECONDARY_COLOR, lw=1.0, ls=(0, (4, 4)))
            if show_ci_band:
                ax.fill_between(grid_df["x"], grid_df["ci_lower"], grid_df["ci_upper"], color=BAND_COLOR, alpha=0.15, label="CI band")
                ax.plot(grid_df["x"], grid_df["ci_lower"], color=BAND_COLOR, lw=1.0, ls="--"); ax.plot(grid_df["x"], grid_df["ci_upper"], color=BAND_COLOR, lw=1.0, ls="--")
            ax.scatter(x, y, color=PRIMARY_COLOR, s=50, alpha=0.85, label=point_label, zorder=3); ax.plot(grid_df["x"], grid_df["fit"], color="#2c3e50", lw=2, label="Fitted line")
            bound_col = sl_get_bound_column_local(spec_side, shelf_basis); bound_color = {"fit": "#2c3e50", "ci": BAND_COLOR, "pi": SECONDARY_COLOR}[shelf_basis]
            if shelf_basis != "fit": ax.plot(grid_df["x"], grid_df[bound_col], color=bound_color, lw=2.5, label=f"Shelf-life bound ({bound_col})")
            ax.axhline(spec_limit, color="#27ae60", ls="--", lw=1.5, label=f"Limit ({spec_label})"); shelf_life = sl_find_crossing_local(grid_df["x"].to_numpy(), grid_df[bound_col].to_numpy(), spec_limit)
            if shelf_life is not None: ax.axvline(shelf_life, color="#27ae60", ls=":", lw=1.5)
            if y_suffix: ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, pos: f"{v:.1f}{y_suffix}"))
            apply_ax_style(ax, title or f"Shelf Life Estimator ({'Lower' if spec_side == 'lower' else 'Upper'} Spec)", xlabel, ylabel, legend=True)
            return fig, shelf_life, bound_col
        c1, c2 = st.columns([1.35, 1])
        with c1:
            s1, s2 = st.columns([1, 5])
            with s1: st.button("Sample Data", key="sample_shelf", on_click=load_sample_text, args=("shelf_xy_input", "shelf"))
            with s2: xy_input = st.text_area("Paste Time and Response data (with or without headers)", height=220, key="shelf_xy_input")
        with c2:
            pred_x_text = st.text_area("Predict future X values (optional)", value="30\n36\n48", height=120)
            decimals = st.slider("Decimals", 1, 8, DEFAULT_DECIMALS, key="sl_dec")
        r1c1, r1c2, r1c3 = st.columns([1, 1, 1.15])
        with r1c1: spec_side = st.selectbox("Spec side", ["lower", "upper"], format_func=lambda x: "Lower spec" if x == "lower" else "Upper spec")
        with r1c2: shelf_basis = st.selectbox("Shelf-life on", ["ci", "pi", "fit"], format_func=lambda x: {"ci": "Confidence bound", "pi": "Prediction bound", "fit": "Fit line"}[x])
        with r1c3: confidence = st.slider("Confidence", 0.80, 0.99, 0.95, 0.01, format="%.2f")
        r2c1, r2c2, r2c3, r2c4 = st.columns([1, 1, 1, 1])
        with r2c1: spec_value_txt = st.text_input("Spec value", value="90")
        with r2c2: spec_label = st.text_input("Spec label", value="Spec")
        with r2c3: show_ci_band = st.checkbox("Show CI band", value=True)
        with r2c4: show_pi_band = st.checkbox("Show PI band", value=False)
        plot_title = st.text_input("Title", value="")
        r3c1, r3c2, r3c3, r3c4 = st.columns([1, 1, 1, 0.8])
        with r3c1: xlabel_override = st.text_input("X label", value="")
        with r3c2: ylabel_override = st.text_input("Y label", value="")
        with r3c3: point_label = st.text_input("Point label", value="Data")
        with r3c4: y_suffix = st.text_input("Y suffix", value="%")
        r4c1, r4c2 = st.columns([1, 1])
        with r4c1: x_min_txt = st.text_input("X min", value="")
        with r4c2: x_max_txt = st.text_input("X max", value="")
        if xy_input:
            try:
                data_df, x_label_from_header, y_label_from_header = parse_xy(xy_input)
                xlabel = xlabel_override.strip() or x_label_from_header or "Time"; ylabel = ylabel_override.strip() or y_label_from_header or "Response"; pred_x = parse_x_values(pred_x_text); spec_limit = parse_optional_float(spec_value_txt)
                if spec_limit is None: raise ValueError("Enter a valid specification value.")
                x_data_max = float(data_df["x"].max()); x_future_max = float(np.max(pred_x)) if len(pred_x) > 0 else x_data_max; x_min = parse_optional_float(x_min_txt); x_max = parse_optional_float(x_max_txt)
                if x_min is None: x_min = min(0.0, float(data_df["x"].min()))
                if x_max is None: x_max = max(x_data_max * 3, x_future_max * 1.15, x_data_max + 12)
                if x_max <= x_min: raise ValueError("X max must be greater than X min.")
                model = fit_linear(data_df["x"], data_df["y"]); grid_x = np.linspace(x_min, x_max, 600); grid_df = sl_predict_local(model, grid_x, confidence=confidence, one_sided=True)
                fig_main, shelf_life, bound_col = sl_plot_local(data_df, grid_df, spec_side, spec_limit, shelf_basis, show_ci_band, show_pi_band, plot_title, xlabel, ylabel, point_label, y_suffix, spec_label)
                show_figure(fig_main)
                summary_tbl = pd.DataFrame({"Intercept": [model["intercept"]], "Slope": [model["slope"]], "R²": [model["r2"]], "Residual SD (s)": [model["s"]], "Degrees of Freedom": [model["df"]], "Shelf-life basis": [bound_col], "Confidence": [f"{confidence:.0%} one-sided"], "Estimated Shelf Life": [np.nan if shelf_life is None else shelf_life]})
                report_table(summary_tbl, "Shelf-life estimation summary", decimals)
                report_table(data_df.rename(columns={"x": x_label_from_header, "y": y_label_from_header}), "Table 1: Parsed data", decimals)
                new_pred_x = np.setdiff1d(pred_x, data_df["x"].to_numpy()) if len(pred_x) > 0 else np.array([])
                combined_pts_df = pd.concat([data_df[["x", "y"]], pd.DataFrame({"x": new_pred_x, "y": np.nan})], ignore_index=True) if len(new_pred_x) > 0 else data_df[["x", "y"]].copy()
                combined_pts_df = combined_pts_df.sort_values("x").reset_index(drop=True)
                intervals_df = sl_predict_local(model, combined_pts_df["x"].unique(), confidence=confidence, one_sided=True)
                final_table_df = pd.merge(combined_pts_df, intervals_df, on="x", how="left")
                final_table_df.columns = [xlabel, f"Actual {ylabel}", f"Fitted {ylabel}", "Lower CI", "Upper CI", "Lower PI", "Upper PI"]
                report_table(final_table_df, "Table 2: Fitted values and one-sided bounds", decimals)
                fig_res = residual_plot(model["fitted"], model["resid"], xlabel="Fitted values", ylabel="Residuals", title="Residuals vs fitted")
                show_figure(fig_res); fig_qq = qq_plot(model["resid"], title="Normal probability plot of stability residuals"); show_figure(fig_qq)
                export_results(prefix="shelf_life_refined", report_title="Statistical Analysis Report", module_name="Shelf Life Estimator", statistical_analysis="A linear regression model was fitted to stability data and one-sided confidence/prediction bounds were used to estimate shelf life.", offer_text="This module projects future response and estimates a conservative shelf-life crossing against a selected limit.", python_tools="pandas, numpy, scipy.stats, matplotlib, openpyxl, reportlab", table_map={"Shelf-life Summary": summary_tbl, "Parsed Data": data_df.rename(columns={"x": x_label_from_header, "y": y_label_from_header}), "Fitted Values and One-Sided Bounds": final_table_df}, figure_map={"Shelf-life plot": fig_to_png_bytes(fig_main), "Residuals vs fitted": fig_to_png_bytes(fig_res), "Normal probability plot": fig_to_png_bytes(fig_qq)}, conclusion="Review the fit, one-sided bound selected, and residual diagnostics before finalizing shelf life.", decimals=decimals)
            except Exception as e:
                st.error(str(e))

    if tool == "💊 Dissolution Comparison (f2)":
        app_header("💊 Dissolution Comparison (f2)", "FDA-style point selection, conventional f2 checks, and optional bootstrap / BCa confidence intervals.")
        col1, col2 = st.columns(2)
        with col1:
            c_sample_ref, c_sample_fill = st.columns([1, 5])
            with c_sample_ref: st.button("Sample Data", key="sample_f2", on_click=load_dual_sample_text, args=("f2_ref_input", "f2_ref", "f2_test_input", "f2_test"))
            with c_sample_fill: ref_text = st.text_area("Reference profile table", height=220, key="f2_ref_input")
        with col2:
            test_text = st.text_area("Test profile table", height=220, key="f2_test_input")
        s1, s2, s3 = st.columns([1.1, 1.4, 0.9])
        with s1: include_zero = st.checkbox("Include time zero", value=False)
        with s2: cutoff_mode = st.selectbox("Point selection", ["all", "apply_85"], format_func=lambda x: "Use all common timepoints" if x == "all" else "FDA-style: stop after first point where both are ≥ threshold")
        with s3: threshold = st.number_input("Threshold", value=85.0, step=1.0)
        b1, b2, b3, b4 = st.columns([1, 1.1, 1.1, 0.8])
        with b1: bootstrap_on = st.checkbox("Bootstrap f2 CI", value=False)
        with b2: boot_method = st.selectbox("Bootstrap CI method", ["percentile", "bca", "both"], disabled=not bootstrap_on)
        with b3: boot_conf = st.slider("Bootstrap confidence", 0.80, 0.99, 0.90, 0.01, format="%.2f", disabled=not bootstrap_on)
        with b4: decimals = st.slider("Decimals", 1, 8, 2, key="f2_dec")
        b5, b6, b7 = st.columns([1, 1, 1])
        with b5: boot_n = st.number_input("Resamples", min_value=200, max_value=50000, value=2000, step=100, disabled=not bootstrap_on)
        with b6: boot_seed = st.number_input("Seed", min_value=0, value=123, step=1, disabled=not bootstrap_on)
        with b7: show_units = st.checkbox("Show individual unit traces", value=True)
        p1, p2 = st.columns(2)
        with p1: profile_title = st.text_input("Profile plot title", value="Dissolution Profiles")
        with p2: y_label = st.text_input("Y label", value="% Dissolved")
        if ref_text and test_text:
            try:
                ref_df = dis_parse_profile_table(ref_text); test_df = dis_parse_profile_table(test_text); ref_summary = dis_profile_summary(ref_df); test_summary = dis_profile_summary(test_df)
                merged = dis_merge_profiles(ref_summary, test_summary); selected, _ = dis_select_points(merged, include_zero, cutoff_mode, threshold); f2 = dis_calc_f2(selected["mean_ref"], selected["mean_test"])
                selected = selected.copy(); selected["abs_diff"] = (selected["mean_ref"] - selected["mean_test"]).abs(); selected["sq_diff"] = (selected["mean_ref"] - selected["mean_test"]) ** 2
                fda_tbl, fda_detail_tbl, conventional_ok = dis_fda_checks(ref_df, test_df, selected, threshold=threshold, include_zero=include_zero)
                all_points_tbl = merged.copy(); all_points_tbl["Used for f2"] = np.where(all_points_tbl["Time"].isin(selected["Time"]), "Yes", "No")
                assess_tbl = pd.DataFrame({"Selected timepoints": [len(selected)], "f2 Statistic": [f2], "Conclusion": ["Similar (f2 ≥ 50)" if f2 >= 50 else "Not similar (f2 < 50)"], "FDA-style applicability": ["Applicable" if conventional_ok else "Criteria warning"]})
                fig_main = dis_plot_profiles(ref_df, test_df, ref_summary, test_summary, selected, show_units=show_units, title=profile_title, ylabel=y_label)
                boot_tbl = None; boot_figs = {}
                if bootstrap_on:
                    selected_times = np.sort(selected["Time"].to_numpy(dtype=float)); ref_mat, _ = dis_get_selected_matrix(ref_df, selected_times); test_mat, _ = dis_get_selected_matrix(test_df, selected_times); boot_vals = dis_bootstrap_f2(ref_mat, test_mat, n_boot=int(boot_n), seed=int(boot_seed))
                    rows = [{"Observed f2": f2, "Bootstrap mean f2": float(np.mean(boot_vals)), "Bootstrap median f2": float(np.median(boot_vals)), "Bootstrap SD": float(np.std(boot_vals, ddof=1)), "Resamples": int(boot_n), "Seed": int(boot_seed), "CI confidence": boot_conf}]
                    pct_low = pct_high = bca_low = bca_high = np.nan
                    if boot_method in ["percentile", "both"]:
                        pct_low, pct_high = dis_percentile_interval(boot_vals, conf=boot_conf); rows[0]["Percentile CI lower"] = pct_low; rows[0]["Percentile CI upper"] = pct_high
                        fig_boot_pct = dis_plot_bootstrap_f2_distribution(boot_vals, f2, pct_low, pct_high, ci_label=f"{int(round(boot_conf*100))}%", title="Bootstrap distribution plot (Percentile CI)")
                        if fig_boot_pct is not None: boot_figs["Bootstrap distribution plot (Percentile CI)"] = fig_boot_pct
                    if boot_method in ["bca", "both"]:
                        jack_vals = dis_jackknife_f2(ref_mat, test_mat); bca_low, bca_high, z0, accel = dis_bca_interval(f2, boot_vals, jack_vals, conf=boot_conf); rows[0]["BCa CI lower"] = bca_low; rows[0]["BCa CI upper"] = bca_high; rows[0]["BCa z0"] = z0; rows[0]["BCa acceleration"] = accel
                        fig_boot_bca = dis_plot_bootstrap_f2_distribution(boot_vals, f2, bca_low, bca_high, ci_label=f"{int(round(boot_conf*100))}%", title="Bootstrap distribution plot (BCa CI)")
                        if fig_boot_bca is not None: boot_figs["Bootstrap distribution plot (BCa CI)"] = fig_boot_bca
                    boot_tbl = pd.DataFrame(rows)
                cols = st.columns(4); cols[0].metric("f2", f"{f2:.{decimals}f}"); cols[1].metric("Selected points", f"{len(selected)}"); cols[2].metric("Similarity decision", "Similar" if f2 >= 50 else "Not similar"); cols[3].metric("FDA-style check", "Pass" if conventional_ok else "Warning")
                t1, t2, t3 = st.tabs(["Summary", "FDA criteria & selected points", "Bootstrap"])
                with t1:
                    report_table(merged.rename(columns={"n_units_ref": "Ref. Units (N)", "mean_ref": "Ref. Mean", "sd_ref": "Ref. SD", "cv_pct_ref": "Ref. CV (%)", "n_units_test": "Test Units (N)", "mean_test": "Test Mean", "sd_test": "Test SD", "cv_pct_test": "Test CV (%)"}), "Profile summary table", decimals)
                    report_table(assess_tbl, "f2 assessment", decimals); show_figure(fig_main)
                with t2:
                    report_table(fda_tbl, "FDA-style criteria check", decimals); report_table(fda_detail_tbl, "FDA-style criteria details", decimals); report_table(all_points_tbl, "Common timepoints and whether they were used in f2", decimals); report_table(selected, "Selected points used for f2 calculation", decimals)
                with t3:
                    if bootstrap_on and boot_tbl is not None:
                        report_table(boot_tbl, "Bootstrap f2 confidence intervals", decimals)
                        for _, fig in boot_figs.items(): show_figure(fig)
                    else:
                        st.info("Enable 'Bootstrap f2 CI' above to calculate percentile and/or BCa confidence intervals and show the extra bootstrap graph.")
                table_map = {"Profile Summary": merged, "f2 Assessment": assess_tbl, "FDA Criteria Check": fda_tbl, "FDA Criteria Details": fda_detail_tbl, "Selected Points Used for f2": selected}
                if boot_tbl is not None: table_map["Bootstrap f2 Confidence Intervals"] = boot_tbl
                figure_map = {"Dissolution profiles": fig_to_png_bytes(fig_main)}
                for title, fig in boot_figs.items(): figure_map[title] = fig_to_png_bytes(fig)
                export_results(prefix="dissolution_f2_enhanced", report_title="Statistical Analysis Report", module_name="Dissolution Comparison (f₂)", statistical_analysis="Reference and test dissolution profiles were summarized, f2 was calculated, FDA-style checks were evaluated, and optional bootstrap intervals were supported.", offer_text="This module shows both the conventional f2 result and the criteria behind its use.", python_tools="pandas, numpy, scipy.stats, matplotlib, openpyxl, reportlab", table_map=table_map, figure_map=figure_map, conclusion="Review the selected points, FDA-style criteria, and any bootstrap interval before finalizing similarity.", decimals=decimals)
            except Exception as e:
                st.error(str(e))

    if tool == "⚖️ Two-Sample Tests":
        app_header("⚖️ Two-Sample Tests", "Paste one table with headers, then choose any two sample columns to compare.")
        c1, c2 = st.columns([1, 5])
        with c1: st.button("Sample Data", key="sample_two", on_click=load_sample_text, args=("two_input", "two_sample"))
        with c2: data_input = st.text_area("Data table (with headers)", height=240, key="two_input")
        mode = st.radio("Comparison type", ["Independent samples", "Paired samples"], horizontal=True)
        alpha = st.slider("Significance level α", 0.001, 0.100, 0.05, 0.001)
        decimals = st.slider("Decimals", 1, 8, DEFAULT_DECIMALS, key="two_dec")
        if data_input:
            try:
                df = parse_pasted_table(data_input, header=True); num_cols = get_numeric_columns(df)
                if len(num_cols) < 2: st.error("Please paste at least two numeric columns with headers.")
                else:
                    c1, c2 = st.columns(2)
                    with c1: sample_a = st.selectbox("Sample A", num_cols, index=0)
                    with c2: sample_b = st.selectbox("Sample B", [c for c in num_cols if c != sample_a], index=0)
                    x = to_numeric(df[sample_a]).dropna().to_numpy(); y = to_numeric(df[sample_b]).dropna().to_numpy()
                    if mode == "Paired samples":
                        paired_df = df[[sample_a, sample_b]].copy().apply(to_numeric).dropna(); x = paired_df[sample_a].to_numpy(); y = paired_df[sample_b].to_numpy()
                        if len(x) < 2: raise ValueError("Paired analysis requires at least two complete pairs.")
                    def ad(a):
                        stat, p = normal_ad(a); return stat, p, p >= alpha
                    a1, p1, n1 = ad(x); a2, p2, n2 = ad(y)
                    desc = pd.DataFrame({"Sample": [sample_a, sample_b], "N": [len(x), len(y)], "Mean": [x.mean(), y.mean()], "Std. Deviation": [x.std(ddof=1), y.std(ddof=1)], "Median": [np.median(x), np.median(y)], "Minimum": [x.min(), y.min()], "Maximum": [x.max(), y.max()], "AD A* Statistic": [a1, a2], "AD P-Value": [p1, p2], "Normal at α": ["Yes" if n1 else "No", "Yes" if n2 else "No"]})
                    report_table(desc, "Sample summary and normality checks", decimals)
                    if mode == "Independent samples":
                        lev_stat, lev_p = stats.levene(x, y); equal_var = lev_p >= alpha; t_stat, t_p = stats.ttest_ind(x, y, equal_var=equal_var); mw_stat, mw_p = stats.mannwhitneyu(x, y, alternative="two-sided")
                        tests = pd.DataFrame({"Test": ["Levene test", "Student/Welch t-test", "Mann–Whitney U"], "Statistic": [lev_stat, t_stat, mw_stat], "P-Value": [lev_p, t_p, mw_p], "Conclusion": ["Equal variances" if equal_var else "Unequal variances", "Significant" if t_p < alpha else "Not significant", "Significant" if mw_p < alpha else "Not significant"]})
                    else:
                        d = x - y; ad_d, p_d, nd = ad(d); t_stat, t_p = stats.ttest_rel(x, y)
                        try: w_stat, w_p = stats.wilcoxon(x, y)
                        except Exception: w_stat, w_p = np.nan, np.nan
                        tests = pd.DataFrame({"Test": ["AD test of paired differences", "Paired t-test", "Wilcoxon signed-rank"], "Statistic": [ad_d, t_stat, w_stat], "P-Value": [p_d, t_p, w_p], "Conclusion": ["Normal differences" if nd else "Non-normal differences", "Significant" if t_p < alpha else "Not significant", "Significant" if (pd.notna(w_p) and w_p < alpha) else "Not significant"]})
                    report_table(tests, f"Two-sample test results (α = {alpha})", decimals)
                    fig_box, ax = plt.subplots(figsize=(FIG_W, FIG_H)); ax.boxplot([x, y], tick_labels=[sample_a, sample_b], patch_artist=True); apply_ax_style(ax, "Two-sample comparison", "Sample", "Value", plot_key="Two-sample box plot"); show_figure(fig_box)
                    fig_violin, axv = plt.subplots(figsize=(FIG_W, FIG_H))
                    violin_parts = axv.violinplot([x, y], positions=[1, 2], showmeans=True, showextrema=True)
                    for body, color in zip(violin_parts["bodies"], [PRIMARY_COLOR, SECONDARY_COLOR]):
                        body.set_facecolor(color)
                        body.set_edgecolor(color)
                        body.set_alpha(0.28)
                    for key in ["cmeans", "cmins", "cmaxes", "cbars"]:
                        if key in violin_parts:
                            violin_parts[key].set_color("#111827")
                            violin_parts[key].set_linewidth(1.0)
                    axv.set_xticks([1, 2]); axv.set_xticklabels([sample_a, sample_b])
                    apply_ax_style(axv, "Two-sample violin plot", "Sample", "Value", plot_key="Two-sample box plot")
                    show_figure(fig_violin)
                    fig_dens, ax2 = plt.subplots(figsize=(FIG_W, FIG_H))
                    if len(np.unique(x)) > 1:
                        xs = np.linspace(np.min(x), np.max(x), 200); ax2.plot(xs, gaussian_kde(x)(xs), color=PRIMARY_COLOR, lw=2, label=sample_a)
                    if len(np.unique(y)) > 1:
                        ys = np.linspace(np.min(y), np.max(y), 200); ax2.plot(ys, gaussian_kde(y)(ys), color=SECONDARY_COLOR, lw=2, label=sample_b)
                    apply_ax_style(ax2, "Density comparison", "Value", "Density", legend=True, plot_key="Two-sample density plot"); show_figure(fig_dens)
                    export_results(prefix="two_sample_tests", report_title="Statistical Analysis Report", module_name="Two-Sample Tests", statistical_analysis="Two selected sample columns were compared using normality checks and either independent- or paired-sample tests.", offer_text="This module compares two populations for differences in means or distributions.", python_tools="pandas, numpy, scipy.stats, statsmodels, matplotlib, openpyxl, reportlab", table_map={"Summary": desc, "Tests": tests}, figure_map={"Box plot": fig_to_png_bytes(fig_box), "Violin plot": fig_to_png_bytes(fig_violin), "Density plot": fig_to_png_bytes(fig_dens)}, conclusion="Review both parametric and non-parametric results when deciding whether the two samples differ.", decimals=decimals)
            except Exception as e:
                st.error(str(e))

    if tool == "📐 Two-Way ANOVA / GLM":
        app_header("📐 Two-Way ANOVA / General Linear Model", "Use classic two-way ANOVA when you have two factors only, or switch naturally to a general linear model when you add more factors and/or covariates.")
        c1, c2 = st.columns([1, 5])
        with c1: st.button("Sample Data", key="sample_anova", on_click=load_sample_text, args=("anova_input", "anova"))
        with c2: data_input = st.text_area("Paste data with headers", height=240, key="anova_input")
        decimals = st.slider("Decimals", 1, 8, DEFAULT_DECIMALS, key="anova2_dec")
        if data_input:
            try:
                df = parse_pasted_table(data_input, header=True)
                if df is None or df.empty: raise ValueError("Could not parse the pasted data.")
                all_cols = list(df.columns)
                response = st.selectbox("Response", all_cols, index=min(len(all_cols)-1, all_cols.index("Response") if "Response" in all_cols else len(all_cols)-1))
                remaining = [c for c in all_cols if c != response]
                factor_defaults = [c for c in ["Operator", "Machine"] if c in remaining] or remaining[:2]
                factors = st.multiselect("Categorical factors", remaining, default=factor_defaults)
                numeric_candidates = [c for c in remaining if c not in factors]
                covariate_defaults = [c for c in ["Temp"] if c in numeric_candidates]
                covariates = st.multiselect("Covariates (numeric)", numeric_candidates, default=[])
                interaction_mode = st.selectbox("Interaction terms", ["None", "Two-way among factors", "Full factorial among factors"], index=1)
                d = df[[response] + factors + covariates].copy()
                d[response] = to_numeric(d[response])
                for c in covariates: d[c] = to_numeric(d[c])
                d = d.dropna().copy()
                if len(factors) == 0 and len(covariates) == 0: raise ValueError("Choose at least one factor or one covariate.")
                terms = [f"C({f})" for f in factors] + covariates
                if len(factors) >= 2:
                    if interaction_mode == "Two-way among factors":
                        for i in range(len(factors)):
                            for j in range(i + 1, len(factors)):
                                terms.append(f"C({factors[i]}):C({factors[j]})")
                    elif interaction_mode == "Full factorial among factors":
                        base = " * ".join([f"C({f})" for f in factors])
                        terms = [base] + covariates
                formula = f"Q('{response}') ~ " + " + ".join(terms)
                model = smf.ols(formula, data=d).fit()
                raw_anova = anova_lm(model, typ=2)
                model_kind = "Two-Way ANOVA" if len(factors) == 2 and len(covariates) == 0 else "General Linear Model"
                st.markdown(f"**Mode:** {model_kind}")
                st.code(formula)
                anova_rows = []
                for idx, row in raw_anova.iterrows():
                    label = idx.replace("C(", "").replace(")", "")
                    if idx == "Residual": label = "Error"
                    dfv = row.get("df", np.nan); ss = row.get("sum_sq", np.nan)
                    anova_rows.append({"Source": label, "DF": dfv, "Sum of Squares": ss, "Mean Square": ss / dfv if pd.notna(dfv) and dfv != 0 else np.nan, "F Value": row.get("F", np.nan), "P Value": row.get("PR(>F)", np.nan)})
                anova_tbl = pd.DataFrame(anova_rows)
                coef_tbl = model.summary2().tables[1].reset_index().rename(columns={"index": "Term", "Coef.": "Coefficient", "Std.Err.": "SE", "P>|t|": "P Value", "[0.025": "Lower CI", "0.975]": "Upper CI"})
                summary_tbl = pd.DataFrame({"R²": [model.rsquared], "Adjusted R²": [model.rsquared_adj], "Residual SD": [np.sqrt(model.mse_resid) if model.df_resid > 0 else np.nan], "N": [int(model.nobs)]})
                report_table(summary_tbl, "Model summary", decimals)
                report_table(anova_tbl, f"{model_kind} table", decimals)
                report_table(coef_tbl, "Coefficients", decimals)
                if len(factors) >= 2:
                    cell_summary = d.groupby(factors)[response].agg(["count", "mean", "std", "min", "max"]).reset_index()
                    cell_summary.columns = factors + ["N", "Mean", "Std. Deviation", "Minimum", "Maximum"]
                    report_table(cell_summary, "Cell summary statistics", decimals)
                if len(factors) == 2:
                    fig_inter, ax = plt.subplots(figsize=(FIG_W, FIG_H))
                    for lvl in d[factors[1]].astype(str).unique():
                        sub = d[d[factors[1]].astype(str) == lvl]
                        means = sub.groupby(factors[0])[response].mean().reset_index()
                        ax.plot(means[factors[0]].astype(str), means[response], marker='o', lw=2, label=f"{factors[1]} = {lvl}")
                    apply_ax_style(ax, "Interaction plot", factors[0], response, legend=True, plot_key="Two-way ANOVA interaction")
                    show_figure(fig_inter)
                else:
                    fig_inter = None
                fig_res = residual_plot(model.fittedvalues, model.resid, xlabel="Fitted values", ylabel="Residuals", title="Residuals vs fitted")
                show_figure(fig_res); fig_qq = qq_plot(model.resid, title="Normal probability plot of model residuals"); show_figure(fig_qq)
                figure_map = {"Residuals vs fitted": fig_to_png_bytes(fig_res), "Normal probability plot": fig_to_png_bytes(fig_qq)}
                if fig_inter is not None: figure_map["Interaction plot"] = fig_to_png_bytes(fig_inter)
                table_map = {"Model Summary": summary_tbl, f"{model_kind} Table": anova_tbl, "Coefficients": coef_tbl}
                if len(factors) >= 2: table_map["Cell Summary"] = cell_summary
                export_results(prefix="glm_module", report_title="Statistical Analysis Report", module_name=model_kind, statistical_analysis="A linear model was fitted using selected categorical factors and optional numeric covariates. ANOVA and coefficient tables were calculated from the fitted model.", offer_text="This module works as classic two-way ANOVA for two factors, and naturally extends to a general linear model when more factors or covariates are added.", python_tools="pandas, numpy, statsmodels, matplotlib, openpyxl, reportlab", table_map=table_map, figure_map=figure_map, conclusion="Review factor effects, covariate effects, interaction terms, and residual diagnostics before interpreting the model.", decimals=decimals)
            except Exception as e:
                st.error(str(e))

    if tool == "🎯 Tolerance & Confidence Intervals":
        app_header("🎯 Tolerance & Confidence Intervals", "Summarize central tendency, normal-theory confidence intervals, tolerance intervals, and supporting distribution diagnostics.")
        c1, c2 = st.columns([1, 5])
        with c1: st.button("Sample Data", key="sample_ti", on_click=load_sample_text, args=("ti_input", "ti"))
        with c2: data_input = st.text_area("Paste one table with headers", height=240, key="ti_input")
        confidence = st.slider("Confidence level (%)", 80, 99, 95)
        coverage = st.slider("Population coverage for TI (%)", 80, 99, 95)
        paired = st.checkbox("Paired comparison for difference in means", value=False)
        decimals = st.slider("Decimals", 1, 8, DEFAULT_DECIMALS, key="ti_dec")
        if data_input:
            try:
                df = parse_pasted_table(data_input, header=True); num_cols = get_numeric_columns(df)
                if not num_cols:
                    st.error("No numeric columns were found.")
                else:
                    c1, c2 = st.columns(2)
                    with c1: sample_a = st.selectbox("Sample A", num_cols, index=0)
                    with c2: sample_b = st.selectbox("Sample B (optional)", ["(None)"] + [c for c in num_cols if c != sample_a], index=0)
                    alpha = 1 - confidence / 100; p = coverage / 100; conf = confidence / 100

                    def mean_ci(a):
                        n = len(a); m = np.mean(a); s = np.std(a, ddof=1); se = s / np.sqrt(n); tcrit = t.ppf(1 - alpha / 2, n - 1)
                        return m, s, m - tcrit * se, m + tcrit * se

                    sample_arrays = []
                    x = to_numeric(df[sample_a]).dropna().to_numpy()
                    if len(x) < 2:
                        raise ValueError("Sample A must contain at least two numeric values.")
                    sample_arrays.append((sample_a, x))
                    y = None
                    if sample_b != "(None)":
                        if paired:
                            pair_df = df[[sample_a, sample_b]].copy().apply(to_numeric).dropna(); x = pair_df[sample_a].to_numpy(); y = pair_df[sample_b].to_numpy()
                            if len(x) < 2:
                                raise ValueError("Paired analysis requires at least two complete pairs.")
                            sample_arrays = [(sample_a, x), (sample_b, y)]
                        else:
                            y = to_numeric(df[sample_b]).dropna().to_numpy()
                            if len(y) < 2:
                                raise ValueError("Sample B must contain at least two numeric values.")
                            sample_arrays.append((sample_b, y))

                    stats_objs = []
                    interval_rows = []
                    detail_rows = []
                    normality_rows = []
                    for label, arr in sample_arrays:
                        s = _one_sample_summary(arr, label, ci_conf=conf, tol_p=p, tol_confidence=conf)
                        s["raw"] = arr
                        stats_objs.append(s)
                        interval_rows.append({
                            "Sample": label,
                            "N": s["n"],
                            "Mean": s["mean"],
                            "Std. Deviation": s["sd"],
                            f"{confidence}% CI Lower": s["ci_lower"],
                            f"{confidence}% CI Upper": s["ci_upper"],
                            f"{coverage}%/{confidence}% TI Lower": s["tol_lower"],
                            f"{coverage}%/{confidence}% TI Upper": s["tol_upper"],
                        })
                        detail_rows.append({
                            "Sample": label,
                            "Minimum": s["min"],
                            "Q1": s["q1"],
                            "Median": s["median"],
                            "Q3": s["q3"],
                            "Maximum": s["max"],
                            "Range": s["max"] - s["min"],
                        })
                        normality_rows.append({
                            "Sample": label,
                            "Anderson-Darling Statistic": s["ad_stat"],
                            "Anderson-Darling P-Value": s["ad_p"],
                            "Shapiro-Wilk Statistic": s["shapiro_stat"],
                            "Shapiro-Wilk P-Value": s["shapiro_p"],
                            "Comment": "No strong normality concern" if ((pd.notna(s["ad_p"]) and s["ad_p"] >= alpha) or (pd.notna(s["shapiro_p"]) and s["shapiro_p"] >= alpha)) else "Check normality visually and analytically",
                        })

                    interval_tbl = pd.DataFrame(interval_rows)
                    detail_tbl = pd.DataFrame(detail_rows)
                    normality_tbl = pd.DataFrame(normality_rows)
                    report_table(detail_tbl, "Descriptive summary", decimals)
                    report_table(interval_tbl, "Confidence and tolerance intervals", decimals)
                    report_table(normality_tbl, "Normality review", decimals)
                    table_map = {
                        "Descriptive Summary": detail_tbl,
                        "Confidence and Tolerance Intervals": interval_tbl,
                        "Normality Review": normality_tbl,
                    }

                    diff_tbl = None
                    if sample_b != "(None)":
                        if paired:
                            d = x - y; md, sd, ld, ud = mean_ci(d)
                            diff_tbl = pd.DataFrame({
                                "Comparison": [f"{sample_a} - {sample_b}"],
                                "Mean Difference": [md],
                                "Std. Deviation of Differences": [sd],
                                f"{confidence}% CI Lower": [ld],
                                f"{confidence}% CI Upper": [ud],
                            })
                        else:
                            nx, ny = len(x), len(y); dx = x.mean() - y.mean(); sx2, sy2 = x.var(ddof=1), y.var(ddof=1)
                            se = np.sqrt(sx2 / nx + sy2 / ny)
                            dfw = (sx2 / nx + sy2 / ny) ** 2 / (((sx2 / nx) ** 2) / (nx - 1) + ((sy2 / ny) ** 2) / (ny - 1))
                            tcrit = t.ppf(1 - alpha / 2, dfw)
                            diff_tbl = pd.DataFrame({
                                "Comparison": [f"{sample_a} - {sample_b}"],
                                "Mean Difference": [dx],
                                f"{confidence}% CI Lower": [dx - tcrit * se],
                                f"{confidence}% CI Upper": [dx + tcrit * se],
                            })
                        report_table(diff_tbl, "Confidence interval for mean difference", decimals)
                        table_map["Mean Difference CI"] = diff_tbl

                    labels = [s["label"] for s in stats_objs]
                    data_list = [s["raw"] for s in stats_objs]

                    fig_interval, ax = plt.subplots(figsize=(FIG_W, FIG_H))
                    colors = [PRIMARY_COLOR, SECONDARY_COLOR, TERTIARY_COLOR]
                    for i, s in enumerate(stats_objs, start=1):
                        col = colors[(i - 1) % len(colors)]
                        ax.plot([s["tol_lower"], s["tol_upper"]], [i, i], color=col, alpha=0.25, lw=8, solid_capstyle="round")
                        ax.plot([s["ci_lower"], s["ci_upper"]], [i, i], color=col, lw=4, solid_capstyle="round")
                        ax.scatter([s["mean"]], [i], color=col, s=80, zorder=3)
                    ax.set_yticks(range(1, len(stats_objs) + 1))
                    ax.set_yticklabels(labels)
                    apply_ax_style(ax, "Mean with confidence and tolerance intervals", "Value", "Sample", plot_key="Tolerance/CI box plot")
                    show_figure(fig_interval, "Mean with confidence and tolerance intervals")

                    fig_box, ax = plt.subplots(figsize=(FIG_W, FIG_H))
                    ax.boxplot(data_list, tick_labels=labels, patch_artist=True)
                    apply_ax_style(ax, "Sample distributions", "Sample", "Value", plot_key="Tolerance/CI box plot")
                    show_figure(fig_box, "Sample distributions")

                    fig_violin, ax = plt.subplots(figsize=(FIG_W, FIG_H))
                    violin_parts = ax.violinplot(data_list, positions=np.arange(1, len(data_list) + 1), showmeans=True, showextrema=True)
                    palette = colors[:len(data_list)]
                    for body, color in zip(violin_parts["bodies"], palette):
                        body.set_facecolor(color)
                        body.set_edgecolor(color)
                        body.set_alpha(0.25)
                    for key in ["cmeans", "cmins", "cmaxes", "cbars"]:
                        if key in violin_parts:
                            violin_parts[key].set_color("#111827")
                            violin_parts[key].set_linewidth(1.0)
                    ax.set_xticks(np.arange(1, len(labels) + 1))
                    ax.set_xticklabels(labels)
                    apply_ax_style(ax, "Violin plot of sample distributions", "Sample", "Value", plot_key="Tolerance/CI box plot")
                    show_figure(fig_violin, "Violin plot of sample distributions")

                    fig_hist, ax = plt.subplots(figsize=(FIG_W, FIG_H))
                    bins = max(5, min(12, int(np.sqrt(max(len(arr) for arr in data_list))) + 2))
                    for i, arr in enumerate(data_list):
                        col = colors[i % len(colors)]
                        ax.hist(arr, bins=bins, density=True, alpha=0.20, color=col, label=f"{labels[i]} histogram")
                        if len(np.unique(arr)) > 1:
                            xs = np.linspace(np.min(arr), np.max(arr), 240)
                            ax.plot(xs, gaussian_kde(arr)(xs), color=col, lw=2, label=f"{labels[i]} density")
                    apply_ax_style(ax, "Histogram and density view", "Value", "Density", legend=True, plot_key="Tolerance/CI box plot")
                    show_figure(fig_hist, "Histogram and density view")

                    qq_w = max(FIG_W * len(stats_objs), FIG_W * 1.2)
                    fig_qq, axes = plt.subplots(1, len(stats_objs), figsize=(qq_w, FIG_H))
                    axes = np.atleast_1d(axes)
                    for ax_i, s in zip(axes, stats_objs):
                        stats.probplot(s["raw"], dist="norm", plot=ax_i)
                        apply_ax_style(ax_i, f"Normal probability plot: {s['label']}", "Theoretical quantiles", "Ordered values", plot_key="Q-Q plot")
                    fig_qq.tight_layout(pad=1.0)
                    show_figure(fig_qq, "Normal probability plots")

                    figure_map = {
                        "Mean with confidence and tolerance intervals": fig_to_png_bytes(fig_interval),
                        "Box plot": fig_to_png_bytes(fig_box),
                        "Violin plot": fig_to_png_bytes(fig_violin),
                        "Histogram and density view": fig_to_png_bytes(fig_hist),
                        "Normal probability plots": fig_to_png_bytes(fig_qq),
                    }

                    export_results(
                        prefix="tolerance_confidence_intervals",
                        report_title="Statistical Analysis Report",
                        module_name="Tolerance & Confidence Intervals",
                        statistical_analysis="Confidence intervals for means and normal-theory tolerance intervals were calculated for the selected samples, with added descriptive summaries, normality review, and supporting graphics.",
                        offer_text="This module summarizes uncertainty around the mean, expected population coverage, and the shape of the observed data.",
                        python_tools="pandas, numpy, scipy.stats, matplotlib, openpyxl, reportlab",
                        table_map=table_map,
                        figure_map=figure_map,
                        conclusion="Review the descriptive summary, interval estimates, and distribution graphics together before using the intervals for decisions.",
                        decimals=decimals,
                    )
            except Exception as e:
                st.error(str(e))

    if tool == "🌐 PCA Analysis":
        app_header("🌐 PCA Analysis", "Reduce multivariate data to principal components and visualize scores and loadings.")
        c1, c2 = st.columns([1, 5])
        with c1: st.button("Sample Data", key="sample_pca", on_click=load_sample_text, args=("pca_input", "pca"))
        with c2: data_input = st.text_area("Paste data with headers", height=240, key="pca_input")
        decimals = st.slider("Decimals", 1, 8, DEFAULT_DECIMALS, key="pca_dec")
        if data_input:
            try:
                df = parse_pasted_table(data_input, header=True); num_cols = get_numeric_columns(df); all_cols = list(df.columns)
                c1, c2, c3 = st.columns([1.25, 1, 1])
                with c1: vars_sel = st.multiselect("Numeric variables", num_cols, default=num_cols)
                with c2: label_col = st.selectbox("Label column (optional)", ["(None)"] + all_cols)
                with c3: group_col = st.selectbox("Group column (optional)", ["(None)"] + [c for c in all_cols if c != label_col])
                c4, c5 = st.columns([1, 1])
                with c4: show_ellipses = st.checkbox("Show ellipses", value=True)
                with c5: ellipse_mode = st.selectbox("Ellipse mode", ["Overall", "By group", "Both"], disabled=not show_ellipses)
                if len(vars_sel) >= 2:
                    X = df[vars_sel].apply(to_numeric).dropna(); Z = (X - X.mean()) / X.std(ddof=1); pca = PCA(n_components=2); scores = pca.fit_transform(Z); loadings = pca.components_.T * np.sqrt(pca.explained_variance_); exp = pca.explained_variance_ratio_ * 100
                    eig = pd.DataFrame({"Principal Component": ["PC1", "PC2"], "Eigenvalue": pca.explained_variance_, "Variance Explained (%)": exp, "Cumulative Variance (%)": np.cumsum(exp)})
                    load_df = pd.DataFrame({"Variable": vars_sel, "PC1": loadings[:, 0], "PC2": loadings[:, 1]})
                    report_table(eig, "Eigenvalues and explained variance", decimals); report_table(load_df, "Loading matrix", decimals)
                    scores_df = pd.DataFrame({"PC1": scores[:, 0], "PC2": scores[:, 1]}, index=X.index)
                    if label_col != "(None)": scores_df["Label"] = df.loc[X.index, label_col].astype(str).values
                    if group_col != "(None)": scores_df["Group"] = df.loc[X.index, group_col].astype(str).values
                    score_cfg = common.safe_get_plot_cfg("PCA score plot"); fig_scores, ax = plt.subplots(figsize=(score_cfg["fig_w"], score_cfg["fig_h"])); color_cycle = [score_cfg["primary_color"], score_cfg["secondary_color"], score_cfg["tertiary_color"], "#9467bd", "#8c564b", "#e377c2"]
                    if group_col != "(None)":
                        unique_groups = list(scores_df["Group"].unique())
                        for i, grp in enumerate(unique_groups):
                            col = color_cycle[i % len(color_cycle)]; m = scores_df["Group"] == grp; ax.scatter(scores_df.loc[m, "PC1"], scores_df.loc[m, "PC2"], s=score_cfg["marker_size"], color=col, label=str(grp))
                            if show_ellipses and ellipse_mode in ["By group", "Both"]:
                                draw_conf_ellipse(scores_df.loc[m, ["PC1", "PC2"]].to_numpy(), ax, edgecolor=col, facecolor=col, plot_key="PCA score plot")
                        if show_ellipses and ellipse_mode in ["Overall", "Both"]:
                            draw_conf_ellipse(scores_df[["PC1", "PC2"]].to_numpy(), ax, edgecolor="#111827", facecolor="#111827", plot_key="PCA score plot")
                    else:
                        col = score_cfg["primary_color"]; ax.scatter(scores_df["PC1"], scores_df["PC2"], s=score_cfg["marker_size"], color=col, label="Scores")
                        if show_ellipses:
                            draw_conf_ellipse(scores_df[["PC1", "PC2"]].to_numpy(), ax, edgecolor=col, facecolor=col, plot_key="PCA score plot")
                    if label_col != "(None)":
                        for _, row in scores_df.iterrows(): ax.text(row["PC1"], row["PC2"], str(row["Label"]), fontsize=8)
                    ax.axhline(0, color="#64748b", lw=score_cfg["aux_line_width"], ls=score_cfg["aux_line_style"]); ax.axvline(0, color="#64748b", lw=score_cfg["aux_line_width"], ls=score_cfg["aux_line_style"])
                    apply_ax_style(ax, "PCA score plot", f"PC1 ({exp[0]:.1f}% var)", f"PC2 ({exp[1]:.1f}% var)", legend=(group_col != "(None)"), plot_key="PCA score plot")
                    show_figure(fig_scores)
                    load_cfg = common.safe_get_plot_cfg("PCA loading plot"); fig_load, ax2 = plt.subplots(figsize=(load_cfg["fig_w"], load_cfg["fig_h"])); ax2.axhline(0, color="#64748b", lw=load_cfg["aux_line_width"], ls=load_cfg["aux_line_style"]); ax2.axvline(0, color="#64748b", lw=load_cfg["aux_line_width"], ls=load_cfg["aux_line_style"])
                    for i, var in enumerate(vars_sel):
                        ax2.arrow(0, 0, loadings[i, 0], loadings[i, 1], head_width=load_cfg["arrow_size"], length_includes_head=True, color=load_cfg["primary_color"], lw=load_cfg["line_width"], ls=load_cfg["line_style"]); ax2.text(loadings[i, 0], loadings[i, 1], var)
                    lim = max(1.1, np.max(np.abs(loadings)) * 1.2); ax2.set_xlim(-lim, lim); ax2.set_ylim(-lim, lim); apply_ax_style(ax2, "PCA loading plot", "PC1", "PC2", plot_key="PCA loading plot"); show_figure(fig_load)
                    export_results(prefix="pca_analysis", report_title="Statistical Analysis Report", module_name="PCA Analysis", statistical_analysis="PCA was performed on selected numeric variables after standardization. Score and loading plots were generated, with optional labels, grouping, and confidence ellipses.", offer_text="This module reduces dimensionality and helps visualize clustering, separation, and variable influence patterns.", python_tools="pandas, numpy, sklearn PCA, matplotlib, openpyxl, reportlab", table_map={"Explained Variance": eig, "Loadings": load_df, "Scores": scores_df.reset_index(drop=True)}, figure_map={"PCA score plot": fig_to_png_bytes(fig_scores), "PCA loading plot": fig_to_png_bytes(fig_load)}, conclusion="Use the score plot, loadings, and optional ellipses together to interpret grouping and variable contribution.", decimals=decimals)
            except Exception as e:
                st.error(str(e))
