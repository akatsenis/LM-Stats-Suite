import modules.common as common
from modules.common import *
from statsmodels.stats.multicomp import pairwise_tukeyhsd

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
    '📊 Descriptive Statistics & Intervals',
    '📈 Regression Analysis',
    '⏳ Shelf Life Estimator',
    '📐 Two-Way ANOVA / GLM',
    '🌐 PCA Analysis',
]

SAMPLE_DATA = {
    "desc": "LotA\tLotB\tLotC\n98.2\t97.5\t99.1\n99.1\t98.1\t99.4\n100.4\t99.0\t100.2\n97.8\t98.4\t98.7\n98.9\t97.9\t99.0\n99.5\t98.8\t99.7\n100.1\t99.2\t100.0\n98.7\t98.0\t98.9\n",
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


def _extended_density_grid(arr, n_points=400, tail_scale=3.0):
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 0.0])
    if arr.size == 1 or np.allclose(arr, arr[0]):
        center = float(arr[0])
        pad = max(abs(center) * 0.05, 0.5)
        grid = np.array([center - pad, center, center + pad], dtype=float)
        dens = np.array([0.0, 1.0, 0.0], dtype=float)
        return grid, dens
    kde = gaussian_kde(arr)
    span = float(np.ptp(arr))
    sd = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    bw = float(np.sqrt(np.squeeze(kde.covariance))) if np.ndim(kde.covariance) else float(np.sqrt(kde.covariance))
    pad = max(0.08 * max(span, 1e-9), tail_scale * bw, 0.20 * max(sd, span, 1.0), 1e-6)
    grid = np.linspace(float(np.min(arr)) - pad, float(np.max(arr)) + pad, int(n_points))
    dens = kde(grid)
    dens = np.asarray(dens, dtype=float)
    dens[0] = 0.0
    dens[-1] = 0.0
    return grid, dens



def _draw_closed_violin(ax, data_list, labels, colors, width=0.82):
    positions = np.arange(1, len(data_list) + 1, dtype=float)
    y_mins = []
    y_maxs = []
    for pos, arr, color in zip(positions, data_list, colors):
        arr = np.asarray(arr, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            continue
        grid, dens = _extended_density_grid(arr)
        y_mins.append(float(np.min(grid)))
        y_maxs.append(float(np.max(grid)))
        max_dens = float(np.max(dens)) if dens.size else 0.0
        half_width = (dens / max_dens) * (width / 2.0) if max_dens > 0 else np.zeros_like(dens)
        ax.fill_betweenx(grid, pos - half_width, pos + half_width, facecolor=color, edgecolor=color, alpha=0.25, linewidth=1.2)
        ax.plot(pos - half_width, grid, color=color, linewidth=1.1)
        ax.plot(pos + half_width, grid, color=color, linewidth=1.1)
        mean_val = float(np.mean(arr))
        ax.plot([pos - width * 0.18, pos + width * 0.18], [mean_val, mean_val], color="#111827", linewidth=1.4, solid_capstyle="round")
        ax.scatter([pos], [mean_val], color="#111827", s=20, zorder=3)
        ax.plot([pos, pos], [float(np.min(arr)), float(np.max(arr))], color="#111827", linewidth=1.0, alpha=0.65)
    ax.set_xlim(0.4, len(data_list) + 0.6)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    if y_mins and y_maxs:
        ax.set_ylim(min(y_mins), max(y_maxs))


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


def _one_sample_summary(arr, label, ci_conf=0.95, tol_p=0.99, tol_confidence=0.95, interval_side="two-sided"):
    arr = np.asarray(arr, dtype=float)
    n = len(arr)
    mean = np.mean(arr)
    sd = np.std(arr, ddof=1) if n > 1 else np.nan
    se = sd / np.sqrt(n) if n > 1 else np.nan
    tcrit = t.ppf(1 - (1 - ci_conf) / 2, n - 1) if n > 1 else np.nan
    ci_half = tcrit * se if n > 1 else np.nan
    two_sided = interval_side == "two-sided"
    _, tol_lower, tol_upper = tolerance_interval_normal(arr, p=tol_p, conf=tol_confidence, two_sided=two_sided)
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


def _strong_normality_concern(ad_p, shapiro_p, alpha=0.05):
    checks = []
    if pd.notna(ad_p):
        checks.append(float(ad_p) < alpha)
    if pd.notna(shapiro_p):
        checks.append(float(shapiro_p) < alpha)
    return bool(checks) and all(checks)


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
    base_colors = [cfg["primary_color"], cfg["secondary_color"], cfg["tertiary_color"], "#9467bd", "#8c564b", "#e377c2", "#17becf", "#bcbd22"]
    colors = [base_colors[i % len(base_colors)] for i in range(len(stats_list))]
    labels = [s["label"] for s in stats_list]
    mins, maxs = [], []
    for s in stats_list:
        for key in ["min", "whisker_lower", "q1", "mean", "tol_lower", "ci_lower"]:
            if pd.notna(s.get(key, np.nan)):
                mins.append(s[key])
        for key in ["max", "whisker_upper", "q3", "mean", "tol_upper", "ci_upper"]:
            if pd.notna(s.get(key, np.nan)):
                maxs.append(s[key])

    sr = None
    if shaded_range is not None:
        sr = np.asarray(shaded_range, dtype=float).ravel()
        if sr.size == 2 and np.all(np.isfinite(sr)):
            mins += [float(np.min(sr))]
            maxs += [float(np.max(sr))]
        else:
            sr = None

    x_min = min(mins) if mins else 0.0
    x_max = max(maxs) if maxs else 1.0
    pad = 0.08 * (x_max - x_min if x_max > x_min else 1.0)
    x_lo, x_hi = x_min - pad, x_max + pad

    fig, (ax, axr) = plt.subplots(
        1,
        2,
        figsize=(max(cfg["fig_w"] * 1.55, 11.4), max(cfg["fig_h"] * 1.55, 7.0)),
        gridspec_kw={"width_ratios": [1.15, 1.0]},
    )

    density_label_y = 7.00
    density_base_y = 6.50
    density_height = 0.52
    row_centers = [5.70, 4.80, 3.90, 3.00, 2.10, 1.20]
    row_names = [
        "Whisker Min/Max",
        "Min/Max",
        "Mean ± 3SD",
        "IQR (Q1, Q3)",
        f"{tol_cov}%/{tol_conf}% Tol. Interval",
        f"{mean_ci_conf}% CI for Mean",
    ]

    if sr is not None:
        ax.axvspan(sr[0], sr[1], color=cfg["band_color"], alpha=0.18)
        ax.axvline(sr[0], color=cfg["secondary_color"], ls=cfg["aux_line_style"], lw=cfg["aux_line_width"])
        ax.axvline(sr[1], color=cfg["secondary_color"], ls=cfg["aux_line_style"], lw=cfg["aux_line_width"])
        if shaded_label:
            ax.text(
                float(np.mean(sr)),
                density_label_y + density_height * 0.90,
                shaded_label,
                color=cfg["secondary_color"],
                ha="center",
                va="bottom",
                fontsize=9,
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=2),
            )

    xgrid = np.linspace(x_lo, x_hi, 600)
    for i, s in enumerate(stats_list):
        arr = s["raw"]
        col = colors[i]
        if len(np.unique(arr)) > 1 and len(arr) >= 3:
            try:
                dens = gaussian_kde(arr)(xgrid)
                dens = dens / dens.max() * density_height if np.max(dens) > 0 else np.zeros_like(xgrid)
            except Exception:
                dens = np.zeros_like(xgrid)
        else:
            dens = np.zeros_like(xgrid)
        ax.plot(xgrid, density_base_y + dens, color=col, lw=cfg["line_width"], ls=cfg["line_style"])
    ax.hlines(density_base_y, x_lo, x_hi, color="#111827", lw=0.8)

    separators = [6.50, 5.25, 4.35, 3.45, 2.55, 1.65, 0.75]
    for y_sep in separators:
        ax.hlines(y_sep, x_lo, x_hi, color="#d1d5db", lw=0.8)

    offsets = np.linspace(0.16, -0.16, len(stats_list)) if len(stats_list) > 1 else np.array([0.0])
    for ridx, yc in enumerate(row_centers):
        for i, s in enumerate(stats_list):
            yy = yc + offsets[i]
            col = colors[i]
            ms = max(4, cfg["marker_size"] / 12)
            if ridx == 0:
                ax.hlines(yy, s["whisker_lower"], s["whisker_upper"], color=col, lw=cfg["line_width"], ls=cfg["line_style"])
                ax.plot(s["median"], yy, "o", color=col, ms=ms)
            elif ridx == 1:
                ax.hlines(yy, s["min"], s["max"], color=col, lw=cfg["line_width"], ls=cfg["line_style"])
                ax.plot(s["median"], yy, "o", color=col, ms=ms)
            elif ridx == 2:
                lo = s["mean"] - 3 * s["sd"] if pd.notna(s["sd"]) else np.nan
                hi = s["mean"] + 3 * s["sd"] if pd.notna(s["sd"]) else np.nan
                if pd.notna(lo) and pd.notna(hi):
                    ax.hlines(yy, lo, hi, color=col, lw=cfg["line_width"], ls=cfg["line_style"])
                ax.plot(s["mean"], yy, "o", color=col, ms=max(4.5, cfg["marker_size"] / 10))
            elif ridx == 3:
                ax.hlines(yy, s["q1"], s["q3"], color=col, lw=cfg["line_width"] + 0.2, ls=cfg["line_style"])
                ax.plot(s["median"], yy, "o", color=col, ms=ms)
            elif ridx == 4:
                ax.hlines(yy, s["tol_lower"], s["tol_upper"], color=col, lw=cfg["line_width"], ls=cfg["line_style"])
                ax.plot(s["mean"], yy, "o", color=col, ms=max(4.5, cfg["marker_size"] / 10))
            else:
                ax.hlines(yy, s["ci_lower"], s["ci_upper"], color=col, lw=cfg["line_width"], ls=cfg["line_style"])
                ax.plot(s["mean"], yy, "o", color=col, ms=max(4.5, cfg["marker_size"] / 10))

    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(0.70, 7.55)
    ax.set_yticks([density_label_y] + row_centers)
    ax.set_yticklabels(["Normal distribution"] + row_names)
    apply_ax_style(ax, title, "", "", legend=False, plot_key="Descriptive summary")
    ax.grid(axis="x", alpha=cfg["grid_alpha"])
    if cfg["show_legend"] and len(labels) > 1:
        handles = [
            plt.Line2D([0], [0], color=colors[i], marker="o", lw=cfg["line_width"], ls=cfg["line_style"], label=labels[i])
            for i in range(len(labels))
        ]
        ax.legend(handles=handles, frameon=False, loc=cfg["legend_loc"])

    axr.axis("off")
    axr.set_title("Graphical Summary with Descriptive Statistics", fontsize=13, weight="bold", pad=10)
    n_stats = len(stats_list)
    if n_stats <= 3:
        col_positions = np.linspace(0.68, 0.96, n_stats) if n_stats > 0 else np.array([0.82])
        header_fs = 11.0
        body_fs = 9.6
        row_step = 0.053
    else:
        col_positions = np.linspace(0.66, 0.97, n_stats) if n_stats > 0 else np.array([0.82])
        header_fs = 10.0
        body_fs = 8.8
        row_step = 0.048
    label_x = 0.02
    y = 0.965
    for xpos, s in zip(col_positions, stats_list):
        axr.text(xpos, y, s["label"], ha="center", va="top", fontsize=header_fs, weight="bold")
    y -= 0.055

    rows = [
        ("Normality", None, True),
        ("   AD, p-value", [f"{s['ad_p']:.3f}" if pd.notna(s['ad_p']) else "-" for s in stats_list], False),
        ("   Shapiro, p-value", [f"{s['shapiro_p']:.3f}" if pd.notna(s['shapiro_p']) else "-" for s in stats_list], False),
        ("Mean", [f"{s['mean']:.3f}" for s in stats_list], True),
        ("SD", [f"{s['sd']:.3f}" if pd.notna(s['sd']) else "-" for s in stats_list], True),
        ("N", [f"{s['n']:.0f}" for s in stats_list], True),
        ("Variance", [f"{s['var']:.3f}" if pd.notna(s['var']) else "-" for s in stats_list], True),
        ("Minimum", [f"{s['min']:.3f}" for s in stats_list], True),
        ("1st Quartile", [f"{s['q1']:.3f}" for s in stats_list], True),
        ("Median", [f"{s['median']:.3f}" for s in stats_list], True),
        ("3rd Quartile", [f"{s['q3']:.3f}" for s in stats_list], True),
        ("Maximum", [f"{s['max']:.3f}" for s in stats_list], True),
        (f"{tol_cov}%/{tol_conf}% LTI", [f"{s['tol_lower']:.3f}" if pd.notna(s['tol_lower']) else "-" for s in stats_list], True),
        (f"{tol_cov}%/{tol_conf}% UTI", [f"{s['tol_upper']:.3f}" if pd.notna(s['tol_upper']) else "-" for s in stats_list], True),
        (f"{mean_ci_conf}% LCI for Mean", [f"{s['ci_lower']:.3f}" if pd.notna(s['ci_lower']) else "-" for s in stats_list], True),
        (f"{mean_ci_conf}% UCI for Mean", [f"{s['ci_upper']:.3f}" if pd.notna(s['ci_upper']) else "-" for s in stats_list], True),
    ]
    for label, vals, bold in rows:
        axr.text(label_x, y, label, ha="left", va="center", fontsize=body_fs, weight=("bold" if bold else "normal"))
        if vals is not None:
            for xpos, val in zip(col_positions, vals):
                axr.text(xpos, y, val, ha="center", va="center", fontsize=body_fs)
        y -= row_step
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    return fig




def _anova_multi_groups(sample_arrays):
    labels = [label for label, _ in sample_arrays]
    arrays = [np.asarray(arr, dtype=float) for _, arr in sample_arrays]
    k = len(arrays)
    n_total = int(sum(len(arr) for arr in arrays))
    grand_mean = np.mean(np.concatenate(arrays))
    group_means = [float(np.mean(arr)) for arr in arrays]
    ss_between = float(sum(len(arr) * (m - grand_mean) ** 2 for arr, m in zip(arrays, group_means)))
    ss_within = float(sum(np.sum((arr - np.mean(arr)) ** 2) for arr in arrays))
    ss_total = ss_between + ss_within
    df_between = k - 1
    df_within = n_total - k
    df_total = n_total - 1
    ms_between = ss_between / df_between if df_between > 0 else np.nan
    ms_within = ss_within / df_within if df_within > 0 else np.nan
    f_stat = ms_between / ms_within if pd.notna(ms_within) and ms_within > 0 else np.nan
    p_value = 1 - stats.f.cdf(f_stat, df_between, df_within) if pd.notna(f_stat) else np.nan
    anova_tbl = pd.DataFrame({
        "Source": ["Between Groups", "Within Groups", "Total"],
        "DF": [df_between, df_within, df_total],
        "SS": [ss_between, ss_within, ss_total],
        "MS": [ms_between, ms_within, np.nan],
        "F": [f_stat, np.nan, np.nan],
        "P-Value": [p_value, np.nan, np.nan],
    })
    rsq = ss_between / ss_total if ss_total > 0 else np.nan
    rsq_adj = 1 - (1 - rsq) * ((n_total - 1) / (n_total - k)) if pd.notna(rsq) and (n_total - k) > 0 else np.nan
    model_tbl = pd.DataFrame({
        "Groups": [k],
        "Total N": [n_total],
        "Grand Mean": [grand_mean],
        "Pooled SD": [np.sqrt(ms_within) if pd.notna(ms_within) and ms_within >= 0 else np.nan],
        "R²": [rsq],
        "Adjusted R²": [rsq_adj],
    })
    return anova_tbl, model_tbl




def _tukey_pairwise_figure(sample_arrays, alpha=0.05):
    if len(sample_arrays) < 3:
        return None
    groups = []
    values = []
    for label, arr in sample_arrays:
        arr = np.asarray(arr, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            continue
        groups.extend([label] * arr.size)
        values.extend(arr.tolist())
    if len(set(groups)) < 3:
        return None
    try:
        tukey = pairwise_tukeyhsd(endog=np.asarray(values, dtype=float), groups=np.asarray(groups, dtype=object), alpha=alpha)
        summary = pd.DataFrame(tukey.summary().data[1:], columns=tukey.summary().data[0])
    except Exception:
        return None
    for col in ["meandiff", "p-adj", "lower", "upper"]:
        if col in summary.columns:
            summary[col] = pd.to_numeric(summary[col], errors="coerce")
    reject = summary["reject"].astype(str).str.lower().isin(["true", "1", "yes"]) if "reject" in summary.columns else pd.Series(False, index=summary.index)
    y = np.arange(len(summary), 0, -1, dtype=float)
    fig_h = max(FIG_H, 0.50 * len(summary) + 1.8)
    fig, ax = plt.subplots(figsize=(FIG_W * 1.05, fig_h))
    cfg = common.safe_get_plot_cfg("Descriptive summary")
    ax.axvline(0, color="#64748b", lw=1.0, ls="--")
    colors = np.where(reject.to_numpy(), cfg["secondary_color"], cfg["primary_color"])
    for i, (_, row) in enumerate(summary.iterrows()):
        ax.hlines(y[i], row["lower"], row["upper"], color=colors[i], lw=2.2)
        ax.scatter(row["meandiff"], y[i], color=colors[i], s=46, zorder=3)
    labels = [f"{g1} - {g2}" for g1, g2 in zip(summary["group1"], summary["group2"])]
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_ylim(0.4, len(summary) + 0.6)
    apply_ax_style(ax, "Tukey HSD simultaneous confidence intervals", "Mean difference", "Comparison", legend=False, plot_key="Tolerance/CI box plot")
    ax.grid(axis="x", alpha=cfg["grid_alpha"])
    return fig

def _welch_mean_diff_ci(ref, test, conf=0.95):
    ref = np.asarray(ref, dtype=float)
    test = np.asarray(test, dtype=float)
    nx, ny = len(ref), len(test)
    diff = float(np.mean(ref) - np.mean(test))
    vx = np.var(ref, ddof=1) if nx > 1 else np.nan
    vy = np.var(test, ddof=1) if ny > 1 else np.nan
    se = np.sqrt(vx / nx + vy / ny) if nx > 1 and ny > 1 else np.nan
    if not np.isfinite(se) or se <= 0:
        return diff, np.nan, np.nan
    dfw_num = (vx / nx + vy / ny) ** 2
    dfw_den = (((vx / nx) ** 2) / (nx - 1)) + (((vy / ny) ** 2) / (ny - 1))
    dfw = dfw_num / dfw_den if dfw_den > 0 else np.nan
    tcrit = t.ppf(1 - (1 - conf) / 2, dfw) if pd.notna(dfw) else np.nan
    return diff, diff - tcrit * se if pd.notna(tcrit) else np.nan, diff + tcrit * se if pd.notna(tcrit) else np.nan


def _reference_comparison_table(ref_label, ref, sample_arrays, alpha=0.05, conf=0.95):
    rows = []
    ref = np.asarray(ref, dtype=float)
    for label, arr in sample_arrays[1:]:
        arr = np.asarray(arr, dtype=float)
        f_stat, f_p = _f_test_equal_var(ref, arr)
        try:
            lev_stat, lev_p = stats.levene(ref, arr, center="mean")
        except Exception:
            lev_stat, lev_p = np.nan, np.nan
        try:
            t_eq = stats.ttest_ind(ref, arr, equal_var=True)
            p_student = float(t_eq.pvalue)
        except Exception:
            p_student = np.nan
        try:
            t_welch = stats.ttest_ind(ref, arr, equal_var=False)
            p_welch = float(t_welch.pvalue)
        except Exception:
            p_welch = np.nan
        try:
            mw = stats.mannwhitneyu(ref, arr, alternative="two-sided")
            p_mw = float(mw.pvalue)
        except Exception:
            p_mw = np.nan
        diff, ci_lower, ci_upper = _welch_mean_diff_ci(ref, arr, conf=conf)
        comment = "Difference vs reference" if pd.notna(p_welch) and p_welch < alpha else "No clear difference vs reference"
        rows.append({
            "Reference": ref_label,
            "Comparison": label,
            "Mean Difference (Ref - Sample)": diff,
            f"{int(round(conf * 100))}% CI Lower": ci_lower,
            f"{int(round(conf * 100))}% CI Upper": ci_upper,
            "Student t-test P-Value": p_student,
            "Welch t-test P-Value": p_welch,
            "Mann-Whitney P-Value": p_mw,
            "F Test P-Value": f_p,
            "Levene P-Value": lev_p,
            "Comment": comment,
        })
    return pd.DataFrame(rows)


def _paired_series(ref_series, sample_series):
    pair_df = pd.concat([to_numeric(ref_series), to_numeric(sample_series)], axis=1)
    pair_df.columns = ["Reference", "Sample"]
    return pair_df.dropna()


def _pairwise_assessment_tables(ref_label, numeric_series, selected_cols, alpha=0.05, conf=0.95, include_paired=False):
    variance_rows = []
    test_rows = []
    paired_normality_rows = []
    include_welch_any = False
    include_mw_any = False
    include_wilcoxon_any = False
    ref_series = to_numeric(numeric_series[ref_label])
    ref = ref_series.dropna().to_numpy(dtype=float)
    ci_label_lo = f"{int(round(conf * 100))}% CI Lower"
    ci_label_hi = f"{int(round(conf * 100))}% CI Upper"
    for label in selected_cols[1:]:
        sample_series = to_numeric(numeric_series[label])
        arr = sample_series.dropna().to_numpy(dtype=float)
        f_stat, f_p = _f_test_equal_var(ref, arr)
        try:
            lev_stat, lev_p = stats.levene(ref, arr, center="mean")
        except Exception:
            lev_stat, lev_p = np.nan, np.nan
        chosen_p = lev_p if pd.notna(lev_p) else f_p
        unequal_var_concern = pd.notna(chosen_p) and chosen_p < alpha
        var_decision = "Strong unequal-variance concern" if unequal_var_concern else "No strong equal-variance concern"
        variance_rows.append({
            "Reference": ref_label,
            "Comparison": label,
            "N (Reference)": len(ref),
            "N (Sample)": len(arr),
            "Variance (Reference)": np.var(ref, ddof=1) if len(ref) > 1 else np.nan,
            "Variance (Sample)": np.var(arr, ddof=1) if len(arr) > 1 else np.nan,
            "F Statistic": f_stat,
            "F Test P-Value": f_p,
            "Levene Statistic": lev_stat,
            "Levene P-Value": lev_p,
            "Comment": var_decision,
        })
        try:
            p_student = float(stats.ttest_ind(ref, arr, equal_var=True).pvalue)
        except Exception:
            p_student = np.nan
        try:
            p_welch = float(stats.ttest_ind(ref, arr, equal_var=False).pvalue)
        except Exception:
            p_welch = np.nan
        try:
            p_mw = float(stats.mannwhitneyu(ref, arr, alternative="two-sided").pvalue)
        except Exception:
            p_mw = np.nan
        ref_ad_p = normal_ad(ref)[1] if len(ref) >= 3 else np.nan
        ref_sh_p = stats.shapiro(ref).pvalue if 3 <= len(ref) <= 5000 else np.nan
        samp_ad_p = normal_ad(arr)[1] if len(arr) >= 3 else np.nan
        samp_sh_p = stats.shapiro(arr).pvalue if 3 <= len(arr) <= 5000 else np.nan
        normality_concern = _strong_normality_concern(ref_ad_p, ref_sh_p, alpha) or _strong_normality_concern(samp_ad_p, samp_sh_p, alpha)
        diff, ci_lower, ci_upper = _welch_mean_diff_ci(ref, arr, conf=conf)
        row = {
            "Reference": ref_label,
            "Comparison": label,
            "Mean (Reference)": np.mean(ref) if len(ref) else np.nan,
            "Mean (Sample)": np.mean(arr) if len(arr) else np.nan,
            "Student t-test P-Value": p_student,
        }
        if include_paired:
            row["Mean Difference (Ref - Sample)"] = diff
            row[ci_label_lo] = ci_lower
            row[ci_label_hi] = ci_upper
        if unequal_var_concern:
            row["Welch t-test P-Value"] = p_welch
            include_welch_any = True
        if normality_concern:
            row["Mann-Whitney P-Value"] = p_mw
            include_mw_any = True
        if include_paired:
            pairs = _paired_series(ref_series, sample_series)
            row["Complete Pairs"] = len(pairs)
            if len(pairs) >= 2:
                diffs = pairs.iloc[:, 0].to_numpy(dtype=float) - pairs.iloc[:, 1].to_numpy(dtype=float)
                try:
                    ad_stat_d, ad_p_d = normal_ad(diffs) if len(diffs) >= 3 else (np.nan, np.nan)
                except Exception:
                    ad_stat_d, ad_p_d = np.nan, np.nan
                try:
                    sh_stat_d, sh_p_d = stats.shapiro(diffs) if 3 <= len(diffs) <= 5000 else (np.nan, np.nan)
                except Exception:
                    sh_stat_d, sh_p_d = np.nan, np.nan
                paired_normality_rows.append({
                    "Sample": f"Difference ({ref_label} - {label})",
                    "Anderson-Darling Statistic": ad_stat_d,
                    "Anderson-Darling P-Value": ad_p_d,
                    "Shapiro-Wilk Statistic": sh_stat_d,
                    "Shapiro-Wilk P-Value": sh_p_d,
                    "Comment": "No strong normality concern" if not _strong_normality_concern(ad_p_d, sh_p_d, alpha) else "Strong normality concern",
                })
                row["Paired Mean Difference"] = np.mean(diffs)
                try:
                    row["Paired t-test P-Value"] = float(stats.ttest_rel(pairs.iloc[:, 0], pairs.iloc[:, 1]).pvalue)
                except Exception:
                    row["Paired t-test P-Value"] = np.nan
                diff_normality_concern = _strong_normality_concern(ad_p_d, sh_p_d, alpha)
                if diff_normality_concern:
                    try:
                        if np.allclose(diffs, 0.0):
                            row["Wilcoxon Signed-Rank P-Value"] = np.nan
                        else:
                            row["Wilcoxon Signed-Rank P-Value"] = float(stats.wilcoxon(pairs.iloc[:, 0], pairs.iloc[:, 1]).pvalue)
                    except Exception:
                        row["Wilcoxon Signed-Rank P-Value"] = np.nan
                    include_wilcoxon_any = True
            else:
                row["Paired Mean Difference"] = np.nan
                row["Paired t-test P-Value"] = np.nan
        test_rows.append(row)
    tests_df = pd.DataFrame(test_rows)
    if not include_welch_any and "Welch t-test P-Value" in tests_df.columns:
        tests_df = tests_df.drop(columns=["Welch t-test P-Value"])
    if not include_mw_any and "Mann-Whitney P-Value" in tests_df.columns:
        tests_df = tests_df.drop(columns=["Mann-Whitney P-Value"])
    if not include_wilcoxon_any and "Wilcoxon Signed-Rank P-Value" in tests_df.columns:
        tests_df = tests_df.drop(columns=["Wilcoxon Signed-Rank P-Value"])
    return pd.DataFrame(variance_rows), tests_df, pd.DataFrame(paired_normality_rows)

def render():
    render_display_settings()
    st.sidebar.title("🔬 lm Stats")
    st.sidebar.markdown("Stats Suite")
    tool = st.sidebar.radio("Stats tool", TOOLS, key="stats_tool")
    st.sidebar.caption("Use the app navigation to switch between Stats Suite, IVIVC Suite, and DoE Studio.")

    if tool == "📊 Descriptive Statistics & Intervals":
        app_header("📊 Descriptive Statistics & Intervals", "Summarize one or many numeric samples, calculate confidence and tolerance intervals, and compare selected samples against a reference.")
        c1, c2 = st.columns([1, 5])
        with c1:
            st.button("Sample Data", key="sample_desc", on_click=load_sample_text, args=("desc_input", "desc"))
        with c2:
            data_input = st.text_area("Data (paste with headers)", height=240, key="desc_input")
        p1, p2, p3, p4, p5, p6 = st.columns([0.8, 1.1, 1.1, 1.1, 1.1, 1.2])
        with p1:
            decimals = st.number_input("Decimals", min_value=1, max_value=8, value=int(DEFAULT_DECIMALS), step=1, key="desc_dec")
        with p2:
            alpha = st.number_input("Significance level α", min_value=0.001, max_value=0.100, value=0.050, step=0.001, format="%.3f", key="desc_alpha")
        with p3:
            mean_ci_conf = st.number_input("Mean CI confidence (%)", min_value=80, max_value=99, value=95, step=1, key="desc_mean_ci")
        with p4:
            tol_cov = st.number_input("Tolerance interval coverage (%)", min_value=80, max_value=99, value=99, step=1, key="desc_tol_cov")
        with p5:
            tol_conf = st.number_input("Tolerance interval confidence (%)", min_value=80, max_value=99, value=95, step=1, key="desc_tol_conf")
        with p6:
            interval_side = st.selectbox("Interval side", ["two-sided", "upper", "lower"], index=0, format_func=lambda x: {"two-sided": "Two-sided", "upper": "Upper one-sided", "lower": "Lower one-sided"}[x], key="desc_interval_side")
        if data_input:
            df = parse_pasted_table(data_input, header=True)
            if df is None or df.empty:
                st.error("Could not parse the pasted data.")
            else:
                st.success(f"Loaded {df.shape[0]} rows × {df.shape[1]} columns")
                with st.expander("Preview data"):
                    st.dataframe(df, width='stretch')
                numeric_cols = get_numeric_columns(df)
                if len(numeric_cols) == 0:
                    st.error("No numeric columns were detected.")
                else:
                    csel1, csel2 = st.columns([1, 2])
                    with csel1:
                        ref_col = st.selectbox("Reference column", numeric_cols, index=0)
                    with csel2:
                        other_options = [c for c in numeric_cols if c != ref_col]
                        default_others = other_options
                        compare_cols = st.multiselect("Additional sample columns to include", other_options, default=default_others)
                    selected_cols = [ref_col] + [c for c in compare_cols if c != ref_col]
                    numeric_series = {col: to_numeric(df[col]) for col in selected_cols}
                    sample_arrays = []
                    too_short = []
                    for col in selected_cols:
                        arr = numeric_series[col].dropna().to_numpy()
                        if len(arr) < 2:
                            too_short.append(col)
                        else:
                            sample_arrays.append((col, arr))
                    if too_short:
                        st.error("These selected columns need at least 2 numeric values: " + ", ".join(too_short))
                    elif len(sample_arrays) == 0:
                        st.error("Select at least one usable numeric sample.")
                    else:
                        conf = mean_ci_conf / 100
                        stats_objs = []
                        for label, arr in sample_arrays:
                            s = _one_sample_summary(arr, label, ci_conf=conf, tol_p=tol_cov / 100, tol_confidence=tol_conf / 100, interval_side=interval_side)
                            s["raw"] = arr
                            stats_objs.append(s)
                        ref = sample_arrays[0][1]
                        is_single = len(sample_arrays) == 1
                        paired_compare = False
                        if not is_single:
                            paired_compare = st.checkbox(
                                "Include paired tests using complete reference/sample row pairs",
                                value=False,
                                key="desc_paired_compare",
                            )

                        desc_tbl = pd.DataFrame([{
                            "Sample": s["label"],
                            "N": s["n"],
                            "Sum": s["sum"],
                            "Mean": s["mean"],
                            "Std. Deviation": s["sd"],
                            "Variance": s["var"],
                            "Minimum": s["min"],
                            "Q1": s["q1"],
                            "Median": s["median"],
                            "Q3": s["q3"],
                            "Maximum": s["max"],
                            "Range": s["max"] - s["min"],
                        } for s in stats_objs])

                        ci_title = f"{mean_ci_conf}% CI" if interval_side == "two-sided" else f"{mean_ci_conf}% {interval_side.title()} CI"
                        ti_title = f"{tol_cov}%/{tol_conf}% TI" if interval_side == "two-sided" else f"{tol_cov}%/{tol_conf}% {interval_side.title()} TI"
                        interval_tbl = pd.DataFrame([{
                            "Sample": s["label"],
                            f"{ci_title} Lower": s["ci_lower"],
                            f"{ci_title} Upper": s["ci_upper"],
                            f"{ti_title} Lower": s["tol_lower"],
                            f"{ti_title} Upper": s["tol_upper"],
                        } for s in stats_objs])

                        normality_tbl = pd.DataFrame([{
                            "Sample": s["label"],
                            "Anderson-Darling Statistic": s["ad_stat"],
                            "Anderson-Darling P-Value": s["ad_p"],
                            "Shapiro-Wilk Statistic": s["shapiro_stat"],
                            "Shapiro-Wilk P-Value": s["shapiro_p"],
                            "Comment": "No strong normality concern" if ((pd.notna(s["ad_p"]) and s["ad_p"] >= alpha) or (pd.notna(s["shapiro_p"]) and s["shapiro_p"] >= alpha)) else "Check normality visually and analytically",
                        } for s in stats_objs])

                        table_map = {
                            "Descriptive Summary": desc_tbl,
                            "Confidence and Tolerance Intervals": interval_tbl,
                        }

                        tukey_fig = None
                        if not is_single:
                            anova_tbl, model_tbl = _anova_multi_groups(sample_arrays)
                            variance_tbl, tests_tbl, paired_norm_tbl = _pairwise_assessment_tables(
                                ref_col,
                                numeric_series,
                                selected_cols,
                                alpha=alpha,
                                conf=conf,
                                include_paired=paired_compare,
                            )
                            if paired_compare and not paired_norm_tbl.empty:
                                normality_tbl = pd.concat([normality_tbl, paired_norm_tbl], ignore_index=True)
                            if len(sample_arrays) >= 3:
                                tukey_fig = _tukey_pairwise_figure(sample_arrays, alpha=alpha)
                            if not variance_tbl.empty:
                                table_map["Equal Variance Checks"] = variance_tbl
                            table_map["ANOVA"] = anova_tbl
                            table_map["ANOVA Model Summary"] = model_tbl
                            if not tests_tbl.empty:
                                table_map["Reference Comparison Tests"] = tests_tbl

                        table_map["Normality Review"] = normality_tbl
                        labels = [s["label"] for s in stats_objs]
                        data_list = [s["raw"] for s in stats_objs]
                        colors = [PRIMARY_COLOR, SECONDARY_COLOR, TERTIARY_COLOR, "#9467bd", "#8c564b", "#e377c2", "#17becf", "#bcbd22"]

                        figure_map = {}
                        info_box("This graphical summary combines distribution shape, spread, quartiles, tolerance interval, and confidence interval for the selected sample set in one comparison view.")
                        fig_summary = _graphical_summary_figure(stats_objs, "Graphical Summary", tol_cov, tol_conf, mean_ci_conf)
                        show_figure(fig_summary, "Graphical summary")
                        figure_map["Graphical Summary"] = fig_to_png_bytes(fig_summary)
                        plt.close(fig_summary)

                        info_box("This table summarizes the main descriptive statistics for each selected sample, including center, spread, and quartiles.")
                        report_table(desc_tbl, "Descriptive summary", decimals)
                        info_box("These intervals show the uncertainty around each sample mean and the expected range covering the chosen proportion of the population.")
                        report_table(interval_tbl, "Confidence and tolerance intervals", decimals)
                        info_box("These normality checks support the visual interpretation of the data distribution and help guide parametric test usage. When paired analysis is requested, the normality of paired differences is also included.")
                        report_table(normality_tbl, "Normality review", decimals)

                        if not is_single:
                            if not variance_tbl.empty:
                                info_box("These equal-variance checks compare each selected sample against the reference using both the F test and Levene's test. They help you judge whether classical equal-variance methods are reasonable.")
                                report_table(variance_tbl, "Equal variance checks", decimals)
                            info_box("This ANOVA table tests whether the selected sample means differ overall across all included groups.")
                            report_table(anova_tbl, "ANOVA", decimals)
                            info_box("This summary reports the pooled within-group variation and the fraction of total variability explained by between-sample differences.")
                            report_table(model_tbl, "Model summary (ANOVA)", decimals)
                            if not tests_tbl.empty:
                                pair_msg = " Paired tests are also shown using complete row-wise pairs because paired analysis was requested." if paired_compare else ""
                                info_box("These reference-based hypothesis tests compare each selected sample against the reference using Student's t-test, and, only when justified by diagnostics, Welch's t-test, Mann-Whitney, or Wilcoxon signed-rank." + pair_msg)
                                report_table(tests_tbl, "Reference comparison tests", decimals)

                        info_box("This comparison plot shows the sample mean together with its confidence interval and tolerance interval for every selected sample.")
                        fig_interval, ax = plt.subplots(figsize=(FIG_W, FIG_H))
                        for i, s in enumerate(stats_objs, start=1):
                            col = colors[(i - 1) % len(colors)]
                            ax.plot([s["tol_lower"], s["tol_upper"]], [i, i], color=col, alpha=0.25, lw=8, solid_capstyle="round")
                            ax.plot([s["ci_lower"], s["ci_upper"]], [i, i], color=col, lw=4, solid_capstyle="round")
                            ax.scatter([s["mean"]], [i], color=col, s=80, zorder=3)
                        ax.set_yticks(range(1, len(stats_objs) + 1))
                        ax.set_yticklabels(labels)
                        apply_ax_style(ax, "Mean with confidence and tolerance intervals", "Value", "Sample", plot_key="Tolerance/CI box plot")
                        show_figure(fig_interval, "Mean with confidence and tolerance intervals")
                        figure_map["Mean with confidence and tolerance intervals"] = fig_to_png_bytes(fig_interval)
                        plt.close(fig_interval)

                        info_box("This box plot compares the distribution, median, quartiles, and spread of all selected samples in one view.")
                        fig_box, ax = plt.subplots(figsize=(FIG_W, FIG_H))
                        bp = ax.boxplot(data_list, tick_labels=labels, patch_artist=True)
                        for patch, col in zip(bp["boxes"], [colors[i % len(colors)] for i in range(len(data_list))]):
                            patch.set_facecolor(col)
                            patch.set_alpha(0.22)
                            patch.set_edgecolor(col)
                        apply_ax_style(ax, "Sample distributions", "Sample", "Value", plot_key="Tolerance/CI box plot")
                        show_figure(fig_box, "Sample distributions")
                        figure_map["Box plot"] = fig_to_png_bytes(fig_box)
                        plt.close(fig_box)

                        info_box("This violin plot compares the full estimated distribution shape of all selected samples using a common vertical scale.")
                        fig_violin, ax = plt.subplots(figsize=(FIG_W, FIG_H))
                        palette = [colors[i % len(colors)] for i in range(len(data_list))]
                        _draw_closed_violin(ax, data_list, labels, palette)
                        apply_ax_style(ax, "Violin plot of sample distributions", "Sample", "Value", plot_key="Tolerance/CI box plot")
                        show_figure(fig_violin, "Violin plot of sample distributions")
                        figure_map["Violin plot"] = fig_to_png_bytes(fig_violin)
                        plt.close(fig_violin)

                        info_box("This combined histogram and density view compares the overall shape, overlap, and tails of all selected samples on a common x-axis.")
                        fig_hist, ax = plt.subplots(figsize=(FIG_W, FIG_H))
                        bins = max(5, min(12, int(np.sqrt(max(len(arr) for arr in data_list))) + 2))
                        density_bounds = [_extended_density_grid(arr) for arr in data_list]
                        global_xmin = min(float(np.min(xs)) for xs, _ in density_bounds)
                        global_xmax = max(float(np.max(xs)) for xs, _ in density_bounds)
                        for i, arr in enumerate(data_list):
                            col = colors[i % len(colors)]
                            xs, ys = density_bounds[i]
                            ax.hist(arr, bins=bins, range=(global_xmin, global_xmax), density=True, alpha=0.16, color=col, label=f"{labels[i]} histogram")
                            if len(np.unique(arr)) > 1:
                                ax.plot(xs, ys, color=col, lw=2, label=f"{labels[i]} density")
                        ax.set_xlim(global_xmin, global_xmax)
                        apply_ax_style(ax, "Histogram and density view", "Value", "Density", legend=True, plot_key="Tolerance/CI box plot")
                        show_figure(fig_hist, "Histogram and density view")
                        figure_map["Histogram and density view"] = fig_to_png_bytes(fig_hist)
                        plt.close(fig_hist)

                        if is_single:
                            info_box("This normal probability plot helps assess whether the sample follows an approximately normal distribution.")
                            fig_qq, ax = plt.subplots(figsize=(FIG_W, FIG_H))
                            stats.probplot(data_list[0], dist="norm", plot=ax)
                            apply_ax_style(ax, f"Normal probability plot: {labels[0]}", "Theoretical quantiles", "Ordered values", plot_key="Q-Q plot")
                            show_figure(fig_qq, "Normal probability plot")
                            figure_map["Normal probability plot"] = fig_to_png_bytes(fig_qq)
                            plt.close(fig_qq)

                        export_results(
                            prefix="descriptive_statistics_intervals",
                            report_title="Statistical Analysis Report",
                            module_name="Descriptive Statistics & Intervals",
                            statistical_analysis="Descriptive statistics, mean confidence intervals, normal-theory tolerance intervals, normality checks, equal-variance checks, and reference-based comparisons were calculated for the selected samples. When multiple samples were included, ANOVA was also performed, and optional paired tests were added when requested.",
                            offer_text="This module summarizes one or many populations and supports interval estimation, ANOVA, variance checks, and reference-based comparison in a single workflow.",
                            python_tools="pandas, numpy, scipy.stats, statsmodels, matplotlib, openpyxl, reportlab",
                            table_map=table_map,
                            figure_map=figure_map,
                            conclusion="Review the descriptive tables, interval estimates, combined comparison plots, and ANOVA or reference comparisons together before drawing conclusions.",
                            decimals=decimals,
                        )

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
                    palette = colors[:len(data_list)]
                    _draw_closed_violin(ax, data_list, labels, palette)
                    apply_ax_style(ax, "Violin plot of sample distributions", "Sample", "Value", plot_key="Tolerance/CI box plot")
                    show_figure(fig_violin, "Violin plot of sample distributions")

                    fig_hist, ax = plt.subplots(figsize=(FIG_W, FIG_H))
                    bins = max(5, min(12, int(np.sqrt(max(len(arr) for arr in data_list))) + 2))
                    density_bounds = [_extended_density_grid(arr) for arr in data_list]
                    global_xmin = min(float(np.min(xs)) for xs, _ in density_bounds)
                    global_xmax = max(float(np.max(xs)) for xs, _ in density_bounds)
                    for i, arr in enumerate(data_list):
                        col = colors[i % len(colors)]
                        xs, ys = density_bounds[i]
                        ax.hist(arr, bins=bins, range=(global_xmin, global_xmax), density=True, alpha=0.20, color=col, label=f"{labels[i]} histogram")
                        if len(np.unique(arr)) > 1:
                            ax.plot(xs, ys, color=col, lw=2, label=f"{labels[i]} density")
                    ax.set_xlim(global_xmin, global_xmax)
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
