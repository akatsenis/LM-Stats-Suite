import modules.common as common
from modules.common import *
from itertools import product, combinations

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

DOE_SAMPLE_RESPONSE_DATA = """Block\tTemp\tSpeed\tpH\tCatalyst\tYield\tPurity\tViscosity
1\t60\t100\t5.0\t0.20\t68.4\t95.1\t412
1\t60\t100\t7.0\t0.20\t72.9\t96.4\t396
1\t60\t200\t5.0\t0.20\t77.3\t95.8\t371
1\t60\t200\t7.0\t0.20\t81.8\t97.2\t352
1\t80\t100\t5.0\t0.20\t79.2\t96.7\t365
1\t80\t100\t7.0\t0.20\t84.4\t98.1\t349
1\t80\t200\t5.0\t0.20\t88.1\t97.4\t330
1\t80\t200\t7.0\t0.20\t92.6\t98.8\t312
2\t60\t100\t5.0\t0.35\t70.1\t95.5\t405
2\t60\t100\t7.0\t0.35\t74.6\t96.8\t389
2\t60\t200\t5.0\t0.35\t79.1\t96.2\t364
2\t60\t200\t7.0\t0.35\t83.5\t97.6\t346
2\t80\t100\t5.0\t0.35\t81.0\t97.0\t357
2\t80\t100\t7.0\t0.35\t86.3\t98.4\t341
2\t80\t200\t5.0\t0.35\t89.8\t97.8\t322
2\t80\t200\t7.0\t0.35\t94.1\t99.1\t304
1\t70\t150\t6.0\t0.275\t84.9\t97.6\t344
1\t70\t150\t6.0\t0.275\t85.2\t97.8\t341
2\t70\t150\t6.0\t0.275\t86.0\t98.0\t338
2\t70\t150\t6.0\t0.275\t85.7\t97.9\t339
"""

DOE_NIST_SAMPLE = """Run\tPressure\tH2_WF6\tUniformity\tStress
1\t80\t6\t4.6\t8.04
2\t42\t6\t6.2\t7.78
3\t68.87\t3.17\t3.4\t7.58
4\t15.13\t8.83\t6.9\t7.27
5\t4\t6\t7.3\t6.49
6\t42\t6\t6.4\t7.69
7\t15.13\t3.17\t8.6\t6.66
8\t42\t2\t6.3\t7.16
9\t68.87\t8.83\t5.1\t8.33
10\t42\t10\t5.4\t8.19
11\t42\t6\t5.0\t7.90
"""

FRACTIONAL_DESIGNS = {
    "2^(3-1) Res-III – 4 runs (k=3)": {
        "k": 3, "resolution": "III", "base_k": 2,
        "generators": [(2, [0, 1])],
        "description": "Half-fraction of 2^3. Generator: X3=X1*X2. Main effects confounded with 2-way interactions.",
    },
    "2^(4-1) Res-IV – 8 runs (k=4)": {
        "k": 4, "resolution": "IV", "base_k": 3,
        "generators": [(3, [0, 1, 2])],
        "description": "Half-fraction of 2^4. Generator: X4=X1*X2*X3. Main effects clear of 2-way interactions.",
    },
    "2^(5-2) Res-III – 8 runs (k=5)": {
        "k": 5, "resolution": "III", "base_k": 3,
        "generators": [(3, [0, 1]), (4, [0, 2])],
        "description": "Quarter-fraction of 2^5. Generators: X4=X1*X2, X5=X1*X3. Good for screening.",
    },
    "2^(5-1) Res-V – 16 runs (k=5)": {
        "k": 5, "resolution": "V", "base_k": 4,
        "generators": [(4, [0, 1, 2, 3])],
        "description": "Half-fraction of 2^5. Generator: X5=X1*X2*X3*X4. All 2FI estimable.",
    },
    "2^(6-2) Res-IV – 16 runs (k=6)": {
        "k": 6, "resolution": "IV", "base_k": 4,
        "generators": [(4, [0, 1, 2]), (5, [0, 1, 3])],
        "description": "Quarter-fraction of 2^6. Generators: X5=X1*X2*X3, X6=X1*X2*X4.",
    },
    "2^(7-3) Res-IV – 16 runs (k=7)": {
        "k": 7, "resolution": "IV", "base_k": 4,
        "generators": [(4, [0, 1, 2]), (5, [0, 1, 3]), (6, [0, 2, 3])],
        "description": "1/8-fraction of 2^7. Generators: X5=X1*X2*X3, X6=X1*X2*X4, X7=X1*X3*X4.",
    },
}


def _load_sample_design():
    factor_names = ["Temp", "Speed", "pH", "Catalyst"]
    lows = [60.0, 100.0, 5.0, 0.20]
    highs = [80.0, 200.0, 7.0, 0.35]
    st.session_state["doe_n_factors"] = 4
    st.session_state["doe_blocks"] = 2
    st.session_state["doe_replicates"] = 1
    st.session_state["doe_center_points"] = 2
    st.session_state["doe_randomize"] = True
    st.session_state["doe_seed"] = 123
    for i, name in enumerate(factor_names):
        st.session_state[f"doe_name_{i}"] = name
        st.session_state[f"doe_low_{i}"] = lows[i]
        st.session_state[f"doe_high_{i}"] = highs[i]
    st.session_state["doe_generated_design"] = _build_factorial_design(
        factor_names, lows, highs, blocks=2, center_points=2, replicates=1, randomize=True, seed=123)


def _load_sample_response_text():
    st.session_state["doe_response_input"] = DOE_SAMPLE_RESPONSE_DATA


def _load_nist_sample():
    st.session_state["doe_response_input"] = DOE_NIST_SAMPLE


def _safe_factor_prefix(i):
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return alphabet[i] if i < len(alphabet) else f"F{i+1}"


def _build_factorial_design(factor_names, low_levels, high_levels,
                              blocks=1, center_points=0, replicates=1,
                              randomize=True, seed=123):
    coded = list(product([-1, 1], repeat=len(factor_names)))
    runs = []
    for block in range(1, blocks + 1):
        for rep in range(1, replicates + 1):
            for combo in coded:
                row = {"Block": block, "Replicate": rep, "RunType": "Factorial"}
                for i, name in enumerate(factor_names):
                    row[f"{_safe_factor_prefix(i)} (coded)"] = combo[i]
                    row[name] = low_levels[i] if combo[i] == -1 else high_levels[i]
                runs.append(row)
            for _ in range(center_points):
                row = {"Block": block, "Replicate": rep, "RunType": "Center"}
                for i, name in enumerate(factor_names):
                    row[f"{_safe_factor_prefix(i)} (coded)"] = 0
                    row[name] = (low_levels[i] + high_levels[i]) / 2
                runs.append(row)
    design = pd.DataFrame(runs)
    if randomize and len(design) > 0:
        rng = np.random.default_rng(seed)
        parts = []
        for _, sub in design.groupby("Block", sort=True):
            sub = sub.sample(frac=1, random_state=int(rng.integers(0, 1_000_000))).reset_index(drop=True)
            parts.append(sub)
        design = pd.concat(parts, ignore_index=True)
    design.insert(0, "Run", np.arange(1, len(design) + 1))
    return design


def _build_ccd_design(factor_names, low_levels, high_levels,
                       ccd_type="CCC", center_points=5, randomize=True, seed=123):
    k = len(factor_names)
    alpha_ccc = round((2 ** k) ** 0.25, 4)
    if ccd_type == "CCC":
        axial_alpha = alpha_ccc
        factorial_scale = 1.0
    elif ccd_type == "CCF":
        axial_alpha = 1.0
        factorial_scale = 1.0
    else:  # CCI
        axial_alpha = 1.0
        factorial_scale = round(1.0 / alpha_ccc, 4)
    runs = []
    for combo in product([-1, 1], repeat=k):
        row = {"RunType": "Factorial"}
        for i, name in enumerate(factor_names):
            cv = combo[i] * factorial_scale
            row[f"{_safe_factor_prefix(i)} (coded)"] = round(cv, 4)
            mid = (low_levels[i] + high_levels[i]) / 2
            half = (high_levels[i] - low_levels[i]) / 2
            row[name] = round(mid + cv * half, 6)
        runs.append(row)
    for i in range(k):
        for sign in [-1, 1]:
            row = {"RunType": "Axial"}
            for j, name in enumerate(factor_names):
                cv = sign * axial_alpha if j == i else 0.0
                row[f"{_safe_factor_prefix(j)} (coded)"] = round(cv, 4)
                mid = (low_levels[j] + high_levels[j]) / 2
                half = (high_levels[j] - low_levels[j]) / 2
                row[name] = round(mid + cv * half, 6)
            runs.append(row)
    for _ in range(center_points):
        row = {"RunType": "Center"}
        for i, name in enumerate(factor_names):
            row[f"{_safe_factor_prefix(i)} (coded)"] = 0.0
            row[name] = (low_levels[i] + high_levels[i]) / 2
        runs.append(row)
    design = pd.DataFrame(runs)
    if randomize and len(design) > 0:
        design = design.sample(frac=1, random_state=seed).reset_index(drop=True)
    design.insert(0, "Run", np.arange(1, len(design) + 1))
    return design


def _build_boxbehnken_design(factor_names, low_levels, high_levels,
                               center_points=3, randomize=True, seed=123):
    k = len(factor_names)
    if k < 3:
        raise ValueError("Box-Behnken requires at least 3 factors.")
    runs = []
    for i, j in combinations(range(k), 2):
        for si, sj in product([-1, 1], repeat=2):
            row = {"RunType": "Factorial"}
            for m, name in enumerate(factor_names):
                cv = float(si) if m == i else (float(sj) if m == j else 0.0)
                row[f"{_safe_factor_prefix(m)} (coded)"] = cv
                mid = (low_levels[m] + high_levels[m]) / 2
                half = (high_levels[m] - low_levels[m]) / 2
                row[name] = round(mid + cv * half, 6)
            runs.append(row)
    for _ in range(center_points):
        row = {"RunType": "Center"}
        for i, name in enumerate(factor_names):
            row[f"{_safe_factor_prefix(i)} (coded)"] = 0.0
            row[name] = (low_levels[i] + high_levels[i]) / 2
        runs.append(row)
    design = pd.DataFrame(runs)
    if randomize and len(design) > 0:
        design = design.sample(frac=1, random_state=seed).reset_index(drop=True)
    design.insert(0, "Run", np.arange(1, len(design) + 1))
    return design


def _build_fractional_factorial(factor_names, low_levels, high_levels, design_key,
                                  center_points=0, randomize=True, seed=123):
    spec = FRACTIONAL_DESIGNS[design_key]
    k = spec["k"]
    base_k = spec["base_k"]
    generators = spec["generators"]
    if len(factor_names) != k:
        raise ValueError(f"Selected design requires exactly {k} factors.")
    base_combos = list(product([-1, 1], repeat=base_k))
    runs = []
    for combo in base_combos:
        full_coded = list(combo)
        for fi, base_indices in sorted(generators, key=lambda x: x[0]):
            val = 1
            for bi in base_indices:
                val *= combo[bi]
            while len(full_coded) <= fi:
                full_coded.append(0)
            full_coded[fi] = val
        row = {"RunType": "Factorial"}
        for i, name in enumerate(factor_names):
            cv = float(full_coded[i]) if i < len(full_coded) else 0.0
            row[f"{_safe_factor_prefix(i)} (coded)"] = cv
            mid = (low_levels[i] + high_levels[i]) / 2
            half = (high_levels[i] - low_levels[i]) / 2
            row[name] = round(mid + cv * half, 6)
        runs.append(row)
    for _ in range(center_points):
        row = {"RunType": "Center"}
        for i, name in enumerate(factor_names):
            row[f"{_safe_factor_prefix(i)} (coded)"] = 0.0
            row[name] = (low_levels[i] + high_levels[i]) / 2
        runs.append(row)
    design = pd.DataFrame(runs)
    if randomize and len(design) > 0:
        design = design.sample(frac=1, random_state=seed).reset_index(drop=True)
    design.insert(0, "Run", np.arange(1, len(design) + 1))
    return design


# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _get_term_factors(term):
    term = str(term).strip()
    if term.startswith("C(") or term in ("Intercept", ""):
        return set()
    if term.startswith("I("):
        inner = term[2:-1]
        return {inner.split("**")[0].strip()}
    if ":" in term:
        return set(term.split(":"))
    return {term}


def _can_remove_term(term, all_terms):
    if term == "Intercept":
        return False
    tf = _get_term_factors(term)
    if not tf:
        return True
    for other in all_terms:
        if other == term:
            continue
        of = _get_term_factors(other)
        if of and tf.issubset(of) and len(of) > len(tf):
            return False
    return True


def _backward_stepwise_aic(safe_df, full_terms, response="Response"):
    def try_fit(terms_list):
        f = response + " ~ " + " + ".join(terms_list)
        try:
            return smf.ols(f, data=safe_df).fit()
        except Exception:
            return None
    terms = [t for t in full_terms if t != "Intercept"]
    model = try_fit(terms)
    if model is None:
        return []
    current_aic = model.aic
    steps = [{"Step": "Start", "Action": "Full model (no removal)",
               "AIC": round(current_aic, 3), "Terms": list(terms), "model": model}]
    for step_n in range(1, 30):
        best_aic = current_aic
        best_remove = None
        best_model_sw = None
        for term in terms:
            if not _can_remove_term(term, terms):
                continue
            candidate = [t for t in terms if t != term]
            if not candidate:
                continue
            m = try_fit(candidate)
            if m is None:
                continue
            if m.aic < best_aic:
                best_aic = m.aic
                best_remove = term
                best_model_sw = m
        if best_remove is None:
            break
        terms = [t for t in terms if t != best_remove]
        current_aic = best_aic
        steps.append({"Step": f"Step {step_n}", "Action": f"Remove '{best_remove}'",
                       "AIC": round(current_aic, 3), "Terms": list(terms), "model": best_model_sw})
    return steps


def _lack_of_fit_test(safe_df, factor_cols, model):
    df_work = safe_df[factor_cols + ["Response"]].copy()
    df_work["_key"] = df_work[factor_cols].round(8).apply(lambda r: tuple(r), axis=1)
    group_counts = df_work.groupby("_key").size()
    if not (group_counts > 1).any():
        return None
    ss_pe = 0.0
    df_pe = 0
    for _key, group in df_work.groupby("_key"):
        y = group["Response"].values
        if len(y) > 1:
            ss_pe += float(np.sum((y - y.mean()) ** 2))
            df_pe += len(y) - 1
    ss_resid = float(np.sum(model.resid ** 2))
    ss_lof = max(ss_resid - ss_pe, 0.0)
    n_unique = int(len(group_counts))
    p_params = int(len(model.params))
    df_lof = n_unique - p_params
    if df_lof <= 0 or df_pe <= 0 or ss_pe < 1e-15:
        return None
    ms_lof = ss_lof / df_lof
    ms_pe = ss_pe / df_pe
    if ms_pe <= 0:
        return None
    f_lof = ms_lof / ms_pe
    p_lof = float(1 - stats.f.cdf(f_lof, df_lof, df_pe)) if np.isfinite(f_lof) else np.nan
    return {"ss_lof": ss_lof, "ss_pe": ss_pe, "df_lof": df_lof, "df_pe": df_pe,
            "ms_lof": ms_lof, "ms_pe": ms_pe, "f_lof": f_lof, "p_lof": p_lof,
            "n_unique": n_unique}


def _pretty_term(term, inv_map):
    term = str(term)
    if term == "Residual":
        return "Error"
    if term == "Intercept":
        return "Intercept"
    if term.startswith("C(Block)"):
        return "Block"
    term = term.replace(":", " x ")
    term = term.replace("I(", "").replace(" ** 2)", "^2").replace("**2)", "^2")
    for safe, orig in inv_map.items():
        term = term.replace(safe, orig)
    return term


def _doe_formula(safe_factors, model_type="interaction"):
    terms = list(safe_factors)
    if model_type in ["interaction", "quadratic"]:
        for i in range(len(safe_factors)):
            for j in range(i + 1, len(safe_factors)):
                terms.append(f"{safe_factors[i]}:{safe_factors[j]}")
    if model_type == "quadratic":
        for f in safe_factors:
            terms.append(f"I({f}**2)")
    return "Response ~ " + " + ".join(terms), terms


def _build_anova_with_lof(model, safe_df, safe_factors, inv_map, lof_result):
    base_anova = anova_lm(model, typ=2).reset_index()
    base_anova.columns = ["Source", "SS", "df", "F", "p"]
    total_ss = float(base_anova["SS"].sum())
    model_rows = base_anova[base_anova["Source"] != "Residual"].copy()
    resid_row = base_anova[base_anova["Source"] == "Residual"].iloc[0]
    rows = []
    for _, r in model_rows.iterrows():
        pt = _pretty_term(r["Source"], inv_map)
        ss = float(r["SS"])
        df_ = int(r["df"])
        ms = ss / df_ if df_ > 0 else np.nan
        rows.append({"Source": pt, "df": df_, "Sum of Squares": ss, "Mean Square": ms,
                     "F-Statistic": float(r["F"]) if pd.notna(r["F"]) else np.nan,
                     "P-Value": float(r["p"]) if pd.notna(r["p"]) else np.nan})
    resid_ss = float(resid_row["SS"])
    resid_df = int(resid_row["df"])
    if lof_result is not None:
        rows.append({"Source": "Lack of Fit", "df": lof_result["df_lof"],
                     "Sum of Squares": lof_result["ss_lof"], "Mean Square": lof_result["ms_lof"],
                     "F-Statistic": lof_result["f_lof"], "P-Value": lof_result["p_lof"]})
        rows.append({"Source": "Pure Error", "df": lof_result["df_pe"],
                     "Sum of Squares": lof_result["ss_pe"], "Mean Square": lof_result["ms_pe"],
                     "F-Statistic": np.nan, "P-Value": np.nan})
    else:
        ms_e = resid_ss / resid_df if resid_df > 0 else np.nan
        rows.append({"Source": "Error", "df": resid_df, "Sum of Squares": resid_ss,
                     "Mean Square": ms_e, "F-Statistic": np.nan, "P-Value": np.nan})
    anova_df = pd.DataFrame(rows)
    anova_df["SS (%)"] = anova_df["Sum of Squares"] / total_ss * 100
    anova_df = anova_df[["Source", "df", "Sum of Squares", "Mean Square", "F-Statistic", "P-Value", "SS (%)"]]
    return anova_df


def _make_interaction_plot(safe_df, factor_a, factor_b, all_safe_factors, model, inv_map, response_name):
    cfg = common.safe_get_plot_cfg("DoE contour")
    a_vals = np.linspace(float(safe_df[factor_a].min()), float(safe_df[factor_a].max()), 40)
    b_low = float(safe_df[factor_b].min())
    b_high = float(safe_df[factor_b].max())
    other_factors = [f for f in all_safe_factors if f not in [factor_a, factor_b]]
    fig, ax = plt.subplots(figsize=(max(4, cfg["fig_w"] * 0.65), max(3, cfg["fig_h"] * 0.75)))
    for b_val, b_label, ls, color in [
        (b_low, f"{inv_map.get(factor_b, factor_b)} Low", "-", cfg["primary_color"]),
        (b_high, f"{inv_map.get(factor_b, factor_b)} High", "--", cfg["secondary_color"]),
    ]:
        grid = pd.DataFrame({factor_a: a_vals, factor_b: np.full(len(a_vals), b_val)})
        for f in other_factors:
            grid[f] = float(safe_df[f].mean())
        pred = model.predict(grid)
        ax.plot(a_vals, pred, linestyle=ls, color=color, linewidth=cfg["line_width"], label=b_label)
    fa_orig = inv_map.get(factor_a, factor_a)
    fb_orig = inv_map.get(factor_b, factor_b)
    common.apply_ax_style(ax, f"Interaction: {fa_orig} x {fb_orig}",
                          fa_orig, response_name, legend=True, plot_key="DoE contour")
    return fig


def _residual_4panel(fitted_vals, residuals, run_order=None):
    cfg = common.safe_get_plot_cfg("DoE residual plot")
    resid = np.asarray(residuals, dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=(cfg["fig_w"], cfg["fig_h"] + 1.5))
    ax = axes[0, 0]
    stats.probplot(resid, dist="norm", plot=ax)
    if len(ax.lines) >= 2:
        ax.lines[0].set_marker(cfg.get("marker_style", "o"))
        ax.lines[0].set_linestyle("None")
        ax.lines[0].set_color(cfg["primary_color"])
        ax.lines[0].set_markersize(max(3, cfg["marker_size"] / 10))
        ax.lines[1].set_color(cfg["secondary_color"])
        ax.lines[1].set_linestyle(cfg["aux_line_style"])
        ax.lines[1].set_linewidth(cfg["aux_line_width"])
    ax.set_title("Normal Probability Plot", fontsize=cfg["title_size"])
    ax.set_xlabel("Theoretical Quantiles", fontsize=cfg["label_size"])
    ax.set_ylabel("Ordered Residuals", fontsize=cfg["label_size"])
    ax = axes[0, 1]
    ax.boxplot(resid, vert=True, patch_artist=True,
               boxprops=dict(facecolor=cfg["band_color"], color=cfg["primary_color"]),
               medianprops=dict(color=cfg["secondary_color"], linewidth=2),
               whiskerprops=dict(color=cfg["primary_color"]),
               capprops=dict(color=cfg["primary_color"]),
               flierprops=dict(marker="o", color=cfg["primary_color"], markersize=4))
    ax.axhline(0, color="#111827", ls=cfg["aux_line_style"], lw=0.8)
    ax.set_title("Box Plot of Residuals", fontsize=cfg["title_size"])
    ax.set_ylabel("Residuals", fontsize=cfg["label_size"])
    ax.set_xticks([])
    ax = axes[1, 0]
    n_bins = max(5, min(15, len(resid) // 2))
    ax.hist(resid, bins=n_bins, color=cfg["primary_color"], alpha=0.72,
            edgecolor=cfg.get("border_color", "#111827"), linewidth=0.5)
    ax.axvline(0, color=cfg["secondary_color"], ls=cfg["aux_line_style"], lw=cfg["aux_line_width"])
    ax.set_title("Histogram of Residuals", fontsize=cfg["title_size"])
    ax.set_xlabel("Residuals", fontsize=cfg["label_size"])
    ax.set_ylabel("Frequency", fontsize=cfg["label_size"])
    ax = axes[1, 1]
    ro = run_order if run_order is not None else np.arange(1, len(resid) + 1)
    ax.scatter(ro, resid, color=cfg["primary_color"], s=cfg["marker_size"],
               marker=cfg.get("marker_style", "o"))
    ax.plot(ro, resid, color=cfg["primary_color"], alpha=0.28, linewidth=0.8)
    ax.axhline(0, color="#111827", ls=cfg["aux_line_style"], lw=cfg["aux_line_width"])
    ax.set_title("Residuals vs Run Order", fontsize=cfg["title_size"])
    ax.set_xlabel("Run Order", fontsize=cfg["label_size"])
    ax.set_ylabel("Residuals", fontsize=cfg["label_size"])
    for a in axes.flat:
        a.tick_params(labelsize=max(7, cfg["tick_label_size"]))
        for spine in a.spines.values():
            spine.set_linewidth(cfg["border_width"])
            spine.set_color(cfg.get("border_color", "#111827"))
    fig.suptitle("Residual Diagnostics", fontsize=cfg.get("title_size", 12) + 1, fontweight="bold")
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# NIST-STYLE PDF REPORT
# ─────────────────────────────────────────────────────────────────────────────
def _doe_make_pdf_report(exp_description, data_df, factor_info, response_cols,
                          all_response_analyses, decimals=3):
    from reportlab.lib import colors as rl_colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.lib.enums import TA_LEFT
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                     Table, TableStyle, Image, PageBreak, HRFlowable)
    from io import BytesIO

    bio = BytesIO()
    doc = SimpleDocTemplate(bio, pagesize=landscape(A4),
                             leftMargin=1.5*cm, rightMargin=1.5*cm,
                             topMargin=1.5*cm, bottomMargin=1.2*cm)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle("Body9", parent=styles["BodyText"], fontSize=9, leading=13, alignment=TA_LEFT))
    styles.add(ParagraphStyle("Bold9", parent=styles["BodyText"], fontSize=9, leading=13, fontName="Helvetica-Bold"))
    styles.add(ParagraphStyle("Section", parent=styles["Heading2"], fontSize=12, leading=15, spaceAfter=4))
    styles.add(ParagraphStyle("Sub", parent=styles["Heading3"], fontSize=10, leading=13, spaceAfter=3))

    def make_tbl(df, dec=decimals, max_rows=60):
        if len(df) > max_rows:
            df = df.head(max_rows)
        fmt_df = df.copy()
        for c in fmt_df.columns:
            if pd.api.types.is_numeric_dtype(fmt_df[c]):
                fmt_df[c] = fmt_df[c].map(lambda x: "-" if pd.isna(x) else f"{x:.{dec}f}")
            else:
                fmt_df[c] = fmt_df[c].fillna("-").astype(str)
        data = [list(fmt_df.columns)] + fmt_df.values.tolist()
        ncols = max(1, len(fmt_df.columns))
        col_w = 25.5*cm / ncols
        tbl = Table(data, repeatRows=1, colWidths=[col_w]*ncols)
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), rl_colors.HexColor("#F8FAFC")),
            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE", (0,0), (-1,-1), 8),
            ("LEADING", (0,0), (-1,-1), 10),
            ("ALIGN", (0,0), (-1,-1), "CENTER"),
            ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
            ("TOPPADDING", (0,0), (-1,-1), 4),
            ("BOTTOMPADDING", (0,0), (-1,-1), 4),
            ("LINEABOVE", (0,0), (-1,0), 1.2, rl_colors.black),
            ("LINEBELOW", (0,0), (-1,0), 0.8, rl_colors.black),
            ("LINEBELOW", (0,-1), (-1,-1), 1.2, rl_colors.black),
        ]))
        return tbl

    story = []
    story.append(Paragraph("Design of Experiments – Analysis Report", styles["Title"]))
    story.append(Spacer(1, 0.3*cm))
    story.append(HRFlowable(width="100%", thickness=1, color=rl_colors.black))
    story.append(Spacer(1, 0.2*cm))

    story.append(Paragraph("Experiment Description", styles["Section"]))
    for line in exp_description:
        story.append(Paragraph(line, styles["Body9"]))
        story.append(Spacer(1, 0.08*cm))
    if factor_info:
        story.append(Spacer(1, 0.15*cm))
        story.append(Paragraph("Factor Summary", styles["Sub"]))
        story.append(make_tbl(pd.DataFrame(factor_info), dec=4))
        story.append(Spacer(1, 0.25*cm))

    story.append(Paragraph("Experimental Data", styles["Section"]))
    story.append(Paragraph("The table shows experimental runs and observed responses.", styles["Body9"]))
    story.append(Spacer(1, 0.12*cm))
    if data_df is not None and len(data_df) > 0:
        story.append(make_tbl(data_df, dec=4))
    story.append(Spacer(1, 0.3*cm))

    for resp_name, analysis in all_response_analyses.items():
        story.append(PageBreak())
        story.append(Paragraph(f"Analysis of DOE Data – Response: {resp_name}", styles["Section"]))
        story.append(Spacer(1, 0.1*cm))

        story.append(Paragraph("Analysis Steps", styles["Sub"]))
        story.append(Paragraph(
            "1. Fit the full model. "
            "2. Backward stepwise regression using AIC criterion to identify important terms. "
            "3. Apply hierarchy principle (retain main effects part of significant higher-order terms). "
            "4. Generate residual diagnostics (normal plot, box plot, histogram, run-order plot). "
            "5. Examine interaction plots, ANOVA statistics (R², Adj R², LOF test). "
            "6. Contour and perspective plots to visualise the response surface.",
            styles["Body9"]))
        story.append(Spacer(1, 0.18*cm))

        if "full_coef" in analysis:
            story.append(Paragraph(f"Full {analysis.get('model_type','').capitalize()} Model – Coefficients", styles["Sub"]))
            story.append(make_tbl(analysis["full_coef"]))
            if "full_stats" in analysis:
                fs = analysis["full_stats"]
                story.append(Paragraph(
                    f"R\u00b2 = {fs['r2']:.4f},  Adj R\u00b2 = {fs['adj_r2']:.4f},  "
                    f"RMSE = {fs['rmse']:.4f},  F = {fs['f_stat']:.3f}  (p = {fs['f_p']:.4f})",
                    styles["Bold9"]))
            story.append(Spacer(1, 0.18*cm))

        if "stepwise_table" in analysis and len(analysis["stepwise_table"]) > 0:
            story.append(Paragraph("Backward Stepwise Regression (AIC)", styles["Sub"]))
            story.append(make_tbl(analysis["stepwise_table"].drop(columns=["Terms"], errors="ignore")))
            if "stepwise_note" in analysis:
                story.append(Spacer(1, 0.08*cm))
                story.append(Paragraph(analysis["stepwise_note"], styles["Body9"]))
            story.append(Spacer(1, 0.18*cm))

        if "anova" in analysis:
            story.append(Paragraph("ANOVA for Selected Model", styles["Sub"]))
            story.append(make_tbl(analysis["anova"]))
            story.append(Spacer(1, 0.1*cm))
        if "coef" in analysis:
            story.append(Paragraph("Selected Model – Coefficients", styles["Sub"]))
            story.append(make_tbl(analysis["coef"]))
            story.append(Spacer(1, 0.1*cm))
        if "fit_stats" in analysis:
            fs = analysis["fit_stats"]
            r2l = (f"R\u00b2 = {fs['r2']:.4f},  Adj R\u00b2 = {fs['adj_r2']:.4f},  "
                   f"RMSE = {fs['rmse']:.4f},  N = {fs['n']}")
            if fs.get("lof_p") is not None:
                r2l += f",  Lack-of-Fit p = {fs['lof_p']:.4f}"
            story.append(Paragraph(r2l, styles["Bold9"]))
            story.append(Spacer(1, 0.1*cm))

        if "model_eq" in analysis:
            story.append(Paragraph("Model Equation", styles["Sub"]))
            story.append(Paragraph(analysis["model_eq"], styles["Body9"]))
            story.append(Spacer(1, 0.12*cm))

        if "conclusions" in analysis and analysis["conclusions"]:
            story.append(Paragraph("Conclusions", styles["Sub"]))
            for c_line in analysis["conclusions"]:
                story.append(Paragraph(f"• {c_line}", styles["Body9"]))
            story.append(Spacer(1, 0.1*cm))

        if "figures" in analysis and analysis["figures"]:
            story.append(PageBreak())
            story.append(Paragraph(f"Figures – {resp_name}", styles["Section"]))
            for fig_cap, fig_bytes in analysis["figures"].items():
                story.append(Paragraph(fig_cap, styles["Sub"]))
                img = Image(BytesIO(fig_bytes))
                img._restrictSize(24.5*cm, 13*cm)
                story.append(img)
                story.append(Spacer(1, 0.35*cm))

    doc.build(story)
    bio.seek(0)
    return bio.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# RENDER
# ─────────────────────────────────────────────────────────────────────────────
def render():
    render_display_settings()
    st.sidebar.title("🧪 DoE Studio")
    st.sidebar.markdown("Design of Experiments")
    app_header("🧪 DoE Studio", "Design builder and NIST-style response analysis in one place.")
    tabs = st.tabs(["Design Builder", "Analyze Responses"])

    # ═══════════════ TAB 1: DESIGN BUILDER ═══════════════
    with tabs[0]:
        st.subheader("Design Builder")
        design_type = st.selectbox(
            "Design type",
            ["2-Level Full Factorial", "Fractional Factorial (2^k-p)",
             "Central Composite Design (CCD)", "Box-Behnken Design (BBD)"],
            key="doe_design_type",
        )

        if design_type == "2-Level Full Factorial":
            info_box("2-level full factorial: all factor-level combinations. Supports blocks, center points, and replicates.")
            st.button("Sample Data", key="sample_doe_design", on_click=_load_sample_design)
            c1, c2 = st.columns(2)
            with c1:
                n_factors = st.number_input("Number of factors", min_value=2, max_value=8, value=3, step=1, key="doe_n_factors")
                blocks = st.number_input("Blocks", min_value=1, max_value=10, value=1, step=1, key="doe_blocks")
                replicates = st.number_input("Replicates", min_value=1, max_value=10, value=1, step=1, key="doe_replicates")
            with c2:
                center_points = st.number_input("Center points per block", min_value=0, max_value=20, value=0, step=1, key="doe_center_points")
                randomize = st.checkbox("Randomize within block", value=True, key="doe_randomize")
                seed = st.number_input("Random seed", min_value=1, max_value=999999, value=123, step=1, key="doe_seed")
            st.markdown("### Factor definitions")
            factor_names, lows, highs = [], [], []
            for i in range(int(n_factors)):
                cols = st.columns([1.3, 1, 1])
                factor_names.append(cols[0].text_input(f"Factor {i+1} name", value=f"Factor {i+1}", key=f"doe_name_{i}"))
                lows.append(cols[1].number_input(f"Low level {i+1}", value=0.0, key=f"doe_low_{i}"))
                highs.append(cols[2].number_input(f"High level {i+1}", value=1.0, key=f"doe_high_{i}"))
            if st.button("Generate design", type="primary", key="gen_design"):
                d2 = _build_factorial_design(factor_names, lows, highs, blocks=int(blocks),
                                              center_points=int(center_points), replicates=int(replicates),
                                              randomize=randomize, seed=int(seed))
                st.session_state["doe_generated_design"] = d2
                st.session_state["doe_design_desc"] = (
                    f"2-Level Full Factorial, {len(factor_names)} factors, {int(blocks)} block(s), "
                    f"{int(replicates)} replicate(s), {int(center_points)} center point(s)/block.")

        elif design_type == "Fractional Factorial (2^k-p)":
            info_box("Fractional factorial designs reduce runs by confounding higher-order interactions.")
            design_key = st.selectbox("Select fractional design", list(FRACTIONAL_DESIGNS.keys()), key="doe_ff_key")
            spec = FRACTIONAL_DESIGNS[design_key]
            st.info(spec["description"])
            k = spec["k"]
            c1, c2 = st.columns(2)
            with c1:
                cp = st.number_input("Center points", min_value=0, max_value=20, value=0, step=1, key="doe_ff_cp")
            with c2:
                seed2 = st.number_input("Random seed", min_value=1, max_value=999999, value=123, step=1, key="doe_ff_seed")
            rand2 = st.checkbox("Randomize run order", value=True, key="doe_ff_rand")
            st.markdown("### Factor definitions")
            fn2, lo2, hi2 = [], [], []
            for i in range(k):
                cols = st.columns([1.3, 1, 1])
                fn2.append(cols[0].text_input(f"Factor {i+1} name", value=f"Factor {i+1}", key=f"doe_ff_name_{i}"))
                lo2.append(cols[1].number_input(f"Low level {i+1}", value=0.0, key=f"doe_ff_low_{i}"))
                hi2.append(cols[2].number_input(f"High level {i+1}", value=1.0, key=f"doe_ff_high_{i}"))
            if st.button("Generate design", type="primary", key="gen_ff_design"):
                try:
                    d2 = _build_fractional_factorial(fn2, lo2, hi2, design_key, center_points=int(cp), randomize=rand2, seed=int(seed2))
                    st.session_state["doe_generated_design"] = d2
                    st.session_state["doe_design_desc"] = f"Fractional Factorial {design_key} | Resolution {spec['resolution']}"
                except Exception as e:
                    st.error(str(e))

        elif design_type == "Central Composite Design (CCD)":
            info_box("CCD adds axial (star) points to a factorial design, enabling second-order (quadratic) model fitting.")
            c1, c2, c3 = st.columns(3)
            with c1:
                n_f_ccd = st.number_input("Number of factors", min_value=2, max_value=6, value=2, step=1, key="doe_ccd_k")
            with c2:
                ccd_sel = st.selectbox("CCD type", ["CCC (Circumscribed)", "CCF (Face-Centered)", "CCI (Inscribed)"], key="doe_ccd_type")
                ccd_code = ccd_sel.split(" ")[0]
            with c3:
                cp_ccd = st.number_input("Center points", min_value=1, max_value=10, value=5, step=1, key="doe_ccd_cp")
            c4, c5 = st.columns(2)
            with c4:
                rand_ccd = st.checkbox("Randomize run order", value=True, key="doe_ccd_rand")
            with c5:
                seed_ccd = st.number_input("Random seed", min_value=1, max_value=999999, value=123, step=1, key="doe_ccd_seed")
            k_ccd = int(n_f_ccd)
            alpha_v = round((2**k_ccd)**0.25, 4)
            n_total_ccd = 2**k_ccd + 2*k_ccd + int(cp_ccd)
            if ccd_code == "CCC":
                st.info(f"CCC: factorial ±1, axial ±{alpha_v} (rotatable). Total runs: {n_total_ccd}")
            elif ccd_code == "CCF":
                st.info(f"CCF: all within ±1 cube, alpha=1 (face-centered). Total runs: {n_total_ccd}")
            else:
                fscale = round(1.0/alpha_v, 4)
                st.info(f"CCI: factorial at ±{fscale}, axial at ±1 (inscribed). Total runs: {n_total_ccd}")
            st.markdown("### Factor definitions")
            fn_ccd, lo_ccd, hi_ccd = [], [], []
            for i in range(k_ccd):
                cols = st.columns([1.3, 1, 1])
                fn_ccd.append(cols[0].text_input(f"Factor {i+1} name", value=f"Factor {i+1}", key=f"doe_ccd_name_{i}"))
                lo_ccd.append(cols[1].number_input(f"Low level {i+1}", value=0.0, key=f"doe_ccd_low_{i}"))
                hi_ccd.append(cols[2].number_input(f"High level {i+1}", value=1.0, key=f"doe_ccd_high_{i}"))
            if st.button("Generate design", type="primary", key="gen_ccd_design"):
                try:
                    d2 = _build_ccd_design(fn_ccd, lo_ccd, hi_ccd, ccd_type=ccd_code,
                                            center_points=int(cp_ccd), randomize=rand_ccd, seed=int(seed_ccd))
                    st.session_state["doe_generated_design"] = d2
                    st.session_state["doe_design_desc"] = f"CCD ({ccd_sel}), {k_ccd} factors, alpha={alpha_v}, {int(cp_ccd)} center points"
                except Exception as e:
                    st.error(str(e))

        else:  # Box-Behnken
            info_box("Box-Behnken: 3-level design using midpoints of edges. No corner runs — all within a sphere. Requires k >= 3.")
            c1, c2, c3 = st.columns(3)
            with c1:
                n_f_bbd = st.number_input("Number of factors (>=3)", min_value=3, max_value=7, value=3, step=1, key="doe_bbd_k")
            with c2:
                cp_bbd = st.number_input("Center points", min_value=1, max_value=10, value=3, step=1, key="doe_bbd_cp")
            with c3:
                seed_bbd = st.number_input("Random seed", min_value=1, max_value=999999, value=123, step=1, key="doe_bbd_seed")
            rand_bbd = st.checkbox("Randomize run order", value=True, key="doe_bbd_rand")
            k_bbd = int(n_f_bbd)
            n_edge_bbd = sum(1 for _ in combinations(range(k_bbd), 2)) * 4
            st.info(f"BBD: {n_edge_bbd} edge runs + {int(cp_bbd)} center points = {n_edge_bbd + int(cp_bbd)} total runs")
            st.markdown("### Factor definitions")
            fn_bbd, lo_bbd, hi_bbd = [], [], []
            for i in range(k_bbd):
                cols = st.columns([1.3, 1, 1])
                fn_bbd.append(cols[0].text_input(f"Factor {i+1} name", value=f"Factor {i+1}", key=f"doe_bbd_name_{i}"))
                lo_bbd.append(cols[1].number_input(f"Low level {i+1}", value=0.0, key=f"doe_bbd_low_{i}"))
                hi_bbd.append(cols[2].number_input(f"High level {i+1}", value=1.0, key=f"doe_bbd_high_{i}"))
            if st.button("Generate design", type="primary", key="gen_bbd_design"):
                try:
                    d2 = _build_boxbehnken_design(fn_bbd, lo_bbd, hi_bbd,
                                                   center_points=int(cp_bbd), randomize=rand_bbd, seed=int(seed_bbd))
                    st.session_state["doe_generated_design"] = d2
                    st.session_state["doe_design_desc"] = f"Box-Behnken Design, {k_bbd} factors, {int(cp_bbd)} center points"
                except Exception as e:
                    st.error(str(e))

        if "doe_generated_design" in st.session_state:
            design = st.session_state["doe_generated_design"]
            st.success(f"Generated design with {len(design)} runs")
            st.dataframe(design, use_container_width=True)
            excel_bytes = make_excel_bytes({"Design": design})
            st.download_button("Download design workbook", excel_bytes, file_name="doe_design.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    # ═══════════════ TAB 2: ANALYZE RESPONSES ═══════════════
    with tabs[1]:
        st.subheader("Analyze Responses")
        info_box("Paste your completed DoE data, select factors and responses, and run a full NIST-style analysis including stepwise regression, LOF test, interaction plots, and residual diagnostics.")

        c_btn, c_text = st.columns([1, 4])
        with c_btn:
            st.button("Sample: 4-factor factorial", key="sample_doe_resp", on_click=_load_sample_response_text)
            st.button("Sample: NIST CCI (2 factors, 2 responses)", key="sample_nist", on_click=_load_nist_sample)
        with c_text:
            data_input = st.text_area("Paste completed DoE data with headers", height=240, key="doe_response_input")

        decimals = st.slider("Decimals", 1, 8, DEFAULT_DECIMALS, key="doe_dec")

        if not data_input:
            st.info("Paste your data above or load a sample to begin.")
        else:
            try:
                df = parse_pasted_table(data_input, header=True)
                if df is None or df.empty:
                    st.warning("No data found. Check the format.")
                else:
                    num_cols = get_numeric_columns(df)
                    all_cols = list(df.columns)
                    st.markdown("---")
                    c1, c2, c3, c4 = st.columns([1.5, 1.2, 1, 1])
                    with c1:
                        factors = st.multiselect("Numeric factors", num_cols,
                                                  default=num_cols[:min(2, len(num_cols))], key="doe_factors")
                    with c2:
                        avail_resp = [c for c in num_cols if c not in factors] or num_cols
                        responses = st.multiselect("Response(s)", avail_resp,
                                                    default=avail_resp[:min(1, len(avail_resp))],
                                                    key="doe_responses")
                    with c3:
                        model_type = st.selectbox("Model type", ["linear", "interaction", "quadratic"], key="doe_model_type")
                    with c4:
                        block_col = st.selectbox("Block column (optional)",
                                                  ["(None)"] + [c for c in all_cols if c not in factors + responses],
                                                  key="doe_block_col")

                    if len(factors) < 2 or not responses:
                        st.info("Select at least 2 factors and 1 response to begin analysis.")
                    else:
                        use_cols = factors + responses + ([block_col] if block_col != "(None)" else [])
                        d = df[[c for c in use_cols if c in df.columns]].copy()
                        for c in factors + responses:
                            d[c] = to_numeric(d[c])
                        d = d.dropna(subset=factors + responses).reset_index(drop=True)

                        safe_factor_names = [f"F{i+1}" for i in range(len(factors))]
                        rename_map = {orig: safe for orig, safe in zip(factors, safe_factor_names)}
                        inv_map = {v: k for k, v in rename_map.items()}

                        # ── Experiment Description ──
                        st.markdown("### Experiment Description")
                        ec = st.columns(4)
                        ec[0].metric("Total Runs", len(d))
                        ec[1].metric("Factors", len(factors))
                        ec[2].metric("Responses", len(responses))
                        ec[3].metric("Model", model_type.capitalize())

                        factor_info_rows = []
                        for f in factors:
                            factor_info_rows.append({
                                "Factor": f, "Low": float(d[f].min()), "High": float(d[f].max()),
                                "Mean": round(float(d[f].mean()), decimals),
                                "Range": round(float(d[f].max() - d[f].min()), decimals)
                            })
                        report_table(pd.DataFrame(factor_info_rows), "Factor Summary", decimals)

                        with st.expander("Experimental Data Table", expanded=False):
                            st.dataframe(d, use_container_width=True)

                        all_response_analyses = {}

                        for resp_idx, response in enumerate(responses):
                            st.markdown("---")
                            st.markdown(f"## Response: {response}")

                            safe_df = d[factors + [response]].copy()
                            safe_df = safe_df.rename(columns=rename_map)
                            safe_df = safe_df.rename(columns={response: "Response"})
                            if block_col != "(None)" and block_col in d.columns:
                                safe_df["Block"] = d[block_col].astype(str).values

                            formula_str, rhs_terms = _doe_formula(safe_factor_names, model_type=model_type)
                            if block_col != "(None)" and "Block" in safe_df.columns:
                                formula_str += " + C(Block)"
                                rhs_terms = rhs_terms + ["C(Block)"]

                            try:
                                full_model = smf.ols(formula_str, data=safe_df).fit()
                            except Exception as e:
                                st.error(f"Model fit failed for {response}: {e}")
                                continue

                            # ── Step 1: Full model ──
                            st.markdown("### Step 1 — Full Model Fit")
                            full_coef = pd.DataFrame({
                                "Term": [_pretty_term(t, inv_map) for t in full_model.params.index],
                                "Estimate": full_model.params.values,
                                "Std. Error": full_model.bse.values,
                                "t-Value": full_model.tvalues.values,
                                "P-Value": full_model.pvalues.values,
                            })
                            report_table(full_coef, f"Full {model_type.capitalize()} Model — Coefficients", decimals)
                            full_stats = {
                                "r2": float(full_model.rsquared),
                                "adj_r2": float(full_model.rsquared_adj),
                                "rmse": float(np.sqrt(full_model.mse_resid)),
                                "f_stat": float(full_model.fvalue) if pd.notna(full_model.fvalue) else np.nan,
                                "f_p": float(full_model.f_pvalue) if pd.notna(full_model.f_pvalue) else np.nan,
                            }
                            ms1, ms2, ms3, ms4 = st.columns(4)
                            ms1.metric("R²", f"{full_stats['r2']:.4f}")
                            ms2.metric("Adj R²", f"{full_stats['adj_r2']:.4f}")
                            ms3.metric("RMSE", f"{full_stats['rmse']:.4f}")
                            ms4.metric("F-stat", f"{full_stats['f_stat']:.3f}" if pd.notna(full_stats['f_stat']) else "-")

                            # ── Step 2: Stepwise AIC ──
                            st.markdown("### Step 2 — Backward Stepwise Regression (AIC Criterion)")
                            with st.spinner("Running stepwise regression..."):
                                sw_steps = _backward_stepwise_aic(safe_df, rhs_terms)

                            step_rows = []
                            if sw_steps:
                                for s in sw_steps:
                                    step_rows.append({"Step": s["Step"], "Action": s["Action"],
                                                       "AIC": s["AIC"],
                                                       "Terms in Model": " + ".join([_pretty_term(t, inv_map) for t in s["Terms"]])})
                                report_table(pd.DataFrame(step_rows), "Stepwise Regression Steps", decimals)
                                selected_model = sw_steps[-1]["model"]
                                selected_terms = sw_steps[-1]["Terms"]
                                step_note = (
                                    f"Selected model has {len(selected_terms)} term(s): "
                                    f"{', '.join([_pretty_term(t, inv_map) for t in selected_terms])}. "
                                    "Main effects that are part of significant interactions are retained (hierarchy principle). "
                                    "Note: not all analysts agree with this principle — always review the model manually.")
                                st.info(step_note)
                            else:
                                selected_model = full_model
                                selected_terms = rhs_terms
                                step_note = "Stepwise regression did not improve AIC; full model retained."
                                st.info(step_note)

                            # ── Step 3: ANOVA + LOF ──
                            st.markdown("### Step 3 — ANOVA for Selected Model")
                            lof_result = _lack_of_fit_test(safe_df, safe_factor_names, selected_model)
                            anova_df = _build_anova_with_lof(selected_model, safe_df, safe_factor_names, inv_map, lof_result)
                            report_table(anova_df, f"ANOVA — {response}", decimals)

                            sel_coef = pd.DataFrame({
                                "Term": [_pretty_term(t, inv_map) for t in selected_model.params.index],
                                "Estimate": selected_model.params.values,
                                "Std. Error": selected_model.bse.values,
                                "t-Value": selected_model.tvalues.values,
                                "P-Value": selected_model.pvalues.values,
                            })
                            report_table(sel_coef, f"Selected Model Coefficients — {response}", decimals)

                            fit_stats = {
                                "r2": float(selected_model.rsquared),
                                "adj_r2": float(selected_model.rsquared_adj),
                                "rmse": float(np.sqrt(selected_model.mse_resid)),
                                "n": int(selected_model.nobs),
                                "lof_p": lof_result["p_lof"] if lof_result else None,
                            }
                            fs1, fs2, fs3, fs4 = st.columns(4)
                            fs1.metric("R²", f"{fit_stats['r2']:.4f}")
                            fs2.metric("Adj R²", f"{fit_stats['adj_r2']:.4f}")
                            fs3.metric("RMSE", f"{fit_stats['rmse']:.4f}")
                            if lof_result:
                                lof_lbl = "OK" if lof_result["p_lof"] > 0.05 else "Significant"
                                fs4.metric("LOF p-value", f"{lof_result['p_lof']:.4f} ({lof_lbl})")
                            else:
                                fs4.metric("LOF test", "N/A (no replicates)")

                            # Model equation
                            st.markdown("### Model Equation")
                            eq_parts = []
                            for term_name, val in selected_model.params.items():
                                pt = _pretty_term(term_name, inv_map)
                                sign = "+" if val >= 0 else "-"
                                if pt == "Intercept":
                                    eq_parts.append(f"{val:.{decimals}f}")
                                else:
                                    eq_parts.append(f"{sign} {abs(val):.{decimals}f}·{pt}")
                            model_eq_str = f"**{response}** = " + " ".join(eq_parts)
                            st.markdown(model_eq_str)

                            # ── Step 4: Interaction plots ──
                            int_figs = {}
                            if model_type in ["interaction", "quadratic"] and len(safe_factor_names) >= 2:
                                st.markdown("### Step 4 — Interaction Plots")
                                factor_pairs = list(combinations(safe_factor_names, 2))
                                cols_per_row = min(3, len(factor_pairs))
                                pair_idx = 0
                                rows_needed = (len(factor_pairs) + cols_per_row - 1) // cols_per_row
                                for row_i in range(rows_needed):
                                    int_cols = st.columns(cols_per_row)
                                    for col_i in range(cols_per_row):
                                        if pair_idx >= len(factor_pairs):
                                            break
                                        fa_s, fb_s = factor_pairs[pair_idx]
                                        fig_int = _make_interaction_plot(
                                            safe_df, fa_s, fb_s, safe_factor_names,
                                            selected_model, inv_map, response)
                                        int_cols[col_i].pyplot(fig_int, use_container_width=True)
                                        cap_int = f"Interaction: {inv_map.get(fa_s,fa_s)} x {inv_map.get(fb_s,fb_s)}"
                                        int_figs[cap_int] = fig_to_png_bytes(fig_int)
                                        plt.close(fig_int)
                                        pair_idx += 1

                            # ── Step 5: Residual diagnostics (4-panel) ──
                            st.markdown("### Step 5 — Residual Diagnostics")
                            fig_4p = _residual_4panel(
                                selected_model.fittedvalues.values,
                                selected_model.resid.values,
                                run_order=np.arange(1, len(safe_df) + 1))
                            st.pyplot(fig_4p, use_container_width=True)

                            # ── Step 6: Response surface plots ──
                            st.markdown("### Step 6 — Response Surface Plots")
                            p1col, p2col = st.columns(2)
                            with p1col:
                                xfac = st.selectbox("X-axis factor", factors, index=0, key=f"xfac_{resp_idx}")
                            with p2col:
                                yfac_opts = [f for f in factors if f != xfac]
                                yfac = st.selectbox("Y-axis factor", yfac_opts, index=0, key=f"yfac_{resp_idx}")

                            xfac_s = rename_map.get(xfac, xfac)
                            yfac_s = rename_map.get(yfac, yfac)
                            other_f = [f for f in factors if f not in [xfac, yfac]]
                            other_fs = [rename_map.get(f, f) for f in other_f]
                            fixed_vals = {}
                            if other_f:
                                st.markdown("**Fixed levels for remaining factors**")
                                fix_cols = st.columns(len(other_f))
                                for i, (f, fs2) in enumerate(zip(other_f, other_fs)):
                                    fixed_vals[fs2] = fix_cols[i].slider(
                                        f, min_value=float(d[f].min()), max_value=float(d[f].max()),
                                        value=float(d[f].mean()), key=f"fix_{resp_idx}_{i}")

                            x_vals = np.linspace(float(d[xfac].min()), float(d[xfac].max()), 40)
                            y_vals2 = np.linspace(float(d[yfac].min()), float(d[yfac].max()), 40)
                            xx, yy = np.meshgrid(x_vals, y_vals2)
                            grid = pd.DataFrame({xfac_s: xx.ravel(), yfac_s: yy.ravel()})
                            for fs2, fv in fixed_vals.items():
                                grid[fs2] = fv
                            if block_col != "(None)" and "Block" in safe_df.columns:
                                grid["Block"] = safe_df["Block"].mode().iloc[0]
                            zz = selected_model.predict(grid).to_numpy().reshape(xx.shape)

                            contour_cfg = common.safe_get_plot_cfg("DoE contour")
                            fig_contour, ax_c = plt.subplots(figsize=(contour_cfg["fig_w"], contour_cfg["fig_h"]))
                            cs = ax_c.contourf(xx, yy, zz, levels=20, cmap="viridis")
                            fig_contour.colorbar(cs, ax=ax_c, label=response)
                            ax_c.scatter(d[xfac], d[yfac], c="white", edgecolor="black", s=contour_cfg["marker_size"])
                            apply_ax_style(ax_c, f"Contour Plot: {response}", xfac, yfac, plot_key="DoE contour")
                            st.pyplot(fig_contour, use_container_width=True)

                            surface_cfg = common.safe_get_plot_cfg("DoE surface")
                            fig_surface = plt.figure(figsize=(surface_cfg["fig_w"], surface_cfg["fig_h"] + 0.3))
                            ax3d = fig_surface.add_subplot(111, projection="3d")
                            surf = ax3d.plot_surface(xx, yy, zz, cmap="viridis", edgecolor="none", alpha=0.88)
                            ax3d.scatter(d[xfac], d[yfac], d[response],
                                         c=surface_cfg.get("marker_color", "black"), s=surface_cfg["marker_size"])
                            ax3d.set_xlabel(xfac); ax3d.set_ylabel(yfac); ax3d.set_zlabel(response)
                            ax3d.set_title(f"Response Surface: {response}")
                            fig_surface.colorbar(surf, ax=ax3d, shrink=0.6, aspect=12)
                            st.pyplot(fig_surface, use_container_width=True)

                            fig_resid, ax_res = plt.subplots(figsize=(contour_cfg["fig_w"], contour_cfg["fig_h"]))
                            ax_res.scatter(selected_model.fittedvalues, selected_model.resid,
                                           color=contour_cfg["primary_color"], s=contour_cfg["marker_size"])
                            ax_res.axhline(0, color="#111827", ls="--", lw=0.9)
                            apply_ax_style(ax_res, "Residuals vs Fitted", "Fitted Values", "Residuals", plot_key="DoE residual plot")
                            st.pyplot(fig_resid, use_container_width=True)

                            # Conclusions
                            st.markdown("### Conclusions")
                            conclusions = []
                            r2_ok = fit_stats["r2"] >= 0.70
                            conclusions.append(
                                f"R\u00b2 = {fit_stats['r2']:.4f}, Adj R\u00b2 = {fit_stats['adj_r2']:.4f} \u2014 "
                                f"{'model fit is reasonable.' if r2_ok else 'consider additional factors or transformations.'}")
                            if lof_result:
                                lof_ok = lof_result["p_lof"] > 0.05
                                conclusions.append(
                                    f"Lack-of-Fit: F = {lof_result['f_lof']:.3f}, p = {lof_result['p_lof']:.4f} \u2014 "
                                    f"{'no significant LOF; model is adequate.' if lof_ok else 'significant LOF; consider a higher-order model.'}")
                            else:
                                conclusions.append("No replicates detected; lack-of-fit test was not performed.")
                            sig_terms = [row["Term"] for _, row in sel_coef.iterrows()
                                         if row["Term"] != "Intercept" and pd.notna(row["P-Value"]) and row["P-Value"] < 0.05]
                            if sig_terms:
                                conclusions.append(f"Significant terms (p < 0.05): {', '.join(sig_terms)}.")
                            else:
                                conclusions.append("No terms were significant at p < 0.05 — review factor ranges or data quality.")
                            if model_type in ["interaction", "quadratic"] and int_figs:
                                conclusions.append("Interaction plots show predicted response as a function of each factor at its low and high levels. Parallel lines indicate no interaction; non-parallel or crossing lines indicate interaction.")
                            for c_txt in conclusions:
                                st.markdown(f"- {c_txt}")

                            # Collect for export
                            resp_figs = {}
                            resp_figs.update(int_figs)
                            resp_figs["4-Panel Residual Diagnostics"] = fig_to_png_bytes(fig_4p)
                            resp_figs["Contour Plot"] = fig_to_png_bytes(fig_contour)
                            resp_figs["Response Surface (3D)"] = fig_to_png_bytes(fig_surface)
                            resp_figs["Residuals vs Fitted"] = fig_to_png_bytes(fig_resid)
                            plt.close("all")

                            sw_tbl_df = pd.DataFrame(step_rows) if step_rows else pd.DataFrame()
                            all_response_analyses[response] = {
                                "model_type": model_type,
                                "full_coef": full_coef,
                                "full_stats": full_stats,
                                "stepwise_table": sw_tbl_df,
                                "stepwise_note": step_note,
                                "anova": anova_df,
                                "coef": sel_coef,
                                "fit_stats": fit_stats,
                                "model_eq": model_eq_str.replace("**", "").replace("*", ""),
                                "conclusions": conclusions,
                                "figures": resp_figs,
                            }

                        # ── Export ──
                        if all_response_analyses:
                            st.markdown("---")
                            st.markdown("### Export")
                            exp_description = [
                                f"Design: {st.session_state.get('doe_design_desc', 'User-supplied data')}",
                                f"Runs: {len(d)}  |  Factors: {len(factors)}  |  Responses: {len(responses)}",
                                f"Model type: {model_type.capitalize()}",
                                f"Factors: " + "; ".join(
                                    f"{r['Factor']} [{r['Low']:.4g} – {r['High']:.4g}]"
                                    for r in factor_info_rows),
                                f"Response(s): {', '.join(responses)}",
                                f"Block column: {block_col}",
                            ]
                            finfo = [{"Factor": r["Factor"], "Low": r["Low"], "High": r["High"]}
                                     for r in factor_info_rows]
                            table_map = {"Data": d}
                            for resp, an in all_response_analyses.items():
                                if not an["full_coef"].empty:
                                    table_map[f"{resp[:20]} Full Coef"] = an["full_coef"]
                                if not an.get("stepwise_table", pd.DataFrame()).empty:
                                    table_map[f"{resp[:20]} Stepwise"] = an["stepwise_table"].drop(columns=["Terms"], errors="ignore")
                                table_map[f"{resp[:20]} ANOVA"] = an["anova"]
                                table_map[f"{resp[:20]} Coef"] = an["coef"]
                            excel_bytes = make_excel_bytes(table_map)
                            try:
                                pdf_bytes = _doe_make_pdf_report(
                                    exp_description=exp_description,
                                    data_df=d,
                                    factor_info=finfo,
                                    response_cols=responses,
                                    all_response_analyses=all_response_analyses,
                                    decimals=decimals)
                            except Exception as pdf_err:
                                pdf_bytes = None
                                st.warning(f"PDF generation issue: {pdf_err}")
                            ec1, ec2 = st.columns(2)
                            with ec1:
                                st.download_button("📥 Download Excel workbook", excel_bytes,
                                                   file_name="doe_analysis.xlsx",
                                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                            with ec2:
                                if pdf_bytes:
                                    st.download_button("📄 Download PDF Report (NIST-style)", pdf_bytes,
                                                       file_name="doe_report.pdf", mime="application/pdf")
            except Exception as outer_e:
                st.error(f"Analysis error: {outer_e}")
                import traceback
                st.code(traceback.format_exc())
