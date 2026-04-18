import re
from io import BytesIO
from itertools import product, combinations

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


# ---------- sample loaders / design builder ----------

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
        factor_names, lows, highs, blocks=2, center_points=2, replicates=1, randomize=True, seed=123
    )


def _load_sample_response_text():
    st.session_state["doe_response_input"] = DOE_SAMPLE_RESPONSE_DATA



def _safe_factor_prefix(i):
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return alphabet[i] if i < len(alphabet) else f"F{i+1}"



def _build_factorial_design(
    factor_names,
    low_levels,
    high_levels,
    blocks=1,
    center_points=0,
    replicates=1,
    randomize=True,
    seed=123,
):
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


# ---------- model helpers ----------

def _make_safe_factor_names(factors):
    safe = [f"X{i+1}" for i in range(len(factors))]
    return safe, {orig: s for orig, s in zip(factors, safe)}, {s: orig for orig, s in zip(factors, safe)}



def _code_factor_series(s):
    arr = pd.to_numeric(s, errors="coerce").astype(float)
    lo = float(np.nanmin(arr))
    hi = float(np.nanmax(arr))
    center = (lo + hi) / 2.0
    half = (hi - lo) / 2.0
    if np.isclose(half, 0.0):
        coded = np.zeros(len(arr), dtype=float)
    else:
        coded = (arr.to_numpy(dtype=float) - center) / half
    return coded, lo, hi, center



def _prepare_doe_dataframe(df, factors, response, block_col=None):
    use_cols = factors + [response] + ([block_col] if block_col and block_col != "(None)" else [])
    d = df[use_cols].copy()
    for c in factors + [response]:
        d[c] = to_numeric(d[c])
    d = d.dropna(subset=factors + [response]).reset_index(drop=True)
    d.insert(0, "RunOrder", np.arange(1, len(d) + 1))

    safe_factors, rename_map, inv_map = _make_safe_factor_names(factors)
    coded_info = []
    safe_df = pd.DataFrame(index=d.index)
    for orig, safe in rename_map.items():
        coded_vals, lo, hi, center = _code_factor_series(d[orig])
        safe_df[safe] = coded_vals
        safe_df[f"{safe}_actual"] = d[orig].to_numpy(dtype=float)
        coded_info.append({
            "Factor": orig,
            "Code": safe,
            "Low": lo,
            "Center": center,
            "High": hi,
            "Range": hi - lo,
        })

    safe_df["Response"] = d[response].to_numpy(dtype=float)
    safe_df["RunOrder"] = d["RunOrder"].to_numpy(dtype=int)
    include_block = False
    if block_col and block_col != "(None)":
        blocks = d[block_col].astype(str).fillna("Missing")
        if blocks.nunique() > 1:
            safe_df["Block"] = blocks.to_numpy()
            include_block = True

    display_df = d.copy()
    for info in coded_info:
        display_df[f"{info['Factor']} (coded)"] = safe_df[info["Code"]]

    factor_summary = pd.DataFrame(coded_info)
    return d, safe_df, safe_factors, rename_map, inv_map, factor_summary, include_block, display_df



def _candidate_terms(safe_factors, model_type):
    terms = list(safe_factors)
    if model_type in ["interaction", "quadratic"]:
        terms.extend([f"{a}:{b}" for a, b in combinations(safe_factors, 2)])
    if model_type == "quadratic":
        terms.extend([f"I({f}**2)" for f in safe_factors])
    return terms



def _build_formula(terms, include_block=False):
    rhs = list(terms)
    if include_block:
        rhs.append("C(Block)")
    return "Response ~ " + (" + ".join(rhs) if rhs else "1")



def _fit_model(data, terms, include_block=False):
    formula = _build_formula(terms, include_block=include_block)
    model = smf.ols(formula, data=data).fit()
    return model, formula



def _normalize_term(term):
    return re.sub(r"\s+", "", str(term))



def _hierarchy_additions(terms):
    raw = {_normalize_term(t): t for t in terms}
    needed = set(raw.keys())
    for t in list(raw.keys()):
        if ":" in t:
            a, b = t.split(":", 1)
            needed.add(a)
            needed.add(b)
        m = re.match(r"I\(([^*]+)\*\*2\)", t)
        if m:
            needed.add(m.group(1))
    return needed



def _enforce_hierarchy(terms, candidate_terms):
    selected_norm = _hierarchy_additions(terms)
    candidate_map = {_normalize_term(t): t for t in candidate_terms}
    out = []
    for t in candidate_terms:
        if _normalize_term(t) in selected_norm:
            out.append(t)
    return out



def _backward_aic_stepwise(data, candidate_terms, include_block=False):
    current = list(candidate_terms)
    history = []
    model, formula = _fit_model(data, current, include_block=include_block)
    current_aic = float(model.aic)
    history.append({"Step": 0, "Action": "Start", "Term": "—", "AIC": current_aic, "Model": formula.replace("Response ~ ", "")})
    step = 0
    while len(current) > 1:
        trial_rows = []
        for term in current:
            trial_terms = [t for t in current if t != term]
            try:
                trial_model, trial_formula = _fit_model(data, trial_terms, include_block=include_block)
                trial_rows.append((float(trial_model.aic), term, trial_terms, trial_formula))
            except Exception:
                continue
        if not trial_rows:
            break
        best_aic, removed_term, best_terms, best_formula = min(trial_rows, key=lambda x: x[0])
        if best_aic + 1e-9 < current_aic:
            step += 1
            current = best_terms
            current_aic = best_aic
            history.append({"Step": step, "Action": "Remove", "Term": removed_term, "AIC": best_aic, "Model": best_formula.replace("Response ~ ", "")})
        else:
            break
    selected_model, selected_formula = _fit_model(data, current, include_block=include_block)
    return list(candidate_terms), pd.DataFrame(history), current, selected_model, selected_formula



def _term_to_pretty(term, inv_map):
    term = str(term)
    if term == "Intercept":
        return "Intercept"
    if term == "Residual":
        return "Error"
    if term.startswith("C(Block)"):
        return "Block"
    term = re.sub(r"I\(([^*]+?)\s*\*\*\s*2\)", lambda m: m.group(1).strip() + "²", term)
    term = term.replace(":", " × ")
    for safe, orig in inv_map.items():
        term = term.replace(safe, orig)
    term = re.sub(r"\s+²", "²", term)
    return term



def _coef_table(model, inv_map):
    return pd.DataFrame({
        "Source": [_term_to_pretty(t, inv_map) for t in model.params.index],
        "Estimate": model.params.values,
        "Std. Error": model.bse.values,
        "t value": model.tvalues.values,
        "Pr(>|t|)": model.pvalues.values,
    })



def _fit_stats_table(model, lack_of_fit=None):
    rows = [
        {"Statistic": "Residual standard error", "Value": float(np.sqrt(model.mse_resid))},
        {"Statistic": "Multiple R-squared", "Value": float(model.rsquared)},
        {"Statistic": "Adjusted R-squared", "Value": float(model.rsquared_adj)},
        {"Statistic": "F-statistic", "Value": float(model.fvalue) if pd.notna(model.fvalue) else np.nan},
        {"Statistic": "Model p-value", "Value": float(model.f_pvalue) if pd.notna(model.f_pvalue) else np.nan},
        {"Statistic": "Residual degrees of freedom", "Value": float(model.df_resid)},
    ]
    if lack_of_fit is not None and not lack_of_fit.empty:
        rows.append({"Statistic": "Lack-of-fit p-value", "Value": float(lack_of_fit.iloc[0]["Pr(>F)"]) if pd.notna(lack_of_fit.iloc[0]["Pr(>F)"]) else np.nan})
    return pd.DataFrame(rows)



def _effects_anova_table(model, inv_map):
    tbl = anova_lm(model, typ=2).reset_index().rename(columns={
        "index": "Source",
        "sum_sq": "Sum of Squares",
        "df": "DF",
        "F": "F value",
        "PR(>F)": "Pr(>F)",
    })
    tbl["Mean Square"] = tbl["Sum of Squares"] / tbl["DF"]
    tbl["Source"] = tbl["Source"].map(lambda x: _term_to_pretty(x, inv_map))
    return tbl[["Source", "DF", "Sum of Squares", "Mean Square", "F value", "Pr(>F)"]]



def _model_summary_anova(model, lack_of_fit=None):
    ss_model = float(getattr(model, "ess", np.nan))
    df_model = float(getattr(model, "df_model", np.nan))
    ms_model = ss_model / df_model if df_model and pd.notna(df_model) else np.nan
    ss_error = float(getattr(model, "ssr", np.nan))
    df_error = float(getattr(model, "df_resid", np.nan))
    ms_error = ss_error / df_error if df_error and pd.notna(df_error) else np.nan
    rows = [
        {"Source": "Model", "DF": df_model, "Sum of Squares": ss_model, "Mean Square": ms_model, "F value": float(model.fvalue) if pd.notna(model.fvalue) else np.nan, "Pr(>F)": float(model.f_pvalue) if pd.notna(model.f_pvalue) else np.nan},
        {"Source": "Total error", "DF": df_error, "Sum of Squares": ss_error, "Mean Square": ms_error, "F value": np.nan, "Pr(>F)": np.nan},
    ]
    if lack_of_fit is not None and not lack_of_fit.empty:
        rows.extend(lack_of_fit.to_dict("records"))
    return pd.DataFrame(rows)



def _lack_of_fit_table(data, model, safe_factors, include_block=False):
    group_cols = list(safe_factors) + (["Block"] if include_block else [])
    tmp = data[group_cols + ["Response"]].copy()
    for c in safe_factors:
        tmp[c] = np.round(tmp[c], 10)

    pure_ss = 0.0
    pure_df = 0
    for _, g in tmp.groupby(group_cols, dropna=False):
        if len(g) > 1:
            vals = g["Response"].to_numpy(dtype=float)
            pure_ss += float(np.sum((vals - vals.mean()) ** 2))
            pure_df += len(g) - 1

    resid_ss = float(np.sum(np.asarray(model.resid, dtype=float) ** 2))
    resid_df = int(round(float(model.df_resid)))
    lof_ss = resid_ss - pure_ss
    lof_df = resid_df - pure_df

    if pure_df > 0 and lof_df > 0 and pure_ss >= -1e-12:
        pure_ss = max(pure_ss, 0.0)
        lof_ss = max(lof_ss, 0.0)
        pure_ms = pure_ss / pure_df if pure_df else np.nan
        lof_ms = lof_ss / lof_df if lof_df else np.nan
        f_val = lof_ms / pure_ms if pure_ms and pure_ms > 0 else np.nan
        p_val = 1 - stats.f.cdf(f_val, lof_df, pure_df) if pd.notna(f_val) else np.nan
        return pd.DataFrame([
            {"Source": "Lack-of-fit", "DF": float(lof_df), "Sum of Squares": float(lof_ss), "Mean Square": float(lof_ms), "F value": float(f_val) if pd.notna(f_val) else np.nan, "Pr(>F)": float(p_val) if pd.notna(p_val) else np.nan},
            {"Source": "Pure error", "DF": float(pure_df), "Sum of Squares": float(pure_ss), "Mean Square": float(pure_ms), "F value": np.nan, "Pr(>F)": np.nan},
        ])
    return pd.DataFrame([{"Source": "Lack-of-fit", "DF": np.nan, "Sum of Squares": np.nan, "Mean Square": np.nan, "F value": np.nan, "Pr(>F)": np.nan}])



def _nested_model_comparison(reduced_model, full_model):
    try:
        cmp = anova_lm(reduced_model, full_model)
        cmp = cmp.reset_index(drop=True)
        cmp.insert(0, "Model", ["Selected model", "Full model"]) 
        return cmp.rename(columns={
            "df_resid": "Residual DF",
            "ssr": "Residual SS",
            "df_diff": "DF difference",
            "ss_diff": "SS difference",
            "F": "F value",
            "Pr(>F)": "Pr(>F)",
        })
    except Exception:
        return pd.DataFrame()



def _model_equation(model, inv_map, response_name):
    pieces = []
    for term, val in model.params.items():
        if term == "Intercept":
            pieces.append(f"{val:.3f}")
            continue
        pretty = _term_to_pretty(term, inv_map)
        sign = "+" if val >= 0 else "-"
        pieces.append(f" {sign} {abs(val):.3f}×{pretty}")
    return f"{response_name} = " + "".join(pieces)



def _diagnostic_flags(model):
    ad_stat, ad_p = normal_ad(model.resid)
    return {
        "normality_p": float(ad_p) if pd.notna(ad_p) else np.nan,
        "r2": float(model.rsquared),
        "adj_r2": float(model.rsquared_adj),
        "model_p": float(model.f_pvalue) if pd.notna(model.f_pvalue) else np.nan,
    }


def _prediction_block_value(model):
    try:
        frame = model.model.data.frame
        if "Block" in frame.columns:
            modes = frame["Block"].astype(str).mode()
            return str(modes.iloc[0]) if len(modes) else str(frame["Block"].astype(str).iloc[0])
    except Exception:
        pass
    return None



def _conclusion_lines(response, selected_terms, inv_map, model, lack_of_fit):
    flags = _diagnostic_flags(model)
    interaction_terms = [_term_to_pretty(t, inv_map) for t in selected_terms if ":" in _normalize_term(t)]
    square_terms = [_term_to_pretty(t, inv_map) for t in selected_terms if _normalize_term(t).startswith("I(")]
    lines = [
        f"The selected model for {response} achieved R² = {flags['r2']:.3f} and adjusted R² = {flags['adj_r2']:.3f}.",
        "The overall model is statistically significant." if pd.notna(flags["model_p"]) and flags["model_p"] < 0.05 else "The overall model is not strongly significant at the 0.05 level.",
    ]
    lof_p = np.nan
    if lack_of_fit is not None and not lack_of_fit.empty and lack_of_fit.iloc[0]["Source"] == "Lack-of-fit":
        lof_p = lack_of_fit.iloc[0]["Pr(>F)"]
    if pd.notna(lof_p):
        lines.append("The lack-of-fit test is not significant, which supports model adequacy." if lof_p >= 0.05 else "The lack-of-fit test is significant, which suggests the fitted model may not be fully adequate.")
    if pd.notna(flags["normality_p"]):
        lines.append("Residual diagnostics do not indicate a strong normality concern." if flags["normality_p"] >= 0.05 else "Residual normality should be reviewed because the Anderson-Darling test is significant.")
    if interaction_terms:
        lines.append(f"The selected model retains interaction term(s): {', '.join(interaction_terms)}.")
    if square_terms:
        lines.append(f"Curvature is represented by quadratic term(s): {', '.join(square_terms)}.")
    return lines



def _grid_search_optimum(model, safe_factors, factor_summary, goal="Minimize", grid_n=51):
    levels = np.linspace(-1, 1, grid_n)
    mesh = np.array(np.meshgrid(*([levels] * len(safe_factors)), indexing="ij"))
    flat = mesh.reshape(len(safe_factors), -1).T
    grid = pd.DataFrame(flat, columns=safe_factors)
    block_value = _prediction_block_value(model)
    if block_value is not None:
        grid["Block"] = block_value
    pred = model.predict(grid)
    idx = int(np.nanargmin(pred)) if goal == "Minimize" else int(np.nanargmax(pred))
    row = grid.iloc[idx].copy()
    pred_val = float(pred.iloc[idx]) if hasattr(pred, "iloc") else float(pred[idx])
    actual_vals = {}
    for _, info in factor_summary.iterrows():
        actual_vals[info["Factor"]] = info["Center"] + row[info["Code"]] * (info["High"] - info["Low"]) / 2.0
    return pd.DataFrame([{**actual_vals, "Predicted response": pred_val, "Goal": goal}])



def _make_interaction_plot(model, factor_summary, xfac, trace_factor, response, fixed_actual=None):
    cfg = common.safe_get_plot_cfg("DoE interaction")
    fig, ax = plt.subplots(figsize=(cfg["fig_w"], cfg["fig_h"]))
    fs = factor_summary.set_index("Factor")
    safe_x = fs.loc[xfac, "Code"]
    safe_t = fs.loc[trace_factor, "Code"]
    x_levels = np.array([-1.0, 1.0])
    fixed = {row["Code"]: 0.0 for _, row in factor_summary.iterrows()}
    if fixed_actual:
        for _, row in factor_summary.iterrows():
            if row["Factor"] in fixed_actual:
                half = (row["High"] - row["Low"]) / 2.0
                fixed[row["Code"]] = 0.0 if np.isclose(half, 0.0) else (fixed_actual[row["Factor"]] - row["Center"]) / half

    legend_labels = []
    for trace_level, label in [(-1.0, f"{trace_factor}: low"), (1.0, f"{trace_factor}: high")]:
        pred_df = pd.DataFrame([{**fixed, safe_x: xv, safe_t: trace_level} for xv in x_levels])
        block_value = _prediction_block_value(model)
        if block_value is not None:
            pred_df["Block"] = block_value
        actual_x = fs.loc[xfac, "Center"] + x_levels * (fs.loc[xfac, "High"] - fs.loc[xfac, "Low"]) / 2.0
        ax.plot(actual_x, model.predict(pred_df), marker=cfg.get("marker_style", "o"), ms=max(4, int(cfg["marker_size"] * 0.35)), lw=cfg["line_width"], label=label)
        legend_labels.append(label)
    apply_ax_style(ax, f"Interaction plot for {response}", xfac, response, legend=True, plot_key="DoE interaction")
    return fig



def _make_contour_plot(model, factor_summary, xfac, yfac, response, fixed_actual=None, observed=None):
    cfg = common.safe_get_plot_cfg("DoE contour")
    fig, ax = plt.subplots(figsize=(cfg["fig_w"], cfg["fig_h"]))
    fs = factor_summary.set_index("Factor")
    safe_x = fs.loc[xfac, "Code"]
    safe_y = fs.loc[yfac, "Code"]
    x_actual = np.linspace(fs.loc[xfac, "Low"], fs.loc[xfac, "High"], 80)
    y_actual = np.linspace(fs.loc[yfac, "Low"], fs.loc[yfac, "High"], 80)
    xx, yy = np.meshgrid(x_actual, y_actual)

    base = {row["Code"]: 0.0 for _, row in factor_summary.iterrows()}
    if fixed_actual:
        for _, row in factor_summary.iterrows():
            if row["Factor"] in fixed_actual:
                half = (row["High"] - row["Low"]) / 2.0
                base[row["Code"]] = 0.0 if np.isclose(half, 0.0) else (fixed_actual[row["Factor"]] - row["Center"]) / half

    rows = []
    for x_val, y_val in zip(xx.ravel(), yy.ravel()):
        r = base.copy()
        hx = (fs.loc[xfac, "High"] - fs.loc[xfac, "Low"]) / 2.0
        hy = (fs.loc[yfac, "High"] - fs.loc[yfac, "Low"]) / 2.0
        r[safe_x] = 0.0 if np.isclose(hx, 0.0) else (x_val - fs.loc[xfac, "Center"]) / hx
        r[safe_y] = 0.0 if np.isclose(hy, 0.0) else (y_val - fs.loc[yfac, "Center"]) / hy
        rows.append(r)
    grid = pd.DataFrame(rows)
    block_value = _prediction_block_value(model)
    if block_value is not None:
        grid["Block"] = block_value
    zz = np.asarray(model.predict(grid), dtype=float).reshape(xx.shape)

    cs = ax.contourf(xx, yy, zz, levels=20, cmap="viridis")
    fig.colorbar(cs, ax=ax, label=response)
    contour_lines = ax.contour(xx, yy, zz, levels=8, colors="white", linewidths=0.5, alpha=0.55)
    ax.clabel(contour_lines, fmt="%.2f", fontsize=max(7, cfg["tick_label_size"] - 1))
    if observed is not None and not observed.empty:
        ax.scatter(observed[xfac], observed[yfac], c="white", edgecolor="black", s=max(24, cfg["marker_size"]), label="Observed runs")
    apply_ax_style(ax, f"Contour plot for {response}", xfac, yfac, legend=True, plot_key="DoE contour")
    return fig, xx, yy, zz



def _make_surface_plot(model, factor_summary, xfac, yfac, response, fixed_actual=None, observed=None):
    cfg = common.safe_get_plot_cfg("DoE surface")
    fig = plt.figure(figsize=(cfg["fig_w"], cfg["fig_h"] + 0.4))
    ax = fig.add_subplot(111, projection="3d")
    fs = factor_summary.set_index("Factor")
    safe_x = fs.loc[xfac, "Code"]
    safe_y = fs.loc[yfac, "Code"]
    x_actual = np.linspace(fs.loc[xfac, "Low"], fs.loc[xfac, "High"], 60)
    y_actual = np.linspace(fs.loc[yfac, "Low"], fs.loc[yfac, "High"], 60)
    xx, yy = np.meshgrid(x_actual, y_actual)

    base = {row["Code"]: 0.0 for _, row in factor_summary.iterrows()}
    if fixed_actual:
        for _, row in factor_summary.iterrows():
            if row["Factor"] in fixed_actual:
                half = (row["High"] - row["Low"]) / 2.0
                base[row["Code"]] = 0.0 if np.isclose(half, 0.0) else (fixed_actual[row["Factor"]] - row["Center"]) / half

    rows = []
    for x_val, y_val in zip(xx.ravel(), yy.ravel()):
        r = base.copy()
        hx = (fs.loc[xfac, "High"] - fs.loc[xfac, "Low"]) / 2.0
        hy = (fs.loc[yfac, "High"] - fs.loc[yfac, "Low"]) / 2.0
        r[safe_x] = 0.0 if np.isclose(hx, 0.0) else (x_val - fs.loc[xfac, "Center"]) / hx
        r[safe_y] = 0.0 if np.isclose(hy, 0.0) else (y_val - fs.loc[yfac, "Center"]) / hy
        rows.append(r)
    grid = pd.DataFrame(rows)
    block_value = _prediction_block_value(model)
    if block_value is not None:
        grid["Block"] = block_value
    zz = np.asarray(model.predict(grid), dtype=float).reshape(xx.shape)

    surf = ax.plot_surface(xx, yy, zz, cmap="viridis", edgecolor="none", alpha=0.9)
    if observed is not None and not observed.empty:
        ax.scatter(observed[xfac], observed[yfac], observed[response], c=cfg.get("marker_color", "black"), s=max(18, cfg["marker_size"] * 0.8), depthshade=True)
    ax.set_xlabel(xfac)
    ax.set_ylabel(yfac)
    ax.set_zlabel(response)
    ax.set_title(f"Response surface for {response}")
    fig.colorbar(surf, ax=ax, shrink=0.68, aspect=12)
    return fig



def _make_residual_diagnostics(model):
    cfg = common.safe_get_plot_cfg("DoE residual diagnostics")
    fig, axes = plt.subplots(2, 2, figsize=(max(8, cfg["fig_w"]), max(5.6, cfg["fig_h"] + 1.2)))
    resid = np.asarray(model.resid, dtype=float)
    fitted = np.asarray(model.fittedvalues, dtype=float)
    run_order = np.arange(1, len(resid) + 1)

    # Normal probability plot
    (osm, osr), (slope, intercept, r) = stats.probplot(resid, dist="norm")
    axes[0, 0].scatter(osm, osr, s=max(22, cfg["marker_size"]), label="Residuals")
    axes[0, 0].plot(osm, slope * np.asarray(osm) + intercept, lw=cfg["line_width"], label="Reference")
    apply_ax_style(axes[0, 0], "Normal probability plot", "Theoretical quantiles", "Residuals", legend=True, plot_key="DoE residual diagnostics")

    # Box plot
    axes[0, 1].boxplot(resid, vert=True, patch_artist=True)
    apply_ax_style(axes[0, 1], "Residual box plot", "", "Residuals", legend=False, plot_key="DoE residual diagnostics")
    axes[0, 1].set_xticks([])

    # Histogram
    bins = min(8, max(4, int(np.sqrt(len(resid)))))
    axes[1, 0].hist(resid, bins=bins, edgecolor="black")
    apply_ax_style(axes[1, 0], "Residual histogram", "Residuals", "Count", legend=False, plot_key="DoE residual diagnostics")

    # Run-order plot
    axes[1, 1].plot(run_order, resid, marker=cfg.get("marker_style", "o"), lw=cfg["line_width"])
    axes[1, 1].axhline(0, ls=cfg.get("aux_line_style", "--"), lw=cfg.get("aux_line_width", 1.0), color="gray")
    apply_ax_style(axes[1, 1], "Run-order plot of residuals", "Run order", "Residuals", legend=False, plot_key="DoE residual diagnostics")
    fig.tight_layout()
    return fig



def _make_predicted_vs_observed(model, response_name):
    cfg = common.safe_get_plot_cfg("DoE predicted vs observed")
    fig, ax = plt.subplots(figsize=(cfg["fig_w"], cfg["fig_h"]))
    obs = np.asarray(model.model.endog, dtype=float)
    pred = np.asarray(model.fittedvalues, dtype=float)
    ax.scatter(obs, pred, s=max(24, cfg["marker_size"]), label="Runs")
    lo = min(obs.min(), pred.min())
    hi = max(obs.max(), pred.max())
    ax.plot([lo, hi], [lo, hi], ls=cfg.get("aux_line_style", "--"), lw=cfg.get("aux_line_width", 1.0), label="Identity")
    apply_ax_style(ax, f"Observed vs predicted for {response_name}", f"Observed {response_name}", f"Predicted {response_name}", legend=True, plot_key="DoE predicted vs observed")
    return fig



def _fit_response_workflow(df, factors, response, model_type, block_col=None):
    d, safe_df, safe_factors, rename_map, inv_map, factor_summary, include_block, display_df = _prepare_doe_dataframe(df, factors, response, block_col=block_col)
    candidate_terms = _candidate_terms(safe_factors, model_type)
    full_terms, stepwise_history, selected_terms_raw, stepwise_model, _ = _backward_aic_stepwise(safe_df, candidate_terms, include_block=include_block)
    full_model, full_formula = _fit_model(safe_df, full_terms, include_block=include_block)
    selected_terms = _enforce_hierarchy(selected_terms_raw, candidate_terms)
    hierarchy_changed = [_normalize_term(t) for t in selected_terms] != [_normalize_term(t) for t in selected_terms_raw]
    selected_model, selected_formula = _fit_model(safe_df, selected_terms, include_block=include_block)

    full_coef = _coef_table(full_model, inv_map)
    selected_coef = _coef_table(selected_model, inv_map)
    full_fit_stats = _fit_stats_table(full_model)
    lack_of_fit = _lack_of_fit_table(safe_df, selected_model, safe_factors, include_block=include_block)
    selected_fit_stats = _fit_stats_table(selected_model, lack_of_fit=lack_of_fit)
    selected_effects = _effects_anova_table(selected_model, inv_map)
    selected_summary = _model_summary_anova(selected_model, lack_of_fit=lack_of_fit)
    model_comparison = _nested_model_comparison(selected_model, full_model) if _normalize_term(selected_formula) != _normalize_term(full_formula) else pd.DataFrame()
    equation = _model_equation(selected_model, inv_map, response)
    conclusions = _conclusion_lines(response, selected_terms, inv_map, selected_model, lack_of_fit)

    def _pretty_model_rhs(rhs):
        parts = [p.strip() for p in str(rhs).split("+")]
        return " + ".join([_term_to_pretty(p, inv_map) for p in parts if p])

    return {
        "data": d,
        "display_df": display_df,
        "safe_df": safe_df,
        "safe_factors": safe_factors,
        "inv_map": inv_map,
        "factor_summary": factor_summary,
        "include_block": include_block,
        "full_terms": full_terms,
        "selected_terms_raw": selected_terms_raw,
        "selected_terms": selected_terms,
        "hierarchy_changed": hierarchy_changed,
        "full_model": full_model,
        "selected_model": selected_model,
        "full_formula": full_formula,
        "selected_formula": selected_formula,
        "full_coef": full_coef,
        "selected_coef": selected_coef,
        "stepwise_history": stepwise_history.assign(
            Term=stepwise_history["Term"].map(lambda x: _term_to_pretty(x, inv_map) if x != "—" else x),
            Model=stepwise_history["Model"].map(_pretty_model_rhs),
        ),
        "full_fit_stats": full_fit_stats,
        "selected_fit_stats": selected_fit_stats,
        "selected_effects": selected_effects,
        "selected_summary": selected_summary,
        "lack_of_fit": lack_of_fit,
        "model_comparison": model_comparison,
        "equation": equation,
        "conclusions": conclusions,
        "response": response,
        "model_type": model_type,
        "block_col": block_col if include_block else None,
    }



def _analysis_description(result):
    fs = result["factor_summary"]
    lines = [
        f"{len(result['data'])} experimental runs were analysed for the response {result['response']}.",
        f"The fitted model structure started from a {result['model_type']} model using coded factors, where the observed minimum and maximum for each factor were transformed to -1 and +1.",
    ]
    if result["block_col"]:
        lines.append(f"Block was included in the model as a categorical term using column {result['block_col']}.")
    lines.append("Stepwise backward elimination based on AIC was applied, after which the hierarchy principle was enforced so that retained interaction or quadratic terms also kept their parent main effects.")
    return " ".join(lines)



def _steps_text():
    return [
        "Fit the full model to the selected response.",
        "Use stepwise backward elimination to identify a parsimonious model.",
        "Apply the hierarchy principle so main effects are retained with selected interaction or quadratic terms.",
        "Review residual diagnostics, interaction behavior, ANOVA statistics, contour plots, and the response surface.",
        "When multiple responses are available, compare contours and explore trade-offs.",
    ]



def _make_overlay_contour(primary_result, secondary_result, xfac, yfac, primary_level, secondary_level, fixed_actual=None):
    cfg = common.safe_get_plot_cfg("DoE overlay contour")
    fig, ax = plt.subplots(figsize=(cfg["fig_w"], cfg["fig_h"]))
    fs = primary_result["factor_summary"].set_index("Factor")
    safe_x = fs.loc[xfac, "Code"]
    safe_y = fs.loc[yfac, "Code"]
    x_actual = np.linspace(fs.loc[xfac, "Low"], fs.loc[xfac, "High"], 80)
    y_actual = np.linspace(fs.loc[yfac, "Low"], fs.loc[yfac, "High"], 80)
    xx, yy = np.meshgrid(x_actual, y_actual)

    def _predict_surface(result):
        base = {row["Code"]: 0.0 for _, row in result["factor_summary"].iterrows()}
        if fixed_actual:
            for _, row in result["factor_summary"].iterrows():
                if row["Factor"] in fixed_actual:
                    half = (row["High"] - row["Low"]) / 2.0
                    base[row["Code"]] = 0.0 if np.isclose(half, 0.0) else (fixed_actual[row["Factor"]] - row["Center"]) / half
        rows = []
        fs_loc = result["factor_summary"].set_index("Factor")
        for x_val, y_val in zip(xx.ravel(), yy.ravel()):
            r = base.copy()
            hx = (fs_loc.loc[xfac, "High"] - fs_loc.loc[xfac, "Low"]) / 2.0
            hy = (fs_loc.loc[yfac, "High"] - fs_loc.loc[yfac, "Low"]) / 2.0
            r[fs_loc.loc[xfac, "Code"]] = 0.0 if np.isclose(hx, 0.0) else (x_val - fs_loc.loc[xfac, "Center"]) / hx
            r[fs_loc.loc[yfac, "Code"]] = 0.0 if np.isclose(hy, 0.0) else (y_val - fs_loc.loc[yfac, "Center"]) / hy
            rows.append(r)
        pred_df = pd.DataFrame(rows)
        block_value = _prediction_block_value(result["selected_model"])
        if block_value is not None:
            pred_df["Block"] = block_value
        return np.asarray(result["selected_model"].predict(pred_df), dtype=float).reshape(xx.shape)

    z1 = _predict_surface(primary_result)
    z2 = _predict_surface(secondary_result)
    cs1 = ax.contour(xx, yy, z1, levels=[primary_level], linewidths=2.0)
    cs2 = ax.contour(xx, yy, z2, levels=[secondary_level], linewidths=2.0, linestyles="--")
    ax.clabel(cs1, fmt={primary_level: f"{primary_result['response']}={primary_level:.2f}"}, inline=True, fontsize=max(7, cfg["tick_label_size"] - 1))
    ax.clabel(cs2, fmt={secondary_level: f"{secondary_result['response']}={secondary_level:.2f}"}, inline=True, fontsize=max(7, cfg["tick_label_size"] - 1))
    apply_ax_style(ax, "Overlay contour plot", xfac, yfac, legend=False, plot_key="DoE overlay contour")
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color="C0", lw=2, label=f"{primary_result['response']} target"),
        Line2D([0], [0], color="C1", lw=2, linestyle="--", label=f"{secondary_result['response']} target"),
    ]
    ax.legend(handles=handles, loc=cfg.get("legend_loc", "best"))
    return fig



def _make_doe_pdf_report(result, tables, figures, decimals=3, overlay_meta=None):
    bio = BytesIO()
    doc = common.SimpleDocTemplate(
        bio,
        pagesize=common.landscape(common.A4),
        leftMargin=1.2 * common.cm,
        rightMargin=1.2 * common.cm,
        topMargin=1.2 * common.cm,
        bottomMargin=1.1 * common.cm,
    )
    styles = common.getSampleStyleSheet()
    styles.add(common.ParagraphStyle(name="SmallBodyDOE", parent=styles["BodyText"], fontSize=9, leading=12, alignment=common.TA_LEFT))
    story = [
        common.Paragraph("Design of Experiments Analysis Report", styles["Title"]),
        common.Spacer(1, 0.12 * common.cm),
        common.Paragraph(f"Module: <b>DoE / Response Surfaces</b>", styles["Heading2"]),
        common.Spacer(1, 0.12 * common.cm),
        common.Paragraph("Data", styles["Heading2"]),
        common.Paragraph("The table below contains the experimental runs used for the selected DoE response analysis.", styles["SmallBodyDOE"]),
    ]
    story.extend(common._pdf_table(result["display_df"], styles, "Experimental data", decimals=decimals, max_rows=60))
    story.extend([
        common.Paragraph("Experimental description", styles["Heading2"]),
        common.Paragraph(_analysis_description(result), styles["SmallBodyDOE"]),
    ])
    story.extend(common._pdf_table(result["factor_summary"], styles, "Factor coding and levels", decimals=decimals, max_rows=20))

    story.extend([
        common.Paragraph("Analysis of DoE data", styles["Heading2"]),
        common.Paragraph("The analysis followed the response-surface workflow below.", styles["SmallBodyDOE"]),
    ])
    for i, step in enumerate(_steps_text(), start=1):
        story.append(common.Paragraph(f"{i}. {step}", styles["SmallBodyDOE"]))
    story.append(common.Spacer(1, 0.1 * common.cm))
    story.append(common.Paragraph(f"Fit full model to the {result['response']} response", styles["Heading3"]))
    story.extend(common._pdf_table(result["full_coef"], styles, "Full model coefficients", decimals=decimals, max_rows=30))
    story.extend(common._pdf_table(result["full_fit_stats"], styles, "Full model fit statistics", decimals=decimals, max_rows=20))
    story.append(common.Paragraph("Stepwise regression", styles["Heading3"]))
    story.extend(common._pdf_table(result["stepwise_history"], styles, "Stepwise AIC reduction history", decimals=decimals, max_rows=30))
    if result["hierarchy_changed"]:
        story.append(common.Paragraph("The hierarchy principle modified the raw stepwise result by retaining parent main effects for selected higher-order terms.", styles["SmallBodyDOE"]))
    story.append(common.Paragraph(f"Analysis of selected model for {result['response']}", styles["Heading3"]))
    story.extend(common._pdf_table(result["selected_summary"], styles, "Selected model ANOVA summary", decimals=decimals, max_rows=20))
    story.extend(common._pdf_table(result["selected_effects"], styles, "Selected model effects ANOVA", decimals=decimals, max_rows=30))
    story.extend(common._pdf_table(result["selected_coef"], styles, "Selected model coefficients", decimals=decimals, max_rows=30))
    story.extend(common._pdf_table(result["selected_fit_stats"], styles, "Selected model fit statistics", decimals=decimals, max_rows=20))
    if not result["model_comparison"].empty:
        story.extend(common._pdf_table(result["model_comparison"], styles, "Comparison of selected and full models", decimals=decimals, max_rows=10))
    story.append(common.Paragraph("Final model equation", styles["Heading3"]))
    story.append(common.Paragraph(result["equation"], styles["SmallBodyDOE"]))
    if overlay_meta:
        story.append(common.Spacer(1, 0.1 * common.cm))
        story.append(common.Paragraph("Multiple-response contour comparison", styles["Heading3"]))
        story.append(common.Paragraph(overlay_meta, styles["SmallBodyDOE"]))
    story.append(common.Spacer(1, 0.15 * common.cm))
    story.append(common.Paragraph("Conclusions", styles["Heading2"]))
    for line in result["conclusions"]:
        story.append(common.Paragraph(f"• {line}", styles["SmallBodyDOE"]))

    if figures:
        story.extend([common.PageBreak(), common.Paragraph("Figures", styles["Heading2"])])
        for caption, fig_bytes in figures:
            story.extend([common.Paragraph(caption, styles["Heading3"]), common.Image(BytesIO(fig_bytes))])
            story[-1]._restrictSize(24.5 * common.cm, 13.3 * common.cm)
            story.append(common.Spacer(1, 0.25 * common.cm))

    doc.build(story)
    bio.seek(0)
    return bio.getvalue()



def _render_analysis_ui(df, factors, response, model_type, block_col, decimals):
    result = _fit_response_workflow(df, factors, response, model_type, block_col=block_col)

    st.markdown("### Data")
    report_table(result["display_df"], "Experimental data (original and coded factors)", decimals)

    st.markdown("### Experimental description")
    info_box(_analysis_description(result))
    report_table(result["factor_summary"], "Factor coding and levels", decimals)

    st.markdown("### Analysis of DoE data")
    st.markdown("<br>".join([f"{i}. {step}" for i, step in enumerate(_steps_text(), start=1)]), unsafe_allow_html=True)

    st.markdown(f"#### Fit full model to the {response} response")
    report_table(result["full_coef"], "Full model coefficients", decimals)
    report_table(result["full_fit_stats"], "Full model fit statistics", decimals)

    st.markdown("#### Stepwise regression")
    report_table(result["stepwise_history"], "Backward AIC reduction history", decimals)
    if result["hierarchy_changed"]:
        st.caption("Hierarchy note: the final selected model was adjusted to retain parent main effects for selected interaction or quadratic terms.")

    st.markdown(f"#### Analysis of selected model for {response}")
    report_table(result["selected_summary"], "Selected model ANOVA summary", decimals)
    report_table(result["selected_effects"], "Selected model effects ANOVA", decimals)
    report_table(result["selected_coef"], "Selected model coefficients", decimals)
    report_table(result["selected_fit_stats"], "Selected model fit statistics", decimals)
    if not result["model_comparison"].empty:
        report_table(result["model_comparison"], "Comparison of selected and full models", decimals)

    st.markdown("#### Final response-surface model")
    st.code(result["equation"])

    st.markdown("### Model Graphs")
    plot_cols = st.columns([1, 1, 1])
    factor_names = list(result["factor_summary"]["Factor"])
    with plot_cols[0]:
        xfac = st.selectbox("X-axis factor", factor_names, index=0, key=f"x_{response}")
    with plot_cols[1]:
        y_opts = [f for f in factor_names if f != xfac]
        yfac = st.selectbox("Y-axis factor", y_opts, index=0, key=f"y_{response}")
    with plot_cols[2]:
        interaction_factor = st.selectbox("Interaction trace factor", [f for f in factor_names if f != xfac], index=0, key=f"trace_{response}")

    other_factors = [f for f in factor_names if f not in [xfac, yfac]]
    fixed_actual = {}
    if other_factors:
        st.markdown("**Fixed levels for remaining factors**")
        cols = st.columns(len(other_factors))
        fs = result["factor_summary"].set_index("Factor")
        for i, fac in enumerate(other_factors):
            fixed_actual[fac] = cols[i].slider(
                fac,
                min_value=float(fs.loc[fac, "Low"]),
                max_value=float(fs.loc[fac, "High"]),
                value=float(fs.loc[fac, "Center"]),
                key=f"fix_{response}_{fac}",
            )

    interaction_fig = _make_interaction_plot(result["selected_model"], result["factor_summary"], xfac, interaction_factor, response, fixed_actual=fixed_actual)
    contour_fig, _, _, _ = _make_contour_plot(result["selected_model"], result["factor_summary"], xfac, yfac, response, fixed_actual=fixed_actual, observed=result["data"])
    surface_fig = _make_surface_plot(result["selected_model"], result["factor_summary"], xfac, yfac, response, fixed_actual=fixed_actual, observed=result["data"])
    diag_fig = _make_residual_diagnostics(result["selected_model"])
    pred_obs_fig = _make_predicted_vs_observed(result["selected_model"], response)

    st.pyplot(interaction_fig)
    st.pyplot(contour_fig)
    st.pyplot(surface_fig)
    st.pyplot(diag_fig)
    st.pyplot(pred_obs_fig)

    overlay_fig = None
    overlay_meta = None
    other_responses = [c for c in get_numeric_columns(df) if c not in factors + [response]]
    if other_responses:
        st.markdown("### Multiple-response contour comparison")
        c1, c2, c3 = st.columns([1.2, 1, 1])
        with c1:
            overlay_response = st.selectbox("Second response for overlay contour", other_responses, key=f"overlay_resp_{response}")
        secondary_result = _fit_response_workflow(df, factors, overlay_response, model_type, block_col=block_col)
        with c2:
            primary_level = st.number_input(f"Target contour for {response}", value=float(result["data"][response].mean()), key=f"lev1_{response}")
        with c3:
            secondary_level = st.number_input(f"Target contour for {overlay_response}", value=float(secondary_result["data"][overlay_response].mean()), key=f"lev2_{response}")
        overlay_fig = _make_overlay_contour(result, secondary_result, xfac, yfac, float(primary_level), float(secondary_level), fixed_actual=fixed_actual)
        overlay_meta = (
            f"Overlay contour lines were generated for {response} at {float(primary_level):.{decimals}f} and "
            f"{overlay_response} at {float(secondary_level):.{decimals}f} using separately selected hierarchical models."
        )
        st.pyplot(overlay_fig)

    st.markdown("### Conclusions from the analysis")
    for line in result["conclusions"]:
        st.markdown(f"- {line}")

    goal = st.selectbox("Optimization goal for selected response", ["Minimize", "Maximize"], key=f"goal_{response}")
    optimum = _grid_search_optimum(result["selected_model"], result["safe_factors"], result["factor_summary"], goal=goal)
    report_table(optimum, f"Grid-search optimum for {response}", decimals)

    workbook_tables = {
        "Experimental Data": result["display_df"],
        "Factor Summary": result["factor_summary"],
        "Full Model Coef": result["full_coef"],
        "Full Model Fit": result["full_fit_stats"],
        "Stepwise History": result["stepwise_history"],
        "Selected ANOVA": result["selected_summary"],
        "Effects ANOVA": result["selected_effects"],
        "Selected Coef": result["selected_coef"],
        "Selected Fit": result["selected_fit_stats"],
        "Optimum": optimum,
    }
    if not result["model_comparison"].empty:
        workbook_tables["Model Comparison"] = result["model_comparison"]

    excel_bytes = make_excel_bytes(workbook_tables)
    figures = [
        ("Interaction plot", fig_to_png_bytes(interaction_fig)),
        ("Contour plot", fig_to_png_bytes(contour_fig)),
        ("Response surface", fig_to_png_bytes(surface_fig)),
        ("Residual diagnostics", fig_to_png_bytes(diag_fig)),
        ("Observed vs predicted", fig_to_png_bytes(pred_obs_fig)),
    ]
    if overlay_fig is not None:
        figures.append(("Overlay contour plot", fig_to_png_bytes(overlay_fig)))

    pdf_bytes = _make_doe_pdf_report(result, workbook_tables, figures, decimals=decimals, overlay_meta=overlay_meta)

    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "Download DOE workbook",
            excel_bytes,
            file_name=f"doe_{response.lower()}_analysis.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    with c2:
        st.download_button(
            "Download DOE report",
            pdf_bytes,
            file_name=f"doe_{response.lower()}_report.pdf",
            mime="application/pdf",
        )


# ---------- public render ----------

def render():
    render_display_settings()
    st.sidebar.title("🧪 DoE Studio")
    st.sidebar.markdown("Design of Experiments")

    app_header("🧪 DoE Studio", "Design builder and NIST-style response-surface analysis in one place.")
    tabs = st.tabs(["Design Builder", "Analyze Responses"])

    with tabs[0]:
        st.subheader("Design Builder")
        info_box("Create a basic 2-level full-factorial design with blocks, center points, replicates, and randomization.")
        st.button("Sample Data", key="sample_doe_design", on_click=_load_sample_design)

        c1, c2 = st.columns([1, 1])
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
            design = _build_factorial_design(
                factor_names, lows, highs,
                blocks=int(blocks),
                center_points=int(center_points),
                replicates=int(replicates),
                randomize=randomize,
                seed=int(seed),
            )
            st.session_state["doe_generated_design"] = design

        if "doe_generated_design" in st.session_state:
            design = st.session_state["doe_generated_design"]
            st.success(f"Generated design with {len(design)} runs")
            st.dataframe(design, width="stretch")
            excel_bytes = make_excel_bytes({"Design": design})
            st.download_button(
                "Download design workbook",
                excel_bytes,
                file_name="doe_design.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
            st.download_button(
                "Download design CSV",
                design.to_csv(index=False).encode("utf-8"),
                file_name="doe_design.csv",
                mime="text/csv",
            )
            info_box("After collecting experimental responses, switch to the Analyze Responses tab and paste the completed design table there.")

    with tabs[1]:
        st.subheader("Response Analysis")
        info_box("Paste completed DoE data with factor columns and one or more response columns to run a NIST-style response-surface workflow: full model, stepwise reduction, hierarchy-aware selected model, diagnostics, contour plots, and reporting.")

        c_sample, c_text = st.columns([1, 5])
        with c_sample:
            st.button("Sample Data", key="sample_doe_response", on_click=_load_sample_response_text)
        with c_text:
            data_input = st.text_area("Paste completed DoE data with headers", height=240, key="doe_response_input")
        decimals = st.slider("Decimals", 1, 8, DEFAULT_DECIMALS, key="doe_dec")

        if data_input:
            try:
                df = parse_pasted_table(data_input, header=True)
                if df is None or df.empty:
                    raise ValueError("No data could be parsed from the pasted table.")
                num_cols = get_numeric_columns(df)
                all_cols = list(df.columns)

                c1, c2, c3, c4 = st.columns([1.45, 1.1, 1.15, 1.2])
                with c1:
                    factors = st.multiselect("Numeric factors", num_cols, default=num_cols[: min(2, len(num_cols))], key="doe_factors")
                with c2:
                    candidate_responses = [c for c in num_cols if c not in factors] or num_cols
                    response = st.selectbox("Response", candidate_responses, key="doe_response")
                with c3:
                    model_type = st.selectbox("Model type", ["linear", "interaction", "quadratic"], index=2, key="doe_model_type")
                with c4:
                    block_col = st.selectbox("Block column (optional)", ["(None)"] + [c for c in all_cols if c not in factors + [response]], key="doe_block")

                if len(factors) < 2:
                    st.info("Select at least two numeric factors to analyze the response surface.")
                else:
                    _render_analysis_ui(df, factors, response, model_type, block_col, decimals)
            except Exception as e:
                st.error(str(e))
