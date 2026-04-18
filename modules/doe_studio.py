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




def _default_cosolvent_settings():
    return {
        "cosolvent_names": ["Propylene Glycol", "Glycerol", "PEG400"],
        "cosolvent_lows": [0.000, 0.000, 0.000],
        "cosolvent_highs": [0.125, 0.200, 0.150],
        "water_name": "Water",
        "total": 1.0,
    }


def _build_cosolvent_design(
    cosolvent_names,
    cosolvent_lows,
    cosolvent_highs,
    water_name="Water",
    total=1.0,
    blocks=1,
    replicates=1,
    randomize=True,
    seed=123,
):
    levels = []
    coded_levels = []
    for low, high in zip(cosolvent_lows, cosolvent_highs):
        low = float(low)
        high = float(high)
        if high < low:
            low, high = high, low
        mid = (low + high) / 2.0
        vals = [low, mid, high]
        unique_vals = []
        unique_codes = []
        for code, val in zip([-1, 0, 1], vals):
            if not any(abs(val - u) < 1e-12 for u in unique_vals):
                unique_vals.append(float(val))
                unique_codes.append(code)
        levels.append(unique_vals)
        coded_levels.append(unique_codes)

    runs = []
    combo_idx = [list(range(len(v))) for v in levels]
    for block in range(1, blocks + 1):
        for rep in range(1, replicates + 1):
            for idxs in product(*combo_idx):
                amounts = [levels[i][j] for i, j in enumerate(idxs)]
                water_amt = float(total) - float(sum(amounts))
                if water_amt < -1e-12:
                    continue
                row = {"Block": block, "Replicate": rep, "RunType": "Co-Solvent constrained blend"}
                for i, name in enumerate(cosolvent_names):
                    row[f"{_safe_factor_prefix(i)} (coded)"] = int(coded_levels[i][idxs[i]])
                    row[name] = float(amounts[i])
                row[water_name] = max(0.0, float(water_amt))
                row["Total fill volume"] = float(total)
                row["Cosolvent total"] = float(sum(amounts))
                for name in list(cosolvent_names) + [water_name]:
                    row[f"{name} (fraction)"] = row[name] / float(total) if float(total) > 0 else np.nan
                runs.append(row)
    design = pd.DataFrame(runs)
    if randomize and len(design) > 0:
        design = _randomize_design_within_block(design, seed=seed)
    if len(design) > 0:
        design.insert(0, "Run", np.arange(1, len(design) + 1))
    return design



def _build_cosolvent_process_design(
    cosolvent_names,
    cosolvent_lows,
    cosolvent_highs,
    process_factor_names,
    process_lows,
    process_highs,
    water_name="Water",
    total=1.0,
    process_design_kind="factorial",
    blocks=1,
    replicates=1,
    randomize=True,
    seed=123,
):
    mix_design = _build_cosolvent_design(
        cosolvent_names,
        cosolvent_lows,
        cosolvent_highs,
        water_name=water_name,
        total=total,
        blocks=1,
        replicates=1,
        randomize=False,
        seed=seed,
    )
    if process_design_kind == "ccd":
        proc_design = _build_ccd_design(process_factor_names, process_lows, process_highs, blocks=1, center_points=2, replicates=1, randomize=False, seed=seed)
    else:
        proc_design = _build_factorial_design(process_factor_names, process_lows, process_highs, blocks=1, center_points=0, replicates=1, randomize=False, seed=seed)
    mix_cols = [c for c in mix_design.columns if c not in ["Run", "Block", "Replicate", "RunType"]]
    proc_cols = [c for c in proc_design.columns if c not in ["Run", "Block", "Replicate", "RunType"]]
    runs = []
    for block in range(1, blocks + 1):
        for rep in range(1, replicates + 1):
            for _, mix_row in mix_design.iterrows():
                for _, proc_row in proc_design.iterrows():
                    row = {"Block": block, "Replicate": rep, "RunType": f"Co-Solvent constrained blend + {process_design_kind.upper()}"}
                    for c in mix_cols:
                        row[c] = mix_row[c]
                    for c in proc_cols:
                        row[c] = proc_row[c]
                    runs.append(row)
    design = pd.DataFrame(runs)
    if randomize and len(design) > 0:
        design = _randomize_design_within_block(design, seed=seed)
    if len(design) > 0:
        design.insert(0, "Run", np.arange(1, len(design) + 1))
    return design



def _make_cosolvent_sample_dataframe(process=False):
    s = _default_cosolvent_settings()
    if process:
        design = _build_cosolvent_process_design(
            s["cosolvent_names"],
            s["cosolvent_lows"],
            s["cosolvent_highs"],
            ["pH", "Temperature"],
            [3.5, 25.0],
            [6.5, 40.0],
            water_name=s["water_name"],
            total=s["total"],
            process_design_kind="factorial",
            blocks=1,
            replicates=1,
            randomize=False,
            seed=123,
        )
    else:
        design = _build_cosolvent_design(
            s["cosolvent_names"],
            s["cosolvent_lows"],
            s["cosolvent_highs"],
            water_name=s["water_name"],
            total=s["total"],
            blocks=1,
            replicates=1,
            randomize=False,
            seed=123,
        )
    pg = design[s["cosolvent_names"][0]].to_numpy(dtype=float)
    gly = design[s["cosolvent_names"][1]].to_numpy(dtype=float)
    peg = design[s["cosolvent_names"][2]].to_numpy(dtype=float)
    water = design[s["water_name"]].to_numpy(dtype=float)
    sol = (
        3.0
        + 42.0 * pg
        + 20.0 * gly
        + 31.0 * peg
        + 55.0 * pg * peg
        + 18.0 * gly * peg
        + 10.0 * pg * gly
        + 5.0 * water
    )
    visc = 2.5 + 180.0 * pg + 420.0 * gly + 320.0 * peg + 55.0 * pg * gly
    if process:
        ph = design["pH"].to_numpy(dtype=float)
        temp = design["Temperature"].to_numpy(dtype=float)
        sol = sol + 0.55 * (ph - 5.0) + 0.08 * (temp - 25.0) + 2.5 * pg * (ph - 5.0)
        visc = visc - 0.05 * (temp - 25.0) + 0.12 * (ph - 5.0)
    design["Solubility (mg/mL)"] = np.round(sol, 3)
    design["Viscosity (cP)"] = np.round(visc, 3)
    design["Water fraction"] = np.round(design[s["water_name"]] / float(s["total"]), 6)
    return design



def _load_sample_response_text_cosolvent():
    st.session_state["doe_analysis_family"] = DOE_FAMILY_MIXTURE
    st.session_state["doe_response_input"] = _make_cosolvent_sample_dataframe(process=False).to_csv(sep="	", index=False)



def _load_sample_response_text_cosolvent_process():
    st.session_state["doe_analysis_family"] = DOE_FAMILY_MIXPROC
    st.session_state["doe_response_input"] = _make_cosolvent_sample_dataframe(process=True).to_csv(sep="	", index=False)

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
            if design_family in ["Co-Solvents Evaluation", "Co-Solvents Evaluation - Process"]:
                info_box("For co-solvent designs, keep the co-solvent amount columns and the auto-calculated water column in the final dataset. In analysis, treat them as mixture components, with any extra numeric settings such as pH or temperature entered as process factors.")
            else:
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

# ---------- extended DOE families (process / mixture / mixture-process) ----------

_legacy_render_analysis_ui = _render_analysis_ui
_legacy_make_doe_pdf_report = _make_doe_pdf_report
_legacy_analysis_description = _analysis_description
_legacy_steps_text = _steps_text
_legacy_fit_response_workflow = _fit_response_workflow

DOE_FAMILY_PROCESS = "Process / RSM"
DOE_FAMILY_MIXTURE = "Mixture"
DOE_FAMILY_MIXPROC = "Mixture-Process"

DOE_SAMPLE_MIXTURE_DATA = """Water\tPEG400\tEthanol\tSolubility\tViscosity
100\t0\t0\t4.2\t1.0
0\t100\t0\t16.8\t92.0
0\t0\t100\t11.5\t1.4
50\t50\t0\t10.7\t43.0
50\t0\t50\t8.9\t1.2
0\t50\t50\t14.1\t46.0
33.333\t33.333\t33.333\t12.4\t29.0
66.667\t16.667\t16.667\t7.3\t15.5
16.667\t66.667\t16.667\t14.8\t61.0
16.667\t16.667\t66.667\t10.9\t9.2
"""

DOE_SAMPLE_MIXPROC_DATA = """Water\tPEG400\tEthanol\tTemp\tpH\tSolubility\tViscosity
100\t0\t0\t25\t4.5\t4.1\t1.0
100\t0\t0\t40\t6.5\t5.0\t0.9
0\t100\t0\t25\t4.5\t16.0\t96.0
0\t100\t0\t40\t6.5\t18.6\t88.0
0\t0\t100\t25\t4.5\t11.2\t1.5
0\t0\t100\t40\t6.5\t13.0\t1.3
50\t50\t0\t25\t6.5\t11.1\t45.0
50\t50\t0\t40\t4.5\t12.7\t41.0
50\t0\t50\t25\t6.5\t9.1\t1.2
50\t0\t50\t40\t4.5\t10.8\t1.1
0\t50\t50\t25\t6.5\t15.0\t48.0
0\t50\t50\t40\t4.5\t16.8\t44.0
33.333\t33.333\t33.333\t25\t4.5\t12.2\t30.0
33.333\t33.333\t33.333\t40\t6.5\t14.0\t27.0
66.667\t16.667\t16.667\t25\t4.5\t7.0\t16.0
16.667\t66.667\t16.667\t40\t6.5\t16.9\t57.0
16.667\t16.667\t66.667\t25\t6.5\t11.4\t9.4
16.667\t16.667\t66.667\t40\t4.5\t12.8\t8.8
"""


def _load_sample_response_text_mixture():
    st.session_state["doe_response_input"] = DOE_SAMPLE_MIXTURE_DATA



def _load_sample_response_text_mixproc():
    st.session_state["doe_response_input"] = DOE_SAMPLE_MIXPROC_DATA



def _load_sample_mixture_design():
    st.session_state["doe_design_family"] = "Mixture simplex-centroid"
    st.session_state["doe_mix_n_components"] = 3
    st.session_state["doe_mix_total"] = 100.0
    st.session_state["doe_blocks"] = 1
    st.session_state["doe_replicates"] = 1
    st.session_state["doe_randomize"] = True
    st.session_state["doe_seed"] = 123
    names = ["Water", "PEG400", "Ethanol"]
    for i, name in enumerate(names):
        st.session_state[f"doe_mix_name_{i}"] = name
    st.session_state["doe_generated_design"] = _build_mixture_design(names, total=100.0, design_kind="simplex-centroid", blocks=1, replicates=1, randomize=True, seed=123)



def _load_sample_mixproc_design():
    st.session_state["doe_design_family"] = "Mixture-Process"
    st.session_state["doe_mix_n_components"] = 3
    st.session_state["doe_mix_total"] = 100.0
    st.session_state["doe_mp_n_process"] = 2
    st.session_state["doe_blocks"] = 1
    st.session_state["doe_replicates"] = 1
    st.session_state["doe_randomize"] = True
    st.session_state["doe_seed"] = 123
    for i, name in enumerate(["Water", "PEG400", "Ethanol"]):
        st.session_state[f"doe_mix_name_{i}"] = name
    proc_names = ["Temp", "pH"]
    proc_lows = [25.0, 4.5]
    proc_highs = [40.0, 6.5]
    for i, name in enumerate(proc_names):
        st.session_state[f"doe_mp_name_{i}"] = name
        st.session_state[f"doe_mp_low_{i}"] = proc_lows[i]
        st.session_state[f"doe_mp_high_{i}"] = proc_highs[i]
    st.session_state["doe_generated_design"] = _build_mixture_process_design(
        ["Water", "PEG400", "Ethanol"],
        ["Temp", "pH"],
        [25.0, 4.5],
        [40.0, 6.5],
        total=100.0,
        mixture_design_kind="simplex-centroid",
        process_design_kind="factorial",
        blocks=1,
        replicates=1,
        randomize=True,
        seed=123,
    )


def _load_sample_cosolvent_design():
    s = _default_cosolvent_settings()
    st.session_state["doe_design_family"] = "Co-Solvents Evaluation"
    st.session_state["doe_cs_n_cosolvents"] = len(s["cosolvent_names"])
    st.session_state["doe_cs_total"] = s["total"]
    st.session_state["doe_cs_water_name"] = s["water_name"]
    st.session_state["doe_blocks"] = 1
    st.session_state["doe_replicates"] = 1
    st.session_state["doe_randomize"] = True
    st.session_state["doe_seed"] = 123
    for i, name in enumerate(s["cosolvent_names"]):
        st.session_state[f"doe_cs_name_{i}"] = name
        st.session_state[f"doe_cs_low_{i}"] = s["cosolvent_lows"][i]
        st.session_state[f"doe_cs_high_{i}"] = s["cosolvent_highs"][i]
    st.session_state["doe_generated_design"] = _build_cosolvent_design(
        s["cosolvent_names"],
        s["cosolvent_lows"],
        s["cosolvent_highs"],
        water_name=s["water_name"],
        total=s["total"],
        blocks=1,
        replicates=1,
        randomize=True,
        seed=123,
    )



def _load_sample_cosolvent_process_design():
    s = _default_cosolvent_settings()
    st.session_state["doe_design_family"] = "Co-Solvents Evaluation - Process"
    st.session_state["doe_cs_n_cosolvents"] = len(s["cosolvent_names"])
    st.session_state["doe_cs_total"] = s["total"]
    st.session_state["doe_cs_water_name"] = s["water_name"]
    st.session_state["doe_mp_n_process"] = 2
    st.session_state["doe_cs_proc_kind"] = "factorial"
    st.session_state["doe_blocks"] = 1
    st.session_state["doe_replicates"] = 1
    st.session_state["doe_randomize"] = True
    st.session_state["doe_seed"] = 123
    for i, name in enumerate(s["cosolvent_names"]):
        st.session_state[f"doe_cs_name_{i}"] = name
        st.session_state[f"doe_cs_low_{i}"] = s["cosolvent_lows"][i]
        st.session_state[f"doe_cs_high_{i}"] = s["cosolvent_highs"][i]
    proc_names = ["pH", "Temperature"]
    proc_lows = [3.5, 25.0]
    proc_highs = [6.5, 40.0]
    for i, name in enumerate(proc_names):
        st.session_state[f"doe_cs_proc_name_{i}"] = name
        st.session_state[f"doe_cs_proc_low_{i}"] = proc_lows[i]
        st.session_state[f"doe_cs_proc_high_{i}"] = proc_highs[i]
    st.session_state["doe_generated_design"] = _build_cosolvent_process_design(
        s["cosolvent_names"],
        s["cosolvent_lows"],
        s["cosolvent_highs"],
        proc_names,
        proc_lows,
        proc_highs,
        water_name=s["water_name"],
        total=s["total"],
        process_design_kind="factorial",
        blocks=1,
        replicates=1,
        randomize=True,
        seed=123,
    )



def _hierarchy_additions(terms):
    raw = {_normalize_term(t): t for t in terms}
    needed = set(raw.keys())
    for t in list(raw.keys()):
        if ":" in t:
            parts = [p for p in t.split(":") if p]
            for p in parts:
                needed.add(p)
            if len(parts) >= 3:
                for i in range(len(parts)):
                    for j in range(i + 1, len(parts)):
                        needed.add(f"{parts[i]}:{parts[j]}")
        m = re.match(r"I\(([^*]+)\*\*2\)", t)
        if m:
            needed.add(m.group(1))
    return needed



def _enforce_hierarchy(terms, candidate_terms):
    selected_norm = _hierarchy_additions(terms)
    out = []
    for t in candidate_terms:
        if _normalize_term(t) in selected_norm:
            out.append(t)
    return out



def _simplex_centroid_points(q):
    pts = []
    idx = list(range(q))
    for k in range(1, q + 1):
        for comb in combinations(idx, k):
            vec = np.zeros(q, dtype=float)
            vec[list(comb)] = 1.0 / k
            pts.append(vec)
    return pts



def _integer_partitions(total, parts):
    if parts == 1:
        yield (total,)
    else:
        for i in range(total + 1):
            for rest in _integer_partitions(total - i, parts - 1):
                yield (i,) + rest



def _simplex_lattice_points(q, degree=2):
    pts = []
    for part in _integer_partitions(int(degree), q):
        pts.append(np.array(part, dtype=float) / float(degree))
    return pts



def _randomize_design_within_block(design, seed=123):
    if design.empty or "Block" not in design.columns:
        return design
    rng = np.random.default_rng(seed)
    parts = []
    for _, sub in design.groupby("Block", sort=True):
        sub = sub.sample(frac=1, random_state=int(rng.integers(0, 1_000_000))).reset_index(drop=True)
        parts.append(sub)
    return pd.concat(parts, ignore_index=True)



def _build_ccd_design(
    factor_names,
    low_levels,
    high_levels,
    blocks=1,
    center_points=2,
    replicates=1,
    randomize=True,
    seed=123,
    alpha=1.0,
):
    coded_factorial = list(product([-1, 1], repeat=len(factor_names)))
    axial = []
    for i in range(len(factor_names)):
        for s in (-alpha, alpha):
            vec = [0.0] * len(factor_names)
            vec[i] = float(s)
            axial.append(tuple(vec))
    runs = []
    for block in range(1, blocks + 1):
        for rep in range(1, replicates + 1):
            for combo in coded_factorial:
                row = {"Block": block, "Replicate": rep, "RunType": "Factorial"}
                for i, name in enumerate(factor_names):
                    row[f"{_safe_factor_prefix(i)} (coded)"] = float(combo[i])
                    center = (low_levels[i] + high_levels[i]) / 2.0
                    half = (high_levels[i] - low_levels[i]) / 2.0
                    row[name] = center + float(combo[i]) * half
                runs.append(row)
            for combo in axial:
                row = {"Block": block, "Replicate": rep, "RunType": "Axial"}
                for i, name in enumerate(factor_names):
                    row[f"{_safe_factor_prefix(i)} (coded)"] = float(combo[i])
                    center = (low_levels[i] + high_levels[i]) / 2.0
                    half = (high_levels[i] - low_levels[i]) / 2.0
                    row[name] = center + float(combo[i]) * half
                runs.append(row)
            for _ in range(center_points):
                row = {"Block": block, "Replicate": rep, "RunType": "Center"}
                for i, name in enumerate(factor_names):
                    row[f"{_safe_factor_prefix(i)} (coded)"] = 0.0
                    row[name] = (low_levels[i] + high_levels[i]) / 2.0
                runs.append(row)
    design = pd.DataFrame(runs)
    if randomize:
        design = _randomize_design_within_block(design, seed=seed)
    design.insert(0, "Run", np.arange(1, len(design) + 1))
    return design



def _build_mixture_design(component_names, total=100.0, design_kind="simplex-centroid", degree=2, blocks=1, replicates=1, randomize=True, seed=123):
    q = len(component_names)
    pts = _simplex_centroid_points(q) if design_kind == "simplex-centroid" else _simplex_lattice_points(q, degree=degree)
    runs = []
    for block in range(1, blocks + 1):
        for rep in range(1, replicates + 1):
            for p in pts:
                row = {"Block": block, "Replicate": rep, "RunType": design_kind.title()}
                for i, name in enumerate(component_names):
                    row[f"{_safe_factor_prefix(i)} (fraction)"] = float(p[i])
                    row[name] = float(total) * float(p[i])
                runs.append(row)
    design = pd.DataFrame(runs)
    if randomize:
        design = _randomize_design_within_block(design, seed=seed)
    design.insert(0, "Run", np.arange(1, len(design) + 1))
    return design



def _build_mixture_process_design(
    component_names,
    process_factor_names,
    process_lows,
    process_highs,
    total=100.0,
    mixture_design_kind="simplex-centroid",
    process_design_kind="factorial",
    blocks=1,
    replicates=1,
    randomize=True,
    seed=123,
):
    mix_design = _build_mixture_design(component_names, total=total, design_kind=mixture_design_kind, degree=2, blocks=1, replicates=1, randomize=False, seed=seed)
    if process_design_kind == "ccd":
        proc_design = _build_ccd_design(process_factor_names, process_lows, process_highs, blocks=1, center_points=2, replicates=1, randomize=False, seed=seed)
    else:
        proc_design = _build_factorial_design(process_factor_names, process_lows, process_highs, blocks=1, center_points=0, replicates=1, randomize=False, seed=seed)
    mix_cols = [c for c in mix_design.columns if c not in ["Run", "Block", "Replicate", "RunType"]]
    proc_cols = [c for c in proc_design.columns if c not in ["Run", "Block", "Replicate", "RunType"]]
    runs = []
    for block in range(1, blocks + 1):
        for rep in range(1, replicates + 1):
            for _, mix_row in mix_design.iterrows():
                for _, proc_row in proc_design.iterrows():
                    row = {"Block": block, "Replicate": rep, "RunType": f"{mixture_design_kind.title()} + {process_design_kind.upper()}"}
                    for c in mix_cols:
                        row[c] = mix_row[c]
                    for c in proc_cols:
                        row[c] = proc_row[c]
                    runs.append(row)
    design = pd.DataFrame(runs)
    if randomize:
        design = _randomize_design_within_block(design, seed=seed)
    design.insert(0, "Run", np.arange(1, len(design) + 1))
    return design



def _build_formula_general(terms, include_block=False, no_intercept=False):
    rhs = list(terms)
    if include_block:
        rhs.append("C(Block)")
    core = " + ".join(rhs) if rhs else "1"
    if no_intercept:
        core = "0 + " + core if core != "1" else "0"
    return "Response ~ " + core



def _fit_model_general(data, terms, include_block=False, no_intercept=False):
    formula = _build_formula_general(terms, include_block=include_block, no_intercept=no_intercept)
    model = smf.ols(formula, data=data).fit()
    return model, formula



def _backward_aic_stepwise_general(data, candidate_terms, include_block=False, no_intercept=False):
    current = list(candidate_terms)
    history = []
    model, formula = _fit_model_general(data, current, include_block=include_block, no_intercept=no_intercept)
    current_aic = float(model.aic)
    history.append({"Step": 0, "Action": "Start", "Term": "—", "AIC": current_aic, "Model": formula.replace("Response ~ ", "")})
    step = 0
    while len(current) > 1:
        trial_rows = []
        for term in current:
            trial_terms = [t for t in current if t != term]
            try:
                trial_model, trial_formula = _fit_model_general(data, trial_terms, include_block=include_block, no_intercept=no_intercept)
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
    selected_model, selected_formula = _fit_model_general(data, current, include_block=include_block, no_intercept=no_intercept)
    return list(candidate_terms), pd.DataFrame(history), current, selected_model, selected_formula



def _candidate_terms_mixture(safe_components, model_type):
    terms = list(safe_components)
    if model_type in ["quadratic", "special_cubic"]:
        terms.extend([":".join(comb) for comb in combinations(safe_components, 2)])
    if model_type == "special_cubic":
        terms.extend([":".join(comb) for comb in combinations(safe_components, 3)])
    return terms



def _prepare_mixture_dataframe(df, components, response, block_col=None):
    use_cols = components + [response] + ([block_col] if block_col and block_col != "(None)" else [])
    d = df[use_cols].copy()
    for c in components + [response]:
        d[c] = to_numeric(d[c])
    d = d.dropna(subset=components + [response]).reset_index(drop=True)
    d.insert(0, "RunOrder", np.arange(1, len(d) + 1))
    totals = d[components].sum(axis=1)
    if (totals <= 0).any():
        raise ValueError("Mixture component rows must have positive totals.")
    total_ref = float(np.nanmedian(totals.to_numpy(dtype=float)))
    normalized = d[components].div(totals, axis=0)
    safe_components = [f"M{i+1}" for i in range(len(components))]
    rename_map = {orig: safe for orig, safe in zip(components, safe_components)}
    inv_map = {safe: orig for orig, safe in zip(components, safe_components)}
    safe_df = pd.DataFrame(index=d.index)
    rows = []
    for orig, safe in rename_map.items():
        safe_df[safe] = normalized[orig].to_numpy(dtype=float)
        rows.append({
            "Factor": orig,
            "Code": safe,
            "Type": "Mixture component",
            "Min": float(normalized[orig].min()),
            "Center": float(normalized[orig].mean()),
            "High": float(normalized[orig].max()),
            "Range": float(normalized[orig].max() - normalized[orig].min()),
            "Input min": float(d[orig].min()),
            "Input max": float(d[orig].max()),
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
    display_df["Mixture total"] = totals
    for orig, safe in rename_map.items():
        display_df[f"{orig} (fraction)"] = normalized[orig]
    factor_summary = pd.DataFrame(rows)
    note = "Selected mixture components were normalised row-wise to a total of 1.0 for model fitting."
    return d, safe_df, safe_components, rename_map, inv_map, factor_summary, include_block, display_df, total_ref, note



def _prepare_mixture_process_dataframe(df, components, process_factors, response, block_col=None):
    use_cols = components + process_factors + [response] + ([block_col] if block_col and block_col != "(None)" else [])
    d = df[use_cols].copy()
    for c in components + process_factors + [response]:
        d[c] = to_numeric(d[c])
    d = d.dropna(subset=components + process_factors + [response]).reset_index(drop=True)
    d.insert(0, "RunOrder", np.arange(1, len(d) + 1))
    totals = d[components].sum(axis=1)
    if (totals <= 0).any():
        raise ValueError("Mixture component rows must have positive totals.")
    total_ref = float(np.nanmedian(totals.to_numpy(dtype=float)))
    normalized = d[components].div(totals, axis=0)

    safe_components = [f"M{i+1}" for i in range(len(components))]
    safe_process = [f"P{i+1}" for i in range(len(process_factors))]
    rename_map = {orig: safe for orig, safe in zip(components, safe_components)}
    rename_map.update({orig: safe for orig, safe in zip(process_factors, safe_process)})
    inv_map = {safe: orig for orig, safe in rename_map.items()}
    safe_df = pd.DataFrame(index=d.index)
    rows = []
    for orig, safe in zip(components, safe_components):
        safe_df[safe] = normalized[orig].to_numpy(dtype=float)
        rows.append({
            "Factor": orig,
            "Code": safe,
            "Type": "Mixture component",
            "Min": float(normalized[orig].min()),
            "Center": float(normalized[orig].mean()),
            "High": float(normalized[orig].max()),
            "Range": float(normalized[orig].max() - normalized[orig].min()),
            "Input min": float(d[orig].min()),
            "Input max": float(d[orig].max()),
        })
    for orig, safe in zip(process_factors, safe_process):
        coded_vals, lo, hi, center = _code_factor_series(d[orig])
        safe_df[safe] = coded_vals
        rows.append({
            "Factor": orig,
            "Code": safe,
            "Type": "Process factor",
            "Min": float(lo),
            "Center": float(center),
            "High": float(hi),
            "Range": float(hi - lo),
            "Input min": float(lo),
            "Input max": float(hi),
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
    display_df["Mixture total"] = totals
    for orig in components:
        display_df[f"{orig} (fraction)"] = normalized[orig]
    factor_summary = pd.DataFrame(rows)
    note = "Mixture components were normalised row-wise to a total of 1.0, while process factors were coded from their observed low and high settings."
    return d, safe_df, safe_components, safe_process, rename_map, inv_map, factor_summary, include_block, display_df, total_ref, note



def _fit_mixture_workflow(df, components, response, model_type, block_col=None):
    d, safe_df, safe_components, rename_map, inv_map, factor_summary, include_block, display_df, total_ref, note = _prepare_mixture_dataframe(df, components, response, block_col=block_col)
    candidate_terms = _candidate_terms_mixture(safe_components, model_type)
    full_terms, stepwise_history, selected_terms_raw, _, _ = _backward_aic_stepwise_general(safe_df, candidate_terms, include_block=include_block, no_intercept=True)
    full_model, full_formula = _fit_model_general(safe_df, full_terms, include_block=include_block, no_intercept=True)
    selected_terms = _enforce_hierarchy(selected_terms_raw, candidate_terms)
    hierarchy_changed = [_normalize_term(t) for t in selected_terms] != [_normalize_term(t) for t in selected_terms_raw]
    selected_model, selected_formula = _fit_model_general(safe_df, selected_terms, include_block=include_block, no_intercept=True)
    lack_of_fit = _lack_of_fit_table(safe_df, selected_model, safe_components, include_block=include_block)
    model_comparison = _nested_model_comparison(selected_model, full_model) if _normalize_term(selected_formula) != _normalize_term(full_formula) else pd.DataFrame()

    def _pretty_model_rhs(rhs):
        parts = [p.strip() for p in str(rhs).split("+")]
        return " + ".join([_term_to_pretty(p, inv_map) for p in parts if p])

    result = {
        "family": DOE_FAMILY_MIXTURE,
        "data": d,
        "display_df": display_df,
        "safe_df": safe_df,
        "safe_factors": safe_components,
        "safe_components": safe_components,
        "safe_process": [],
        "mixture_components": list(components),
        "process_factors": [],
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
        "full_coef": _coef_table(full_model, inv_map),
        "selected_coef": _coef_table(selected_model, inv_map),
        "stepwise_history": stepwise_history.assign(
            Term=stepwise_history["Term"].map(lambda x: _term_to_pretty(x, inv_map) if x != "—" else x),
            Model=stepwise_history["Model"].map(_pretty_model_rhs),
        ),
        "full_fit_stats": _fit_stats_table(full_model),
        "selected_fit_stats": _fit_stats_table(selected_model, lack_of_fit=lack_of_fit),
        "selected_effects": _effects_anova_table(selected_model, inv_map),
        "selected_summary": _model_summary_anova(selected_model, lack_of_fit=lack_of_fit),
        "lack_of_fit": lack_of_fit,
        "model_comparison": model_comparison,
        "equation": _model_equation(selected_model, inv_map, response),
        "conclusions": _conclusion_lines(response, selected_terms, inv_map, selected_model, lack_of_fit),
        "response": response,
        "model_type": model_type,
        "block_col": block_col if include_block else None,
        "total_ref": total_ref,
        "normalization_note": note,
    }
    return result



def _fit_mixture_process_workflow(df, components, process_factors, response, mixture_model_type, process_model_type, block_col=None):
    d, safe_df, safe_components, safe_process, rename_map, inv_map, factor_summary, include_block, display_df, total_ref, note = _prepare_mixture_process_dataframe(df, components, process_factors, response, block_col=block_col)
    mix_terms = _candidate_terms_mixture(safe_components, mixture_model_type)
    proc_terms = _candidate_terms(safe_process, process_model_type) if safe_process else []
    cross_terms = [f"{m}:{p}" for m in safe_components for p in safe_process]
    candidate_terms = mix_terms + proc_terms + cross_terms
    full_terms, stepwise_history, selected_terms_raw, _, _ = _backward_aic_stepwise_general(safe_df, candidate_terms, include_block=include_block, no_intercept=True)
    full_model, full_formula = _fit_model_general(safe_df, full_terms, include_block=include_block, no_intercept=True)
    selected_terms = _enforce_hierarchy(selected_terms_raw, candidate_terms)
    hierarchy_changed = [_normalize_term(t) for t in selected_terms] != [_normalize_term(t) for t in selected_terms_raw]
    selected_model, selected_formula = _fit_model_general(safe_df, selected_terms, include_block=include_block, no_intercept=True)
    lack_of_fit = _lack_of_fit_table(safe_df, selected_model, safe_components + safe_process, include_block=include_block)
    model_comparison = _nested_model_comparison(selected_model, full_model) if _normalize_term(selected_formula) != _normalize_term(full_formula) else pd.DataFrame()

    def _pretty_model_rhs(rhs):
        parts = [p.strip() for p in str(rhs).split("+")]
        return " + ".join([_term_to_pretty(p, inv_map) for p in parts if p])

    result = {
        "family": DOE_FAMILY_MIXPROC,
        "data": d,
        "display_df": display_df,
        "safe_df": safe_df,
        "safe_factors": safe_components + safe_process,
        "safe_components": safe_components,
        "safe_process": safe_process,
        "mixture_components": list(components),
        "process_factors": list(process_factors),
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
        "full_coef": _coef_table(full_model, inv_map),
        "selected_coef": _coef_table(selected_model, inv_map),
        "stepwise_history": stepwise_history.assign(
            Term=stepwise_history["Term"].map(lambda x: _term_to_pretty(x, inv_map) if x != "—" else x),
            Model=stepwise_history["Model"].map(_pretty_model_rhs),
        ),
        "full_fit_stats": _fit_stats_table(full_model),
        "selected_fit_stats": _fit_stats_table(selected_model, lack_of_fit=lack_of_fit),
        "selected_effects": _effects_anova_table(selected_model, inv_map),
        "selected_summary": _model_summary_anova(selected_model, lack_of_fit=lack_of_fit),
        "lack_of_fit": lack_of_fit,
        "model_comparison": model_comparison,
        "equation": _model_equation(selected_model, inv_map, response),
        "conclusions": _conclusion_lines(response, selected_terms, inv_map, selected_model, lack_of_fit),
        "response": response,
        "model_type": f"Mixture {mixture_model_type}; Process {process_model_type}",
        "block_col": block_col if include_block else None,
        "total_ref": total_ref,
        "normalization_note": note,
        "mixture_model_type": mixture_model_type,
        "process_model_type": process_model_type,
    }
    return result



def _analysis_description(result):
    family = result.get("family", DOE_FAMILY_PROCESS)
    if family == DOE_FAMILY_PROCESS:
        return _legacy_analysis_description(result)
    if family == DOE_FAMILY_MIXTURE:
        comps = ", ".join(result.get("mixture_components", []))
        lines = [
            f"{len(result['data'])} experimental runs were analysed for the response {result['response']} using a mixture-design workflow.",
            f"The selected mixture components were {comps}. {result.get('normalization_note', '')}",
            f"The full model started from a {result['model_type']} Scheffé-style mixture model, followed by backward AIC reduction and hierarchy enforcement.",
        ]
        if result.get("block_col"):
            lines.append(f"Block was included in the model as a categorical term using column {result['block_col']}.")
        return " ".join(lines)
    comps = ", ".join(result.get("mixture_components", []))
    procs = ", ".join(result.get("process_factors", []))
    lines = [
        f"{len(result['data'])} experimental runs were analysed for the response {result['response']} using a mixture-process workflow.",
        f"Mixture components: {comps}. Process factors: {procs}. {result.get('normalization_note', '')}",
        f"The full model combined a {result.get('mixture_model_type', 'quadratic')} mixture model with a {result.get('process_model_type', 'quadratic')} process model plus mixture-by-process interaction terms.",
        "Backward AIC reduction was used, after which the hierarchy principle was enforced.",
    ]
    if result.get("block_col"):
        lines.append(f"Block was included in the model as a categorical term using column {result['block_col']}.")
    return " ".join(lines)



def _steps_text(family=DOE_FAMILY_PROCESS):
    if family == DOE_FAMILY_PROCESS:
        return _legacy_steps_text()
    if family == DOE_FAMILY_MIXTURE:
        return [
            "Fit the full mixture model to the selected response using normalised component proportions.",
            "Use stepwise backward elimination to identify a parsimonious model.",
            "Apply the hierarchy principle so retained blend terms keep their parent component terms.",
            "Review residual diagnostics, ANOVA statistics, coefficient tables, ternary or profile plots, and predicted-versus-observed agreement.",
            "Use constrained optimisation on the simplex to identify promising blend compositions.",
        ]
    return [
        "Fit the full mixture-process model using normalised mixture components and coded process factors.",
        "Use stepwise backward elimination to identify a parsimonious model.",
        "Apply the hierarchy principle so retained higher-order terms keep their parent component or process terms.",
        "Review residual diagnostics, ANOVA statistics, contour/profile plots, and predicted-versus-observed agreement.",
        "Use constrained optimisation across the simplex and process-factor space to identify promising operating conditions.",
    ]



def _predict_result(result, frame):
    pred_df = frame.copy()
    block_value = _prediction_block_value(result["selected_model"])
    if block_value is not None and "Block" not in pred_df.columns:
        pred_df["Block"] = block_value
    return result["selected_model"].predict(pred_df)



def _ternary_xy(a, b, c):
    x = b + 0.5 * c
    y = (np.sqrt(3.0) / 2.0) * c
    return x, y



def _simplex_grid(q=3, denom=25):
    pts = []
    for part in _integer_partitions(int(denom), q):
        pts.append(np.array(part, dtype=float) / float(denom))
    return np.array(pts, dtype=float)



def _make_ternary_contour_plot(result, fixed_process_actual=None, denom=28):
    cfg = common.safe_get_plot_cfg("DoE contour")
    comps = result.get("mixture_components", [])
    if len(comps) != 3:
        return None
    fig, ax = plt.subplots(figsize=(cfg["fig_w"], cfg["fig_h"]))
    pts = _simplex_grid(3, denom=denom)
    pred_df = pd.DataFrame(pts, columns=result["safe_components"])
    if result.get("safe_process"):
        fs = result["factor_summary"].set_index("Factor")
        base = {p: 0.0 for p in result["safe_process"]}
        if fixed_process_actual:
            for fac in result.get("process_factors", []):
                if fac in fixed_process_actual:
                    code = fs.loc[fac, "Code"]
                    center = float(fs.loc[fac, "Center"])
                    half = (float(fs.loc[fac, "High"]) - float(fs.loc[fac, "Min"])) / 2.0 if pd.notna(fs.loc[fac, "Min"]) else 0.0
                    if np.isclose(half, 0.0):
                        base[code] = 0.0
                    else:
                        base[code] = (float(fixed_process_actual[fac]) - center) / half
        for code, val in base.items():
            pred_df[code] = val
    zz = np.asarray(_predict_result(result, pred_df), dtype=float)
    x, y = _ternary_xy(pts[:, 0], pts[:, 1], pts[:, 2])
    tri = ax.tricontourf(x, y, zz, levels=20, cmap="viridis")
    ax.tricontour(x, y, zz, levels=8, colors="white", linewidths=0.5, alpha=0.6)
    obs_frac = result["display_df"][[f"{c} (fraction)" for c in comps]].to_numpy(dtype=float)
    ox, oy = _ternary_xy(obs_frac[:, 0], obs_frac[:, 1], obs_frac[:, 2])
    ax.scatter(ox, oy, c="white", edgecolor="black", s=max(24, cfg["marker_size"]), label="Observed runs")
    corners = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3.0) / 2.0], [0, 0]])
    ax.plot(corners[:, 0], corners[:, 1], color="black", lw=1.0)
    ax.text(-0.03, -0.03, comps[0], ha="right", va="top")
    ax.text(1.03, -0.03, comps[1], ha="left", va="top")
    ax.text(0.5, np.sqrt(3.0) / 2.0 + 0.03, comps[2], ha="center", va="bottom")
    ax.set_title(f"Ternary contour plot for {result['response']}")
    ax.set_aspect("equal")
    ax.axis("off")
    fig.colorbar(tri, ax=ax, label=result["response"])
    ax.legend(loc="best")
    return fig



def _make_component_profile_plot(result, component_name, fixed_mix=None, fixed_process_actual=None):
    cfg = common.safe_get_plot_cfg("DoE interaction")
    fig, ax = plt.subplots(figsize=(cfg["fig_w"], cfg["fig_h"]))
    comps = result.get("mixture_components", [])
    if component_name not in comps:
        return fig
    base = {c: 1.0 / len(comps) for c in comps}
    if fixed_mix:
        for c in comps:
            if c in fixed_mix:
                base[c] = float(fixed_mix[c])
        total = sum(base.values())
        if total > 0:
            base = {k: v / total for k, v in base.items()}
    vals = np.linspace(0.0, 1.0, 60)
    rows = []
    others = [c for c in comps if c != component_name]
    other_weights = np.array([base[c] for c in others], dtype=float)
    if other_weights.sum() <= 0:
        other_weights = np.ones(len(others), dtype=float) / max(1, len(others))
    else:
        other_weights = other_weights / other_weights.sum()
    safe_comp = {orig: code for orig, code in zip(result["mixture_components"], result["safe_components"])}
    fs = result["factor_summary"].set_index("Factor") if result.get("process_factors") else None
    process_code_map = {orig: code for orig, code in zip(result.get("process_factors", []), result.get("safe_process", []))}
    for v in vals:
        mix = {component_name: float(v)}
        remain = max(0.0, 1.0 - float(v))
        for c, w in zip(others, other_weights):
            mix[c] = remain * float(w)
        row = {safe_comp[c]: mix[c] for c in comps}
        for pfac, code in process_code_map.items():
            center = float(fs.loc[pfac, "Center"])
            half = (float(fs.loc[pfac, "High"]) - float(fs.loc[pfac, "Min"])) / 2.0 if pd.notna(fs.loc[pfac, "Min"]) else 0.0
            actual = center
            if fixed_process_actual and pfac in fixed_process_actual:
                actual = float(fixed_process_actual[pfac])
            row[code] = 0.0 if np.isclose(half, 0.0) else (actual - center) / half
        rows.append(row)
    pred_df = pd.DataFrame(rows)
    pred = np.asarray(_predict_result(result, pred_df), dtype=float)
    ax.plot(vals * result.get("total_ref", 100.0), pred, lw=cfg["line_width"], marker=cfg.get("marker_style", "o"), ms=max(2, int(cfg["marker_size"] * 0.25)))
    xlabel = f"{component_name} amount"
    apply_ax_style(ax, f"Profile plot for {component_name}", xlabel, result["response"], legend=False, plot_key="DoE interaction")
    return fig



def _make_process_profile_plot(result, process_factor, fixed_mix=None, fixed_process_actual=None):
    cfg = common.safe_get_plot_cfg("DoE interaction")
    fig, ax = plt.subplots(figsize=(cfg["fig_w"], cfg["fig_h"]))
    if process_factor not in result.get("process_factors", []):
        return fig
    fs = result["factor_summary"].set_index("Factor")
    safe_comp = {orig: code for orig, code in zip(result["mixture_components"], result["safe_components"])}
    process_code_map = {orig: code for orig, code in zip(result.get("process_factors", []), result.get("safe_process", []))}
    base_mix = {c: 1.0 / len(result["mixture_components"]) for c in result["mixture_components"]}
    if fixed_mix:
        total = sum(float(fixed_mix.get(c, base_mix[c])) for c in result["mixture_components"])
        if total > 0:
            base_mix = {c: float(fixed_mix.get(c, base_mix[c])) / total for c in result["mixture_components"]}
    vals = np.linspace(float(fs.loc[process_factor, "Min"]), float(fs.loc[process_factor, "High"]), 60)
    rows = []
    for v in vals:
        row = {safe_comp[c]: base_mix[c] for c in result["mixture_components"]}
        for pfac, code in process_code_map.items():
            center = float(fs.loc[pfac, "Center"])
            half = (float(fs.loc[pfac, "High"]) - float(fs.loc[pfac, "Min"])) / 2.0
            actual = v if pfac == process_factor else (float(fixed_process_actual.get(pfac, center)) if fixed_process_actual else center)
            row[code] = 0.0 if np.isclose(half, 0.0) else (actual - center) / half
        rows.append(row)
    pred_df = pd.DataFrame(rows)
    pred = np.asarray(_predict_result(result, pred_df), dtype=float)
    ax.plot(vals, pred, lw=cfg["line_width"], marker=cfg.get("marker_style", "o"), ms=max(2, int(cfg["marker_size"] * 0.25)))
    apply_ax_style(ax, f"Process profile plot for {process_factor}", process_factor, result["response"], legend=False, plot_key="DoE interaction")
    return fig



def _simplex_optimum_points(q, denom=18):
    return _simplex_grid(q=q, denom=denom)



def _grid_search_optimum_family(result, goal="Maximize"):
    family = result.get("family", DOE_FAMILY_PROCESS)
    if family == DOE_FAMILY_PROCESS:
        return _grid_search_optimum(result["selected_model"], result["safe_factors"], result["factor_summary"], goal=goal)
    if family == DOE_FAMILY_MIXTURE:
        pts = _simplex_optimum_points(len(result["safe_components"]), denom=24)
        pred_df = pd.DataFrame(pts, columns=result["safe_components"])
        pred = np.asarray(_predict_result(result, pred_df), dtype=float)
        idx = int(np.nanargmin(pred)) if goal == "Minimize" else int(np.nanargmax(pred))
        best = pts[idx]
        out = {c: float(best[i] * result.get("total_ref", 100.0)) for i, c in enumerate(result["mixture_components"])}
        out.update({f"{c} fraction": float(best[i]) for i, c in enumerate(result["mixture_components"])})
        out["Predicted response"] = float(pred[idx])
        out["Goal"] = goal
        return pd.DataFrame([out])
    # mixture-process
    mix_pts = _simplex_optimum_points(len(result["safe_components"]), denom=18)
    proc_n = len(result.get("safe_process", []))
    levels = np.linspace(-1.0, 1.0, 9 if proc_n <= 2 else 7)
    proc_grid = np.array(np.meshgrid(*([levels] * proc_n), indexing="ij")).reshape(proc_n, -1).T if proc_n else np.zeros((1, 0))
    mix_rep = np.repeat(mix_pts, len(proc_grid), axis=0)
    proc_rep = np.tile(proc_grid, (len(mix_pts), 1)) if proc_n else np.zeros((len(mix_pts), 0))
    pred_df = pd.DataFrame(mix_rep, columns=result["safe_components"])
    for i, code in enumerate(result.get("safe_process", [])):
        pred_df[code] = proc_rep[:, i]
    pred = np.asarray(_predict_result(result, pred_df), dtype=float)
    idx = int(np.nanargmin(pred)) if goal == "Minimize" else int(np.nanargmax(pred))
    best_mix = mix_rep[idx]
    out = {c: float(best_mix[i] * result.get("total_ref", 100.0)) for i, c in enumerate(result["mixture_components"])}
    out.update({f"{c} fraction": float(best_mix[i]) for i, c in enumerate(result["mixture_components"])})
    fs = result["factor_summary"].set_index("Factor")
    for i, pfac in enumerate(result.get("process_factors", [])):
        code = result["safe_process"][i]
        center = float(fs.loc[pfac, "Center"])
        half = (float(fs.loc[pfac, "High"]) - float(fs.loc[pfac, "Min"])) / 2.0
        out[pfac] = center + float(proc_rep[idx, i]) * half
    out["Predicted response"] = float(pred[idx])
    out["Goal"] = goal
    return pd.DataFrame([out])



def _make_generic_doe_pdf_report(result, figures, decimals=3):
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
    styles.add(common.ParagraphStyle(name="SmallBodyDOE2", parent=styles["BodyText"], fontSize=9, leading=12, alignment=common.TA_LEFT))
    story = [
        common.Paragraph("Design of Experiments Analysis Report", styles["Title"]),
        common.Spacer(1, 0.12 * common.cm),
        common.Paragraph(f"Module: <b>{result.get('family', 'DoE')}</b>", styles["Heading2"]),
        common.Spacer(1, 0.12 * common.cm),
        common.Paragraph("Data", styles["Heading2"]),
        common.Paragraph("The table below contains the experimental runs used for the selected DoE response analysis.", styles["SmallBodyDOE2"]),
    ]
    story.extend(common._pdf_table(result["display_df"], styles, "Experimental data", decimals=decimals, max_rows=60))
    story.extend([
        common.Paragraph("Experimental description", styles["Heading2"]),
        common.Paragraph(_analysis_description(result), styles["SmallBodyDOE2"]),
    ])
    story.extend(common._pdf_table(result["factor_summary"], styles, "Factor summary", decimals=decimals, max_rows=30))
    story.extend([
        common.Paragraph("Analysis of DoE data", styles["Heading2"]),
        common.Paragraph("The analysis followed the workflow below.", styles["SmallBodyDOE2"]),
    ])
    for i, step in enumerate(_steps_text(result.get("family", DOE_FAMILY_PROCESS)), start=1):
        story.append(common.Paragraph(f"{i}. {step}", styles["SmallBodyDOE2"]))
    story.append(common.Spacer(1, 0.1 * common.cm))
    story.append(common.Paragraph(f"Fit full model to the {result['response']} response", styles["Heading3"]))
    story.extend(common._pdf_table(result["full_coef"], styles, "Full model coefficients", decimals=decimals, max_rows=40))
    story.extend(common._pdf_table(result["full_fit_stats"], styles, "Full model fit statistics", decimals=decimals, max_rows=20))
    story.append(common.Paragraph("Stepwise regression", styles["Heading3"]))
    story.extend(common._pdf_table(result["stepwise_history"], styles, "Stepwise AIC reduction history", decimals=decimals, max_rows=40))
    if result["hierarchy_changed"]:
        story.append(common.Paragraph("The hierarchy principle modified the raw stepwise result by retaining parent terms for selected higher-order effects.", styles["SmallBodyDOE2"]))
    story.append(common.Paragraph(f"Analysis of selected model for {result['response']}", styles["Heading3"]))
    story.extend(common._pdf_table(result["selected_summary"], styles, "Selected model ANOVA summary", decimals=decimals, max_rows=20))
    story.extend(common._pdf_table(result["selected_effects"], styles, "Selected model effects ANOVA", decimals=decimals, max_rows=40))
    story.extend(common._pdf_table(result["selected_coef"], styles, "Selected model coefficients", decimals=decimals, max_rows=40))
    story.extend(common._pdf_table(result["selected_fit_stats"], styles, "Selected model fit statistics", decimals=decimals, max_rows=20))
    if not result["model_comparison"].empty:
        story.extend(common._pdf_table(result["model_comparison"], styles, "Comparison of selected and full models", decimals=decimals, max_rows=10))
    story.append(common.Paragraph("Final model equation", styles["Heading3"]))
    story.append(common.Paragraph(result["equation"], styles["SmallBodyDOE2"]))
    story.append(common.Spacer(1, 0.15 * common.cm))
    story.append(common.Paragraph("Conclusions", styles["Heading2"]))
    for line in result["conclusions"]:
        story.append(common.Paragraph(f"• {line}", styles["SmallBodyDOE2"]))
    if figures:
        story.extend([common.PageBreak(), common.Paragraph("Figures", styles["Heading2"])])
        for caption, fig_bytes in figures:
            story.extend([common.Paragraph(caption, styles["Heading3"]), common.Image(BytesIO(fig_bytes))])
            story[-1]._restrictSize(24.5 * common.cm, 13.3 * common.cm)
            story.append(common.Spacer(1, 0.25 * common.cm))
    doc.build(story)
    bio.seek(0)
    return bio.getvalue()



def _render_analysis_ui(df, family, factors, response, model_type, block_col, decimals, mixture_components=None, process_factors=None, mixture_model_type=None, process_model_type=None):
    if family == DOE_FAMILY_PROCESS:
        return _legacy_render_analysis_ui(df, factors, response, model_type, block_col, decimals)
    if family == DOE_FAMILY_MIXTURE:
        result = _fit_mixture_workflow(df, mixture_components, response, mixture_model_type or "quadratic", block_col=block_col)
    else:
        result = _fit_mixture_process_workflow(df, mixture_components, process_factors, response, mixture_model_type or "quadratic", process_model_type or "quadratic", block_col=block_col)

    st.markdown("### Data")
    report_table(result["display_df"], "Experimental data", decimals)

    st.markdown("### Experimental description")
    info_box(_analysis_description(result))
    report_table(result["factor_summary"], "Factor summary", decimals)

    st.markdown("### Analysis of DoE data")
    st.markdown("<br>".join([f"{i}. {step}" for i, step in enumerate(_steps_text(result['family']), start=1)]), unsafe_allow_html=True)

    st.markdown(f"#### Fit full model to the {response} response")
    report_table(result["full_coef"], "Full model coefficients", decimals)
    report_table(result["full_fit_stats"], "Full model fit statistics", decimals)

    st.markdown("#### Stepwise regression")
    report_table(result["stepwise_history"], "Backward AIC reduction history", decimals)
    if result["hierarchy_changed"]:
        st.caption("Hierarchy note: the final selected model was adjusted to retain parent terms for selected higher-order effects.")

    st.markdown(f"#### Analysis of selected model for {response}")
    report_table(result["selected_summary"], "Selected model ANOVA summary", decimals)
    report_table(result["selected_effects"], "Selected model effects ANOVA", decimals)
    report_table(result["selected_coef"], "Selected model coefficients", decimals)
    report_table(result["selected_fit_stats"], "Selected model fit statistics", decimals)
    if not result["model_comparison"].empty:
        report_table(result["model_comparison"], "Comparison of selected and full models", decimals)

    st.markdown("#### Final selected model")
    st.code(result["equation"])

    st.markdown("### Model Graphs")
    figures = []
    if len(result.get("mixture_components", [])) == 3:
        fixed_process_actual = {}
        if result.get("process_factors"):
            st.markdown("**Fixed levels for process factors in ternary plot**")
            cols = st.columns(len(result["process_factors"]))
            fs = result["factor_summary"].set_index("Factor")
            for i, fac in enumerate(result["process_factors"]):
                fixed_process_actual[fac] = cols[i].slider(
                    fac,
                    min_value=float(fs.loc[fac, "Min"]),
                    max_value=float(fs.loc[fac, "High"]),
                    value=float(fs.loc[fac, "Center"]),
                    key=f"mixproc_fix_{response}_{fac}",
                )
        ternary_fig = _make_ternary_contour_plot(result, fixed_process_actual=fixed_process_actual if fixed_process_actual else None)
        if ternary_fig is not None:
            st.pyplot(ternary_fig)
            figures.append(("Ternary contour plot", fig_to_png_bytes(ternary_fig)))
    comp_choice = st.selectbox("Mixture component profile", result["mixture_components"], key=f"comp_profile_{response}")
    comp_fig = _make_component_profile_plot(result, comp_choice)
    st.pyplot(comp_fig)
    figures.append((f"Component profile plot - {comp_choice}", fig_to_png_bytes(comp_fig)))

    if result.get("process_factors"):
        proc_choice = st.selectbox("Process factor profile", result["process_factors"], key=f"proc_profile_{response}")
        proc_fig = _make_process_profile_plot(result, proc_choice)
        st.pyplot(proc_fig)
        figures.append((f"Process profile plot - {proc_choice}", fig_to_png_bytes(proc_fig)))

    diag_fig = _make_residual_diagnostics(result["selected_model"])
    pred_obs_fig = _make_predicted_vs_observed(result["selected_model"], response)
    st.pyplot(diag_fig)
    st.pyplot(pred_obs_fig)
    figures.extend([
        ("Residual diagnostics", fig_to_png_bytes(diag_fig)),
        ("Observed vs predicted", fig_to_png_bytes(pred_obs_fig)),
    ])

    st.markdown("### Conclusions from the analysis")
    for line in result["conclusions"]:
        st.markdown(f"- {line}")

    goal = st.selectbox("Optimization goal for selected response", ["Minimize", "Maximize"], key=f"goal_{response}")
    optimum = _grid_search_optimum_family(result, goal=goal)
    report_table(optimum, f"Constrained optimum for {response}", decimals)

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
    pdf_bytes = _make_generic_doe_pdf_report(result, figures, decimals=decimals)
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



def render():
    render_display_settings()
    st.sidebar.title("🧪 DoE Studio")
    st.sidebar.markdown("Design of Experiments")

    app_header("🧪 DoE Studio", "Design builder and DOE analysis for process, mixture, and mixture-process studies.")
    tabs = st.tabs(["Design Builder", "Analyze Responses"])

    with tabs[0]:
        st.subheader("Design Builder")
        info_box("Build process, mixture, or mixture-process designs. The new options are intended for pharma work such as process optimization, blend studies, and cosolvent/solubility screening.")
        design_family = st.selectbox(
            "Design family",
            [
                "Process factorial",
                "Process CCD",
                "Mixture simplex-centroid",
                "Mixture simplex-lattice",
                "Mixture-Process",
                "Co-Solvents Evaluation",
                "Co-Solvents Evaluation - Process",
            ],
            key="doe_design_family",
        )
        sample_cols = st.columns(5)
        sample_cols[0].button("Process sample", key="sample_doe_design_proc2", on_click=_load_sample_design)
        sample_cols[1].button("Mixture sample", key="sample_doe_design_mix2", on_click=_load_sample_mixture_design)
        sample_cols[2].button("Mixture-process sample", key="sample_doe_design_mixproc2", on_click=_load_sample_mixproc_design)
        sample_cols[3].button("Co-Solvents Evaluation", key="sample_doe_design_cos2", on_click=_load_sample_cosolvent_design)
        sample_cols[4].button("Co-Solvents Evaluation - Process", key="sample_doe_design_cosp2", on_click=_load_sample_cosolvent_process_design)

        blocks = st.number_input("Blocks", min_value=1, max_value=10, value=int(st.session_state.get("doe_blocks", 1)), step=1, key="doe_blocks")
        replicates = st.number_input("Replicates", min_value=1, max_value=10, value=int(st.session_state.get("doe_replicates", 1)), step=1, key="doe_replicates")
        randomize = st.checkbox("Randomize within block", value=bool(st.session_state.get("doe_randomize", True)), key="doe_randomize")
        seed = st.number_input("Random seed", min_value=1, max_value=999999, value=int(st.session_state.get("doe_seed", 123)), step=1, key="doe_seed")

        design = None
        if design_family in ["Process factorial", "Process CCD"]:
            c1, c2 = st.columns(2)
            with c1:
                n_factors = st.number_input("Number of process factors", min_value=2, max_value=8, value=int(st.session_state.get("doe_n_factors", 3)), step=1, key="doe_n_factors")
            with c2:
                center_points = st.number_input("Center points per block", min_value=0, max_value=20, value=int(st.session_state.get("doe_center_points", 2 if design_family == "Process CCD" else 0)), step=1, key="doe_center_points")
            st.markdown("### Process factor definitions")
            factor_names, lows, highs = [], [], []
            for i in range(int(n_factors)):
                cols = st.columns([1.3, 1, 1])
                factor_names.append(cols[0].text_input(f"Process factor {i+1} name", value=st.session_state.get(f"doe_name_{i}", f"Factor {i+1}"), key=f"doe_name_{i}"))
                lows.append(cols[1].number_input(f"Low level {i+1}", value=float(st.session_state.get(f"doe_low_{i}", 0.0)), key=f"doe_low_{i}"))
                highs.append(cols[2].number_input(f"High level {i+1}", value=float(st.session_state.get(f"doe_high_{i}", 1.0)), key=f"doe_high_{i}"))
            if st.button("Generate design", type="primary", key="gen_design_extended_process"):
                if design_family == "Process factorial":
                    design = _build_factorial_design(factor_names, lows, highs, blocks=int(blocks), center_points=int(center_points), replicates=int(replicates), randomize=randomize, seed=int(seed))
                else:
                    design = _build_ccd_design(factor_names, lows, highs, blocks=int(blocks), center_points=int(center_points), replicates=int(replicates), randomize=randomize, seed=int(seed))
                st.session_state["doe_generated_design"] = design
        elif design_family in ["Mixture simplex-centroid", "Mixture simplex-lattice"]:
            c1, c2 = st.columns(2)
            with c1:
                n_components = st.number_input("Number of mixture components", min_value=2, max_value=6, value=int(st.session_state.get("doe_mix_n_components", 3)), step=1, key="doe_mix_n_components")
            with c2:
                total = st.number_input("Mixture total", min_value=0.01, value=float(st.session_state.get("doe_mix_total", 100.0)), key="doe_mix_total")
            degree = 2
            if design_family == "Mixture simplex-lattice":
                degree = st.selectbox("Lattice degree", [2, 3], index=0, key="doe_mix_degree")
            st.markdown("### Mixture component definitions")
            component_names = []
            for i in range(int(n_components)):
                component_names.append(st.text_input(f"Component {i+1} name", value=st.session_state.get(f"doe_mix_name_{i}", f"Component {i+1}"), key=f"doe_mix_name_{i}"))
            if st.button("Generate design", type="primary", key="gen_design_extended_mix"):
                design = _build_mixture_design(component_names, total=float(total), design_kind="simplex-centroid" if design_family == "Mixture simplex-centroid" else "simplex-lattice", degree=int(degree), blocks=int(blocks), replicates=int(replicates), randomize=randomize, seed=int(seed))
                st.session_state["doe_generated_design"] = design
        elif design_family == "Co-Solvents Evaluation":
            defaults = _default_cosolvent_settings()
            c1, c2, c3 = st.columns(3)
            with c1:
                n_cosolvents = st.number_input("Number of co-solvents", min_value=1, max_value=5, value=int(st.session_state.get("doe_cs_n_cosolvents", 3)), step=1, key="doe_cs_n_cosolvents")
            with c2:
                total = st.number_input("Final fill volume", min_value=0.01, value=float(st.session_state.get("doe_cs_total", defaults["total"])), key="doe_cs_total")
            with c3:
                water_name = st.text_input("Water / q.s. component name", value=st.session_state.get("doe_cs_water_name", defaults["water_name"]), key="doe_cs_water_name")
            info_box("Enter the allowable low and high amount for each co-solvent per final fill. Water is calculated automatically as q.s. to the final fill volume.")
            st.markdown("### Co-solvent limits")
            cos_names, cos_lows, cos_highs = [], [], []
            default_names = defaults["cosolvent_names"]
            default_lows = defaults["cosolvent_lows"]
            default_highs = defaults["cosolvent_highs"]
            for i in range(int(n_cosolvents)):
                cols = st.columns([1.5, 1, 1])
                cos_names.append(cols[0].text_input(f"Co-solvent {i+1} name", value=st.session_state.get(f"doe_cs_name_{i}", default_names[i] if i < len(default_names) else f"Co-solvent {i+1}"), key=f"doe_cs_name_{i}"))
                cos_lows.append(cols[1].number_input(f"Low limit {i+1}", min_value=0.0, value=float(st.session_state.get(f"doe_cs_low_{i}", default_lows[i] if i < len(default_lows) else 0.0)), key=f"doe_cs_low_{i}"))
                cos_highs.append(cols[2].number_input(f"High limit {i+1}", min_value=0.0, value=float(st.session_state.get(f"doe_cs_high_{i}", default_highs[i] if i < len(default_highs) else 0.1)), key=f"doe_cs_high_{i}"))
            if st.button("Generate design", type="primary", key="gen_design_extended_cosolv"):
                design = _build_cosolvent_design(cos_names, cos_lows, cos_highs, water_name=water_name, total=float(total), blocks=int(blocks), replicates=int(replicates), randomize=randomize, seed=int(seed))
                st.session_state["doe_generated_design"] = design
        elif design_family == "Co-Solvents Evaluation - Process":
            defaults = _default_cosolvent_settings()
            c1, c2, c3 = st.columns(3)
            with c1:
                n_cosolvents = st.number_input("Number of co-solvents", min_value=1, max_value=5, value=int(st.session_state.get("doe_cs_n_cosolvents", 3)), step=1, key="doe_cs_n_cosolvents")
            with c2:
                total = st.number_input("Final fill volume", min_value=0.01, value=float(st.session_state.get("doe_cs_total", defaults["total"])), key="doe_cs_total")
            with c3:
                n_process = st.number_input("Number of process factors", min_value=1, max_value=4, value=int(st.session_state.get("doe_mp_n_process", 2)), step=1, key="doe_mp_n_process")
            c1, c2 = st.columns(2)
            with c1:
                water_name = st.text_input("Water / q.s. component name", value=st.session_state.get("doe_cs_water_name", defaults["water_name"]), key="doe_cs_water_name")
            with c2:
                process_design_kind = st.selectbox("Process sub-design", ["factorial", "ccd"], index=0 if st.session_state.get("doe_cs_proc_kind", "factorial") == "factorial" else 1, key="doe_cs_proc_kind")
            info_box("Co-solvent limits define the constrained blend space. Water is calculated automatically as q.s. to the final fill volume, and the resulting blend is crossed with the selected process-factor design.")
            st.markdown("### Co-solvent limits")
            cos_names, cos_lows, cos_highs = [], [], []
            default_names = defaults["cosolvent_names"]
            default_lows = defaults["cosolvent_lows"]
            default_highs = defaults["cosolvent_highs"]
            for i in range(int(n_cosolvents)):
                cols = st.columns([1.5, 1, 1])
                cos_names.append(cols[0].text_input(f"Co-solvent {i+1} name", value=st.session_state.get(f"doe_cs_name_{i}", default_names[i] if i < len(default_names) else f"Co-solvent {i+1}"), key=f"doe_cs_name_{i}"))
                cos_lows.append(cols[1].number_input(f"Low limit {i+1}", min_value=0.0, value=float(st.session_state.get(f"doe_cs_low_{i}", default_lows[i] if i < len(default_lows) else 0.0)), key=f"doe_cs_low_{i}"))
                cos_highs.append(cols[2].number_input(f"High limit {i+1}", min_value=0.0, value=float(st.session_state.get(f"doe_cs_high_{i}", default_highs[i] if i < len(default_highs) else 0.1)), key=f"doe_cs_high_{i}"))
            st.markdown("### Process factor definitions")
            proc_names, proc_lows, proc_highs = [], [], []
            default_proc_names = ["pH", "Temperature"]
            default_proc_lows = [3.5, 25.0]
            default_proc_highs = [6.5, 40.0]
            for i in range(int(n_process)):
                cols = st.columns([1.3, 1, 1])
                proc_names.append(cols[0].text_input(f"Process factor {i+1} name", value=st.session_state.get(f"doe_cs_proc_name_{i}", default_proc_names[i] if i < len(default_proc_names) else f"Process {i+1}"), key=f"doe_cs_proc_name_{i}"))
                proc_lows.append(cols[1].number_input(f"Low level {i+1}", value=float(st.session_state.get(f"doe_cs_proc_low_{i}", default_proc_lows[i] if i < len(default_proc_lows) else 0.0)), key=f"doe_cs_proc_low_{i}"))
                proc_highs.append(cols[2].number_input(f"High level {i+1}", value=float(st.session_state.get(f"doe_cs_proc_high_{i}", default_proc_highs[i] if i < len(default_proc_highs) else 1.0)), key=f"doe_cs_proc_high_{i}"))
            if st.button("Generate design", type="primary", key="gen_design_extended_cosolv_proc"):
                design = _build_cosolvent_process_design(cos_names, cos_lows, cos_highs, proc_names, proc_lows, proc_highs, water_name=water_name, total=float(total), process_design_kind=process_design_kind, blocks=int(blocks), replicates=int(replicates), randomize=randomize, seed=int(seed))
                st.session_state["doe_generated_design"] = design
        else:
            c1, c2, c3 = st.columns(3)
            with c1:
                n_components = st.number_input("Number of mixture components", min_value=2, max_value=5, value=int(st.session_state.get("doe_mix_n_components", 3)), step=1, key="doe_mix_n_components")
            with c2:
                total = st.number_input("Mixture total", min_value=0.01, value=float(st.session_state.get("doe_mix_total", 100.0)), key="doe_mix_total")
            with c3:
                n_process = st.number_input("Number of process factors", min_value=1, max_value=4, value=int(st.session_state.get("doe_mp_n_process", 2)), step=1, key="doe_mp_n_process")
            c1, c2 = st.columns(2)
            with c1:
                mixture_design_kind = st.selectbox("Mixture sub-design", ["simplex-centroid", "simplex-lattice"], index=0, key="doe_mp_mix_kind")
            with c2:
                process_design_kind = st.selectbox("Process sub-design", ["factorial", "ccd"], index=0, key="doe_mp_proc_kind")
            st.markdown("### Mixture components")
            component_names = []
            for i in range(int(n_components)):
                component_names.append(st.text_input(f"Component {i+1} name", value=st.session_state.get(f"doe_mix_name_{i}", f"Component {i+1}"), key=f"doe_mix_name_{i}"))
            st.markdown("### Process factor definitions")
            proc_names, proc_lows, proc_highs = [], [], []
            for i in range(int(n_process)):
                cols = st.columns([1.3, 1, 1])
                proc_names.append(cols[0].text_input(f"Process factor {i+1} name", value=st.session_state.get(f"doe_mp_name_{i}", f"Process {i+1}"), key=f"doe_mp_name_{i}"))
                proc_lows.append(cols[1].number_input(f"Low level {i+1}", value=float(st.session_state.get(f"doe_mp_low_{i}", 0.0)), key=f"doe_mp_low_{i}"))
                proc_highs.append(cols[2].number_input(f"High level {i+1}", value=float(st.session_state.get(f"doe_mp_high_{i}", 1.0)), key=f"doe_mp_high_{i}"))
            if st.button("Generate design", type="primary", key="gen_design_extended_mixproc"):
                design = _build_mixture_process_design(component_names, proc_names, proc_lows, proc_highs, total=float(total), mixture_design_kind=mixture_design_kind, process_design_kind=process_design_kind, blocks=int(blocks), replicates=int(replicates), randomize=randomize, seed=int(seed))
                st.session_state["doe_generated_design"] = design

        if "doe_generated_design" in st.session_state:
            design = st.session_state["doe_generated_design"]
            st.success(f"Generated design with {len(design)} runs")
            st.dataframe(design, width="stretch")
            excel_bytes = make_excel_bytes({"Design": design})
            st.download_button("Download design workbook", excel_bytes, file_name="doe_design.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            st.download_button("Download design CSV", design.to_csv(index=False).encode("utf-8"), file_name="doe_design.csv", mime="text/csv")
            if design_family in ["Co-Solvents Evaluation", "Co-Solvents Evaluation - Process"]:
                info_box("For co-solvent designs, keep the co-solvent amount columns and the auto-calculated water column in the final dataset. In analysis, treat them as mixture components, with any extra numeric settings such as pH or temperature entered as process factors.")
            else:
                info_box("After collecting experimental responses, switch to the Analyze Responses tab and paste the completed design table there.")

    with tabs[1]:
        st.subheader("Response Analysis")
        family = st.selectbox("DoE family", [DOE_FAMILY_PROCESS, DOE_FAMILY_MIXTURE, DOE_FAMILY_MIXPROC], key="doe_analysis_family")
        family_help = {
            DOE_FAMILY_PROCESS: "Use this for independent numeric process factors such as temperature, time, pH, agitation, spray rate, or pressure.",
            DOE_FAMILY_MIXTURE: "Use this when the selected component columns are proportions of a blend that must sum to a constant, such as solvent blends or excipient ratios.",
            DOE_FAMILY_MIXPROC: "Use this when you have both a constrained blend and independent process factors, such as cosolvent ratios plus temperature or pH.",
        }
        info_box(family_help[family])

        c_sample, c_text = st.columns([2.2, 5])
        with c_sample:
            if family == DOE_FAMILY_PROCESS:
                st.button("Process sample data", key="sample_doe_response_proc2", on_click=_load_sample_response_text)
            elif family == DOE_FAMILY_MIXTURE:
                st.button("Mixture sample data", key="sample_doe_response_mix2", on_click=_load_sample_response_text_mixture)
                st.button("Co-Solvents Evaluation", key="sample_doe_response_cos2", on_click=_load_sample_response_text_cosolvent)
            else:
                st.button("Mixture-process sample data", key="sample_doe_response_mixproc2", on_click=_load_sample_response_text_mixproc)
                st.button("Co-Solvents Evaluation - Process", key="sample_doe_response_cosp2", on_click=_load_sample_response_text_cosolvent_process)
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

                if family == DOE_FAMILY_PROCESS:
                    c1, c2, c3, c4 = st.columns([1.45, 1.1, 1.15, 1.2])
                    with c1:
                        factors = st.multiselect("Numeric process factors", num_cols, default=num_cols[: min(2, len(num_cols))], key="doe_factors")
                    with c2:
                        candidate_responses = [c for c in num_cols if c not in factors] or num_cols
                        response = st.selectbox("Response", candidate_responses, key="doe_response")
                    with c3:
                        model_type = st.selectbox("Model type", ["linear", "interaction", "quadratic"], index=2, key="doe_model_type")
                    with c4:
                        block_col = st.selectbox("Block column (optional)", ["(None)"] + [c for c in all_cols if c not in factors + [response]], key="doe_block")
                    if len(factors) < 2:
                        st.info("Select at least two numeric process factors to analyze the response surface.")
                    else:
                        _render_analysis_ui(df, family, factors, response, model_type, block_col, decimals)
                elif family == DOE_FAMILY_MIXTURE:
                    c1, c2, c3, c4 = st.columns([1.6, 1.1, 1.15, 1.2])
                    with c1:
                        mixture_components = st.multiselect("Mixture component columns", num_cols, default=num_cols[: min(3, len(num_cols))], key="doe_mix_components")
                    with c2:
                        candidate_responses = [c for c in num_cols if c not in mixture_components] or num_cols
                        response = st.selectbox("Response", candidate_responses, key="doe_mix_response")
                    with c3:
                        mixture_model_type = st.selectbox("Mixture model", ["linear", "quadratic", "special_cubic"], index=1, key="doe_mix_model_type")
                    with c4:
                        block_col = st.selectbox("Block column (optional)", ["(None)"] + [c for c in all_cols if c not in mixture_components + [response]], key="doe_mix_block")
                    if len(mixture_components) < 2:
                        st.info("Select at least two mixture components.")
                    else:
                        _render_analysis_ui(df, family, mixture_components, response, None, block_col, decimals, mixture_components=mixture_components, mixture_model_type=mixture_model_type)
                else:
                    c1, c2, c3, c4, c5 = st.columns([1.45, 1.25, 1.05, 1.05, 1.2])
                    with c1:
                        mixture_components = st.multiselect("Mixture component columns", num_cols, default=num_cols[: min(3, len(num_cols))], key="doe_mp_components")
                    remaining_numeric = [c for c in num_cols if c not in mixture_components]
                    with c2:
                        process_factors = st.multiselect("Process factor columns", remaining_numeric, default=remaining_numeric[: min(2, len(remaining_numeric))], key="doe_mp_process")
                    candidate_responses = [c for c in num_cols if c not in mixture_components + process_factors] or num_cols
                    with c3:
                        response = st.selectbox("Response", candidate_responses, key="doe_mp_response")
                    with c4:
                        mixture_model_type = st.selectbox("Mixture model", ["linear", "quadratic", "special_cubic"], index=1, key="doe_mp_mix_model")
                    with c5:
                        process_model_type = st.selectbox("Process model", ["linear", "interaction", "quadratic"], index=2, key="doe_mp_proc_model")
                    block_col = st.selectbox("Block column (optional)", ["(None)"] + [c for c in all_cols if c not in mixture_components + process_factors + [response]], key="doe_mp_block")
                    if len(mixture_components) < 2 or len(process_factors) < 1:
                        st.info("Select at least two mixture components and one process factor.")
                    else:
                        _render_analysis_ui(df, family, mixture_components + process_factors, response, None, block_col, decimals, mixture_components=mixture_components, process_factors=process_factors, mixture_model_type=mixture_model_type, process_model_type=process_model_type)
            except Exception as e:
                st.error(str(e))
