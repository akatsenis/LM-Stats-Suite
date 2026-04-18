import modules.common as common
from modules.common import *
from scipy.optimize import least_squares
from scipy.stats import gaussian_kde
from io import StringIO

st = common.st
pd = common.pd
np = common.np
plt = common.plt


WEIBULL_SAMPLE_DATA = """Time\tDissolution
0\t0
24\t4.062280072
48\t5.406121389
72\t7.927264033
96\t11.3652943
168\t38.3658273
216\t54.87508027
264\t66.8571673
336\t75.1739213
432\t87.68825396
528\t88.68380822
600\t92.799906
696\t92.9
768\t93.29317879
"""

TIME_UNIT_TO_HOURS = {
    "Minutes": 1 / 60.0,
    "Hours": 1.0,
    "Days": 24.0,
}


def load_dual_sample_text(state_key_a, sample_key_a, state_key_b, sample_key_b):
    from modules.stats_suite import SAMPLE_DATA
    st.session_state[state_key_a] = SAMPLE_DATA[sample_key_a]
    st.session_state[state_key_b] = SAMPLE_DATA[sample_key_b]


def load_weibull_sample_text(state_key):
    st.session_state[state_key] = WEIBULL_SAMPLE_DATA


def _coerce_numeric_df(text):
    df = pd.read_csv(StringIO(text.strip()), sep=r"[\t,; ]+", engine="python")
    if df.shape[1] < 2:
        raise ValueError("Please provide at least two columns: time and one dissolution column.")
    df = df.dropna(how="all")
    df.columns = [str(c).strip() if str(c).strip() else f"Column_{i+1}" for i, c in enumerate(df.columns)]
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=[df.columns[0]])
    return df


def parse_dissolution_weibull_table(text):
    df = _coerce_numeric_df(text)
    time_col = df.columns[0]
    rep_cols = [c for c in df.columns[1:] if df[c].notna().any()]
    if not rep_cols:
        raise ValueError("At least one dissolution response column is required.")
    out = df[[time_col] + rep_cols].copy()
    out = out.rename(columns={time_col: "Time_input"})
    out = out.sort_values("Time_input").reset_index(drop=True)
    return out


def weibull_single(t, Fmax, MDT1, b1):
    t = np.asarray(t, dtype=float)
    return Fmax * (1.0 - np.exp(-np.power(np.clip(t, 0, None) / np.maximum(MDT1, 1e-12), np.maximum(b1, 1e-12))))


def weibull_double(t, Fmax, f1, MDT1, b1, MDT2, b2):
    t = np.asarray(t, dtype=float)
    tt = np.clip(t, 0, None)
    return Fmax * (
        1.0
        - f1 * np.exp(-np.power(tt / np.maximum(MDT1, 1e-12), np.maximum(b1, 1e-12)))
        - (1.0 - f1) * np.exp(-np.power(tt / np.maximum(MDT2, 1e-12), np.maximum(b2, 1e-12)))
    )


def weibull_triple(t, Fmax, f1, f2, MDT1, b1, MDT2, b2, MDT3, b3):
    t = np.asarray(t, dtype=float)
    tt = np.clip(t, 0, None)
    f3 = np.clip(1.0 - f1 - f2, 0.0, 1.0)
    return Fmax * (
        1.0
        - f1 * np.exp(-np.power(tt / np.maximum(MDT1, 1e-12), np.maximum(b1, 1e-12)))
        - f2 * np.exp(-np.power(tt / np.maximum(MDT2, 1e-12), np.maximum(b2, 1e-12)))
        - f3 * np.exp(-np.power(tt / np.maximum(MDT3, 1e-12), np.maximum(b3, 1e-12)))
    )


MODEL_SPECS = {
    "Single Weibull": {
        "func": weibull_single,
        "param_names": ["Fmax", "MDT1_h", "b1"],
        "display_names": ["Fmax", "MDT1", "β1"],
    },
    "Double Weibull": {
        "func": weibull_double,
        "param_names": ["Fmax", "f1", "MDT1_h", "b1", "MDT2_h", "b2"],
        "display_names": ["Fmax", "f1", "MDT1", "β1", "MDT2", "β2"],
    },
    "Triple Weibull": {
        "func": weibull_triple,
        "param_names": ["Fmax", "f1", "f2", "MDT1_h", "b1", "MDT2_h", "b2", "MDT3_h", "b3"],
        "display_names": ["Fmax", "f1", "f2", "MDT1", "β1", "MDT2", "β2", "MDT3", "β3"],
    },
}


def _default_bounds_and_start(model_name, t_h, y):
    t_h = np.asarray(t_h, dtype=float)
    y = np.asarray(y, dtype=float)
    t_pos = t_h[(t_h > 0) & np.isfinite(t_h)]
    tmax = float(np.nanmax(t_h)) if len(t_h) else 1.0
    tmin_pos = float(np.nanmin(t_pos)) if len(t_pos) else max(tmax / 100.0, 1e-6)
    ymax = float(np.nanmax(y)) if np.isfinite(np.nanmax(y)) else 100.0
    F_guess = min(100.0, max(1.0, ymax * 1.02))

    if model_name == "Single Weibull":
        p0 = [F_guess, max(tmax * 0.35, tmin_pos), 1.2]
        lb = [0.0, max(tmin_pos * 0.01, 1e-6), 1e-6]
        ub = [100.0, max(tmax * 10.0, 1.0), 50.0]
    elif model_name == "Double Weibull":
        p0 = [F_guess, 0.55, max(tmax * 0.20, tmin_pos), 0.8, max(tmax * 0.75, tmin_pos * 1.5), 1.5]
        lb = [0.0, 0.0, max(tmin_pos * 0.01, 1e-6), 1e-6, max(tmin_pos * 0.01, 1e-6), 1e-6]
        ub = [100.0, 1.0, max(tmax * 10.0, 1.0), 50.0, max(tmax * 10.0, 1.0), 50.0]
    else:
        p0 = [F_guess, 0.30, 0.30, max(tmax * 0.12, tmin_pos), 0.8, max(tmax * 0.45, tmin_pos * 1.5), 1.4, max(tmax * 0.95, tmin_pos * 2), 2.0]
        lb = [0.0, 0.0, 0.0, max(tmin_pos * 0.01, 1e-6), 1e-6, max(tmin_pos * 0.01, 1e-6), 1e-6, max(tmin_pos * 0.01, 1e-6), 1e-6]
        ub = [100.0, 1.0, 1.0, max(tmax * 10.0, 1.0), 50.0, max(tmax * 10.0, 1.0), 50.0, max(tmax * 10.0, 1.0), 50.0]
    return np.asarray(p0, dtype=float), np.asarray(lb, dtype=float), np.asarray(ub, dtype=float)


def build_weibull_parameter_tables(t_h, y):
    tables = {}
    for model_name, spec in MODEL_SPECS.items():
        p0, lb, ub = _default_bounds_and_start(model_name, t_h, y)
        df = pd.DataFrame({
            "Parameter": spec["display_names"],
            "Initial Value": p0,
            "Min (≥)": lb,
            "Max (≤)": ub,
        })
        tables[model_name] = df
    return tables


def _slugify(s):
    return ''.join(ch.lower() if ch.isalnum() else '_' for ch in str(s)).strip('_')


def _sanitize_editor_table(model_name, editor_df):
    spec = MODEL_SPECS[model_name]
    param_names = spec["param_names"]
    display_names = spec["display_names"]
    if editor_df is None or len(editor_df) != len(param_names):
        raise ValueError(f"Parameter table for {model_name} is incomplete.")
    df = editor_df.copy().reset_index(drop=True)
    if "Parameter" not in df.columns:
        df.insert(0, "Parameter", display_names)
    df["Parameter"] = display_names
    for col in ["Initial Value", "Min (≥)", "Max (≤)"]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' is required in the parameter table for {model_name}.")
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if df[["Initial Value", "Min (≥)", "Max (≤)"]].isna().any().any():
        raise ValueError(f"All initial, minimum, and maximum values must be numeric for {model_name}.")

    init = df["Initial Value"].to_numpy(dtype=float)
    lb = df["Min (≥)"].to_numpy(dtype=float)
    ub = df["Max (≤)"].to_numpy(dtype=float)

    for i, pname in enumerate(param_names):
        if pname == "Fmax":
            lb[i] = max(lb[i], 0.0)
            ub[i] = min(ub[i], 100.0)
        elif pname in {"f1", "f2"}:
            lb[i] = max(lb[i], 0.0)
            ub[i] = min(ub[i], 1.0)
        elif pname.startswith("MDT"):
            lb[i] = max(lb[i], 1e-12)
        elif pname.startswith("b"):
            lb[i] = max(lb[i], 1e-12)
        if ub[i] <= lb[i]:
            raise ValueError(f"For {model_name}, max must be greater than min for {display_names[i]}.")
        init[i] = float(np.clip(init[i], lb[i], ub[i]))

    if model_name == "Triple Weibull":
        i1 = param_names.index("f1")
        i2 = param_names.index("f2")
        if lb[i1] + lb[i2] > 1.0 + 1e-12:
            raise ValueError("For Triple Weibull, the minimum bounds for f1 and f2 are incompatible because Min(f1) + Min(f2) must be ≤ 1.")
        if init[i1] + init[i2] > 1.0:
            total = init[i1] + init[i2]
            init[i1] = init[i1] / total * 0.98
            init[i2] = init[i2] / total * 0.98
    return init, lb, ub


def _resolve_fit_settings(model_name, t_h, y, parameter_tables=None):
    if parameter_tables and model_name in parameter_tables:
        return _sanitize_editor_table(model_name, parameter_tables[model_name])
    return _default_bounds_and_start(model_name, t_h, y)


def _enforce_weight_constraints(model_name, params):
    p = np.asarray(params, dtype=float).copy()
    if "Fmax" in MODEL_SPECS[model_name]["param_names"]:
        p[0] = float(np.clip(p[0], 0.0, 100.0))
    if model_name == "Double Weibull":
        p[1] = float(np.clip(p[1], 0.0, 1.0))
    elif model_name == "Triple Weibull":
        p[1] = float(np.clip(p[1], 0.0, 1.0))
        p[2] = float(np.clip(p[2], 0.0, 1.0))
        if p[1] + p[2] > 1.0:
            total = p[1] + p[2]
            if total > 0:
                p[1] /= total
                p[2] /= total
    return p


def _candidate_starts(model_name, base_init, lb, ub):
    base = np.asarray(base_init, dtype=float)
    starts = [base.copy()]
    if model_name == "Single Weibull":
        starts += [
            np.array([base[0] * 0.95, base[1] * 0.7, max(base[2] * 0.8, 0.2)]),
            np.array([min(100.0, base[0] * 1.02), base[1] * 1.4, max(base[2] * 1.4, 1.2)]),
        ]
    elif model_name == "Double Weibull":
        starts += [
            np.array([base[0], 0.30, base[2] * 0.7, max(base[3] * 0.8, 0.2), base[4] * 1.2, max(base[5] * 1.2, 0.5)]),
            np.array([base[0], 0.70, base[2] * 1.1, max(base[3] * 1.3, 0.4), base[4] * 0.8, max(base[5] * 0.8, 0.3)]),
        ]
    else:
        starts += [
            np.array([base[0], 0.20, 0.20, base[3] * 0.7, max(base[4] * 0.8, 0.2), base[5], max(base[6] * 1.1, 0.3), base[7] * 1.2, max(base[8] * 1.1, 0.4)]),
            np.array([base[0], 0.45, 0.15, base[3], max(base[4] * 1.2, 0.3), base[5] * 0.85, max(base[6] * 0.8, 0.2), base[7] * 1.1, max(base[8] * 1.3, 0.5)]),
            np.array([base[0], 0.15, 0.45, base[3] * 1.1, max(base[4] * 0.9, 0.2), base[5] * 1.1, max(base[6] * 1.3, 0.3), base[7] * 0.9, max(base[8] * 0.8, 0.2)]),
        ]
    out = []
    for s in starts:
        s = np.clip(s, lb + 1e-12, ub - 1e-12)
        s = _enforce_weight_constraints(model_name, s)
        out.append(s)
    return out


def _residuals_with_constraints(params, t_h, y, model_name):
    params = _enforce_weight_constraints(model_name, params)
    pred = MODEL_SPECS[model_name]["func"](t_h, *params)
    resid = pred - y
    if model_name == "Triple Weibull":
        f1 = params[1]
        f2 = params[2]
        excess = max(f1 + f2 - 1.0, 0.0)
        if excess > 0:
            resid = np.concatenate([resid, np.repeat(excess * 1e4, max(3, len(t_h) // 2))])
    return resid


def _numerical_jacobian(func, t_h, params, lb=None, ub=None, eps=1e-6):
    params = np.asarray(params, dtype=float)
    base = func(t_h, *params)
    n = len(base)
    p = len(params)
    J = np.zeros((n, p), dtype=float)
    for j in range(p):
        step = eps * max(abs(params[j]), 1.0)
        if lb is not None:
            step = min(step, max(params[j] - lb[j], 1e-8))
        if ub is not None:
            step = min(step, max(ub[j] - params[j], 1e-8))
        step = max(step, 1e-8)
        p_plus = params.copy()
        p_minus = params.copy()
        p_plus[j] = params[j] + step
        p_minus[j] = params[j] - step
        if ub is not None:
            p_plus[j] = min(p_plus[j], ub[j])
        if lb is not None:
            p_minus[j] = max(p_minus[j], lb[j])
        f_plus = func(t_h, *_enforce_weight_constraints("Triple Weibull" if len(params) == 9 else "Double Weibull" if len(params) == 6 else "Single Weibull", p_plus))
        f_minus = func(t_h, *_enforce_weight_constraints("Triple Weibull" if len(params) == 9 else "Double Weibull" if len(params) == 6 else "Single Weibull", p_minus))
        denom = p_plus[j] - p_minus[j]
        if abs(denom) < 1e-12:
            J[:, j] = 0.0
        else:
            J[:, j] = (f_plus - f_minus) / denom
    return J


def _infer_parameter_statistics(model_name, t_h, y, params, lb, ub, alpha=0.05):
    func = MODEL_SPECS[model_name]["func"]
    params = _enforce_weight_constraints(model_name, params)
    yhat = func(t_h, *params)
    resid = y - yhat
    n = len(y)
    p = len(params)
    dof = max(n - p, 0)
    J = _numerical_jacobian(func, t_h, params, lb=lb, ub=ub)
    if dof <= 0:
        se = np.full(p, np.nan)
        t_val = np.full(p, np.nan)
        p_val = np.full(p, np.nan)
        lcl = np.full(p, np.nan)
        ucl = np.full(p, np.nan)
        cov = np.full((p, p), np.nan)
    else:
        mse = float(np.sum(resid ** 2) / dof)
        JTJ = J.T @ J
        cov = mse * np.linalg.pinv(JTJ)
        se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
        t_val = np.divide(params, se, out=np.full_like(params, np.nan), where=se > 0)
        p_val = 2.0 * (1.0 - stats.t.cdf(np.abs(t_val), dof))
        tcrit = stats.t.ppf(1.0 - alpha / 2.0, dof)
        lcl = params - tcrit * se
        ucl = params + tcrit * se
    rse = np.divide(se * 100.0, np.abs(params), out=np.full_like(params, np.nan), where=np.abs(params) > 0)
    return {
        "jacobian": J,
        "covariance": cov,
        "se": se,
        "rse_pct": rse,
        "t_value": t_val,
        "p_value": p_val,
        "lcl": lcl,
        "ucl": ucl,
        "dof": dof,
        "yhat": yhat,
    }


def fit_weibull_model(t_h, y, model_name, parameter_tables=None):
    spec = MODEL_SPECS[model_name]
    mask = np.isfinite(t_h) & np.isfinite(y)
    t_h = np.asarray(t_h, dtype=float)[mask]
    y = np.asarray(y, dtype=float)[mask]
    if len(t_h) < len(spec["param_names"]) + 1:
        raise ValueError(f"{model_name} needs more data points than fitted parameters.")
    init, lb, ub = _resolve_fit_settings(model_name, t_h, y, parameter_tables=parameter_tables)
    best = None
    for start in _candidate_starts(model_name, init, lb, ub):
        try:
            res = least_squares(
                _residuals_with_constraints,
                x0=start,
                bounds=(lb, ub),
                args=(t_h, y, model_name),
                max_nfev=100000,
                method="trf",
            )
            if not res.success:
                continue
            popt = _enforce_weight_constraints(model_name, res.x)
            yhat = spec["func"](t_h, *popt)
            rss = float(np.sum((y - yhat) ** 2))
            infer = _infer_parameter_statistics(model_name, t_h, y, popt, lb, ub)
            n = len(y)
            k = len(popt)
            aic = n * np.log(max(rss, 1e-12) / n) + 2 * k
            bic = n * np.log(max(rss, 1e-12) / n) + k * np.log(max(n, 1))
            tss = float(np.sum((y - np.mean(y)) ** 2))
            r2 = 1.0 - rss / tss if tss > 0 else np.nan
            adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / max(n - k - 1, 1) if np.isfinite(r2) else np.nan
            candidate = {
                "params": popt,
                "init": init,
                "lb": lb,
                "ub": ub,
                "se": infer["se"],
                "rse_pct": infer["rse_pct"],
                "t_value": infer["t_value"],
                "p_value": infer["p_value"],
                "lcl": infer["lcl"],
                "ucl": infer["ucl"],
                "jacobian": infer["jacobian"],
                "covariance": infer["covariance"],
                "dof": infer["dof"],
                "rss": rss,
                "aic": float(aic),
                "bic": float(bic),
                "r2": float(r2) if np.isfinite(r2) else np.nan,
                "adj_r2": float(adj_r2) if np.isfinite(adj_r2) else np.nan,
                "yhat": yhat,
            }
            if (best is None) or (rss < best["rss"]):
                best = candidate
        except Exception:
            continue
    if best is None:
        raise ValueError(f"{model_name} fit did not converge for the provided profile.")
    return best


def _parameter_summary_row(model_name, fit, replicate_count):
    spec = MODEL_SPECS[model_name]
    rows = []
    for pname, dname, est, se in zip(spec["param_names"], spec["display_names"], fit["params"], fit["se"]):
        rows.append({
            "Model": model_name,
            "Parameter": dname,
            "Parameter key": pname,
            "Estimate": est,
            "SE": se,
            "Profiles summarized": replicate_count,
        })
    return rows


def _single_profile_detail_rows(model_name, fit):
    spec = MODEL_SPECS[model_name]
    rows = []
    for idx, (pname, dname) in enumerate(zip(spec["param_names"], spec["display_names"])):
        rows.append({
            "Model": model_name,
            "Coeff.": dname,
            "Init. Val.": fit["init"][idx],
            "Min (≥)": fit["lb"][idx],
            "Max (≤)": fit["ub"][idx],
            "Estimate": fit["params"][idx],
            "S.E.": fit["se"][idx],
            "R.S.E (%)": fit["rse_pct"][idx],
            "t-Value": fit["t_value"][idx],
            "p-Value": fit["p_value"][idx],
            "95% LCL": fit["lcl"][idx],
            "95% UCL": fit["ucl"][idx],
        })
    return rows


def fit_weibull_suite(df, time_unit_label, parameter_tables=None):
    factor = TIME_UNIT_TO_HOURS[time_unit_label]
    t_in = df["Time_input"].to_numpy(dtype=float)
    t_h = t_in * factor
    rep_cols = [c for c in df.columns if c != "Time_input"]
    results = []
    per_rep_rows = []
    param_rows = []
    single_detail_rows = []
    mean_profile = pd.DataFrame({"Time_input": t_in, "Time_h": t_h})
    mean_profile["Mean"] = df[rep_cols].mean(axis=1, skipna=True)
    mean_profile["SD"] = df[rep_cols].std(axis=1, ddof=1, skipna=True)
    mean_profile["SE"] = mean_profile["SD"] / np.sqrt(max(len(rep_cols), 1))
    mean_models = {}

    for model_name in MODEL_SPECS:
        rep_fits = {}
        for rep in rep_cols:
            y = df[rep].to_numpy(dtype=float)
            fit = fit_weibull_model(t_h, y, model_name, parameter_tables=parameter_tables)
            rep_fits[rep] = fit
            row = {"Model": model_name, "Replicate": rep, "AIC": fit["aic"], "BIC": fit["bic"], "RSS": fit["rss"], "R²": fit["r2"], "Adjusted R²": fit["adj_r2"]}
            per_rep_rows.append(row)
            if len(rep_cols) == 1:
                single_detail_rows.extend(_single_profile_detail_rows(model_name, fit))
        param_mat = np.vstack([rep_fits[r]["params"] for r in rep_cols])
        spec = MODEL_SPECS[model_name]
        if param_mat.shape[0] > 1:
            means = np.nanmean(param_mat, axis=0)
            ses = np.nanstd(param_mat, axis=0, ddof=1) / np.sqrt(param_mat.shape[0])
            fit_for_summary = {"params": means, "se": ses}
            for idx, (pname, dname) in enumerate(zip(spec["param_names"], spec["display_names"])):
                param_rows.append({
                    "Model": model_name,
                    "Parameter": dname,
                    "Parameter key": pname,
                    "Mean estimate": means[idx],
                    "SE across replicates": ses[idx],
                    "Profiles summarized": param_mat.shape[0],
                })
        else:
            one_fit = rep_fits[rep_cols[0]]
            for idx, (pname, dname) in enumerate(zip(spec["param_names"], spec["display_names"])):
                param_rows.append({
                    "Model": model_name,
                    "Parameter": dname,
                    "Parameter key": pname,
                    "Mean estimate": one_fit["params"][idx],
                    "SE across replicates": one_fit["se"][idx],
                    "Profiles summarized": 1,
                })
        mean_fit = fit_weibull_model(t_h, mean_profile["Mean"].to_numpy(dtype=float), model_name, parameter_tables=parameter_tables)
        mean_models[model_name] = mean_fit
        results.append({
            "Model": model_name,
            "Replicates fitted": len(rep_cols),
            "Mean replicate AIC": float(np.mean([rep_fits[r]["aic"] for r in rep_cols])),
            "Mean replicate BIC": float(np.mean([rep_fits[r]["bic"] for r in rep_cols])),
            "Mean replicate RSS": float(np.mean([rep_fits[r]["rss"] for r in rep_cols])),
            "Mean replicate R²": float(np.mean([rep_fits[r]["r2"] for r in rep_cols])),
            "Mean-profile AIC": mean_fit["aic"],
            "Mean-profile BIC": mean_fit["bic"],
            "Mean-profile RSS": mean_fit["rss"],
            "Mean-profile R²": mean_fit["r2"],
        })

    summary_df = pd.DataFrame(results).sort_values(["Mean-profile AIC", "Mean replicate AIC"]).reset_index(drop=True)
    best_model = summary_df.iloc[0]["Model"]
    fit_grid_h = np.linspace(float(np.min(t_h)), float(np.max(t_h)), 400)
    fit_grid_in = fit_grid_h / factor
    curve_rows = []
    for model_name, fit in mean_models.items():
        yfit = MODEL_SPECS[model_name]["func"](fit_grid_h, *fit["params"])
        for x_in, x_h, yv in zip(fit_grid_in, fit_grid_h, yfit):
            curve_rows.append({"Model": model_name, "Time_input": x_in, "Time_h": x_h, "Predicted": yv})

    return {
        "summary_df": summary_df,
        "per_rep_df": pd.DataFrame(per_rep_rows),
        "param_df": pd.DataFrame(param_rows),
        "single_profile_detail_df": pd.DataFrame(single_detail_rows),
        "mean_profile_df": mean_profile,
        "curve_df": pd.DataFrame(curve_rows),
        "best_model": best_model,
        "mean_model_fits": mean_models,
        "time_factor": factor,
        "replicate_cols": rep_cols,
        "input_df": df.copy(),
    }


def plot_weibull_profile_fits(df, fit_pack, time_unit_label):
    cfg = safe_get_plot_cfg("Dissolution comparison")
    fig, ax = plt.subplots(figsize=(cfg["fig_w"], cfg["fig_h"]))
    time_col = df["Time_input"].to_numpy(dtype=float)
    rep_cols = fit_pack["replicate_cols"]
    for rep in rep_cols:
        ax.plot(time_col, df[rep].to_numpy(dtype=float), color=cfg["secondary_color"], alpha=0.25, linewidth=max(0.8, cfg["aux_line_width"]))
    mean_df = fit_pack["mean_profile_df"]
    ax.plot(time_col, mean_df["Mean"], marker="o", color=cfg["primary_color"], linewidth=cfg["line_width"], label="Observed mean")
    for model_name in MODEL_SPECS:
        sub = fit_pack["curve_df"].loc[fit_pack["curve_df"]["Model"] == model_name]
        lw = cfg["line_width"] + (0.8 if model_name == fit_pack["best_model"] else 0.0)
        alpha = 0.95 if model_name == fit_pack["best_model"] else 0.7
        ax.plot(sub["Time_input"], sub["Predicted"], linewidth=lw, alpha=alpha, label=model_name)
    apply_ax_style(ax, "Observed dissolution profile and Weibull fits", f"Time ({time_unit_label})", "% Dissolved", legend=True, plot_key="Dissolution comparison")
    return fig


def plot_model_comparison_aic(fit_pack):
    cfg = safe_get_plot_cfg("Dissolution comparison")
    fig, ax = plt.subplots(figsize=(cfg["fig_w"], cfg["fig_h"]))
    sdf = fit_pack["summary_df"].copy().sort_values("Mean-profile AIC")
    x = np.arange(len(sdf))
    width = 0.36
    ax.bar(x - width / 2, sdf["Mean-profile AIC"], width=width, label="Mean-profile AIC", alpha=0.85)
    ax.bar(x + width / 2, sdf["Mean replicate AIC"], width=width, label="Mean replicate AIC", alpha=0.65)
    ax.set_xticks(x)
    ax.set_xticklabels(sdf["Model"].tolist(), rotation=0)
    apply_ax_style(ax, "Weibull model comparison by AIC", "Model", "AIC", legend=True, plot_key="Dissolution comparison")
    return fig


def plot_residuals_best_model(fit_pack, time_unit_label):
    cfg = safe_get_plot_cfg("Dissolution comparison")
    best = fit_pack["best_model"]
    mean_fit = fit_pack["mean_model_fits"][best]
    mean_df = fit_pack["mean_profile_df"]
    resid = mean_df["Mean"].to_numpy(dtype=float) - mean_fit["yhat"]
    t_in = mean_df["Time_input"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(cfg["fig_w"], cfg["fig_h"]))
    ax.axhline(0.0, color=cfg["tertiary_color"], linewidth=cfg["aux_line_width"], linestyle=cfg["aux_line_style"])
    ax.plot(t_in, resid, marker="o", color=cfg["primary_color"], linewidth=cfg["line_width"])
    apply_ax_style(ax, f"Residual plot for best model ({best})", f"Time ({time_unit_label})", "Residual", legend=False, plot_key="Dissolution comparison")
    return fig


def save_invitrofit_to_session(fit_pack, time_unit_label):
    best = fit_pack["best_model"]
    best_params = fit_pack["param_df"].loc[fit_pack["param_df"]["Model"] == best].copy()
    param_map = dict(zip(best_params["Parameter key"], best_params["Mean estimate"]))
    param_se_map = dict(zip(best_params["Parameter key"], best_params["SE across replicates"]))
    st.session_state["InVitroFit"] = {
        "name": "InVitroFit",
        "model": best,
        "time_input_units": time_unit_label,
        "stored_time_units": "Hours",
        "parameter_estimates": param_map,
        "parameter_se": param_se_map,
        "model_comparison": fit_pack["summary_df"].to_dict(orient="records"),
    }


def render():
    render_display_settings()
    st.sidebar.title("💊 lm Stats")
    st.sidebar.markdown("IVIVC Suite")
    tool = st.sidebar.radio("IVIVC tool", ["💊 Dissolution Comparison (f₂)", "📈 In Vitro Weibull Fit"], key="ivivc_tool")
    st.sidebar.caption("This page contains dissolution similarity tools and in vitro profile fitting tools for IVIVC workflows.")

    if tool == "💊 Dissolution Comparison (f₂)":
        app_header("💊 Dissolution Comparison (f₂)", "FDA-style point selection, conventional f2 checks, and optional bootstrap / BCa confidence intervals.")
        col1, col2 = st.columns(2)
        with col1:
            c_sample_ref, c_sample_fill = st.columns([1, 5])
            with c_sample_ref:
                st.button("Sample Data", key="sample_f2_ivivc", on_click=load_dual_sample_text, args=("f2_ref_input_ivivc", "f2_ref", "f2_test_input_ivivc", "f2_test"))
            with c_sample_fill:
                ref_text = st.text_area("Reference profile table", height=220, key="f2_ref_input_ivivc")
        with col2:
            test_text = st.text_area("Test profile table", height=220, key="f2_test_input_ivivc")
        s1, s2, s3 = st.columns([1.1, 1.4, 0.9])
        with s1:
            include_zero = st.checkbox("Include time zero", value=False)
        with s2:
            cutoff_mode = st.selectbox(
                "Point selection",
                ["all", "apply_85"],
                format_func=lambda x: "Use all common timepoints" if x == "all" else "FDA-style: stop after first point where both are ≥ threshold",
            )
        with s3:
            threshold = st.number_input("Threshold", value=85.0, step=1.0)
        b1, b2, b3, b4 = st.columns([1, 1.1, 1.1, 0.8])
        with b1:
            bootstrap_on = st.checkbox("Bootstrap f2 CI", value=False)
        with b2:
            boot_method = st.selectbox("Bootstrap CI method", ["percentile", "bca", "both"], disabled=not bootstrap_on)
        with b3:
            boot_conf = st.slider("Bootstrap confidence", 0.80, 0.99, 0.90, 0.01, format="%.2f", disabled=not bootstrap_on)
        with b4:
            decimals = st.slider("Decimals", 1, 8, 2, key="f2_dec_ivivc")
        b5, b6, b7 = st.columns([1, 1, 1])
        with b5:
            boot_n = st.number_input("Resamples", min_value=200, max_value=50000, value=2000, step=100, disabled=not bootstrap_on)
        with b6:
            boot_seed = st.number_input("Seed", min_value=0, value=123, step=1, disabled=not bootstrap_on)
        with b7:
            show_units = st.checkbox("Show individual unit traces", value=True)
        p1, p2 = st.columns(2)
        with p1:
            profile_title = st.text_input("Profile plot title", value="Dissolution Profiles")
        with p2:
            y_label = st.text_input("Y label", value="% Dissolved")

        if ref_text and test_text:
            try:
                ref_df = dis_parse_profile_table(ref_text)
                test_df = dis_parse_profile_table(test_text)
                ref_summary = dis_profile_summary(ref_df)
                test_summary = dis_profile_summary(test_df)
                merged = dis_merge_profiles(ref_summary, test_summary)
                selected, _ = dis_select_points(merged, include_zero, cutoff_mode, threshold)
                f2 = dis_calc_f2(selected["mean_ref"], selected["mean_test"])
                selected = selected.copy()
                selected["abs_diff"] = (selected["mean_ref"] - selected["mean_test"]).abs()
                selected["sq_diff"] = (selected["mean_ref"] - selected["mean_test"]) ** 2
                fda_tbl, fda_detail_tbl, conventional_ok = dis_fda_checks(ref_df, test_df, selected, threshold=threshold, include_zero=include_zero)
                all_points_tbl = merged.copy()
                all_points_tbl["Used for f2"] = np.where(all_points_tbl["Time"].isin(selected["Time"]), "Yes", "No")
                assess_tbl = pd.DataFrame({
                    "Selected timepoints": [len(selected)],
                    "f2 Statistic": [f2],
                    "Conclusion": ["Similar (f2 ≥ 50)" if f2 >= 50 else "Not similar (f2 < 50)"],
                    "FDA-style applicability": ["Applicable" if conventional_ok else "Criteria warning"],
                })
                fig_main = dis_plot_profiles(ref_df, test_df, ref_summary, test_summary, selected, show_units=show_units, title=profile_title, ylabel=y_label)
                boot_tbl = None
                boot_figs = {}
                if bootstrap_on:
                    selected_times = np.sort(selected["Time"].to_numpy(dtype=float))
                    ref_mat, _ = dis_get_selected_matrix(ref_df, selected_times)
                    test_mat, _ = dis_get_selected_matrix(test_df, selected_times)
                    boot_vals = dis_bootstrap_f2(ref_mat, test_mat, n_boot=int(boot_n), seed=int(boot_seed))
                    rows = [{
                        "Observed f2": f2,
                        "Bootstrap mean f2": float(np.mean(boot_vals)),
                        "Bootstrap median f2": float(np.median(boot_vals)),
                        "Bootstrap SD": float(np.std(boot_vals, ddof=1)),
                        "Resamples": int(boot_n),
                        "Seed": int(boot_seed),
                        "CI confidence": boot_conf,
                    }]
                    if boot_method in ["percentile", "both"]:
                        pct_low, pct_high = dis_percentile_interval(boot_vals, conf=boot_conf)
                        rows[0]["Percentile CI lower"] = pct_low
                        rows[0]["Percentile CI upper"] = pct_high
                        fig_boot_pct = dis_plot_bootstrap_f2_distribution(
                            boot_vals, f2, pct_low, pct_high,
                            ci_label=f"{int(round(boot_conf * 100))}%", title="Bootstrap distribution plot (Percentile CI)"
                        )
                        if fig_boot_pct is not None:
                            boot_figs["Bootstrap distribution plot (Percentile CI)"] = fig_boot_pct
                    if boot_method in ["bca", "both"]:
                        jack_vals = dis_jackknife_f2(ref_mat, test_mat)
                        bca_low, bca_high, z0, accel = dis_bca_interval(f2, boot_vals, jack_vals, conf=boot_conf)
                        rows[0]["BCa CI lower"] = bca_low
                        rows[0]["BCa CI upper"] = bca_high
                        rows[0]["BCa z0"] = z0
                        rows[0]["BCa acceleration"] = accel
                        fig_boot_bca = dis_plot_bootstrap_f2_distribution(
                            boot_vals, f2, bca_low, bca_high,
                            ci_label=f"{int(round(boot_conf * 100))}%", title="Bootstrap distribution plot (BCa CI)"
                        )
                        if fig_boot_bca is not None:
                            boot_figs["Bootstrap distribution plot (BCa CI)"] = fig_boot_bca
                    boot_tbl = pd.DataFrame(rows)

                cols = st.columns(4)
                cols[0].metric("f2", f"{f2:.{decimals}f}")
                cols[1].metric("Selected points", f"{len(selected)}")
                cols[2].metric("Similarity decision", "Similar" if f2 >= 50 else "Not similar")
                cols[3].metric("FDA-style check", "Pass" if conventional_ok else "Warning")

                t1, t2, t3 = st.tabs(["Summary", "FDA criteria & selected points", "Bootstrap"])
                with t1:
                    report_table(
                        merged.rename(columns={
                            "n_units_ref": "Ref. Units (N)", "mean_ref": "Ref. Mean", "sd_ref": "Ref. SD", "cv_pct_ref": "Ref. CV (%)",
                            "n_units_test": "Test Units (N)", "mean_test": "Test Mean", "sd_test": "Test SD", "cv_pct_test": "Test CV (%)"
                        }),
                        "Profile summary table",
                        decimals,
                    )
                    report_table(assess_tbl, "f2 assessment", decimals)
                    show_figure(fig_main)
                with t2:
                    report_table(fda_tbl, "FDA-style criteria check", decimals)
                    report_table(fda_detail_tbl, "FDA-style criteria details", decimals)
                    report_table(all_points_tbl, "Common timepoints and whether they were used in f2", decimals)
                    report_table(selected, "Selected points used for f2 calculation", decimals)
                with t3:
                    if bootstrap_on and boot_tbl is not None:
                        report_table(boot_tbl, "Bootstrap f2 confidence intervals", decimals)
                        for _, fig in boot_figs.items():
                            show_figure(fig)
                    else:
                        st.info("Enable 'Bootstrap f2 CI' above to calculate percentile and/or BCa confidence intervals and show the extra bootstrap graph.")

                table_map = {
                    "Profile Summary": merged,
                    "f2 Assessment": assess_tbl,
                    "FDA Criteria Check": fda_tbl,
                    "FDA Criteria Details": fda_detail_tbl,
                    "Selected Points Used for f2": selected,
                }
                if boot_tbl is not None:
                    table_map["Bootstrap f2 Confidence Intervals"] = boot_tbl
                figure_map = {"Dissolution profiles": fig_to_png_bytes(fig_main)}
                for title, fig in boot_figs.items():
                    figure_map[title] = fig_to_png_bytes(fig)
                export_results(
                    prefix="dissolution_f2_enhanced",
                    report_title="Statistical Analysis Report",
                    module_name="Dissolution Comparison (f₂)",
                    statistical_analysis="Reference and test dissolution profiles were summarized, f2 was calculated, FDA-style checks were evaluated, and optional bootstrap intervals were supported.",
                    offer_text="This module shows both the conventional f2 result and the criteria behind its use.",
                    python_tools="pandas, numpy, scipy.stats, matplotlib, openpyxl, reportlab",
                    table_map=table_map,
                    figure_map=figure_map,
                    conclusion="Review the selected points, FDA-style criteria, and any bootstrap interval before finalizing similarity.",
                    decimals=decimals,
                )
            except Exception as e:
                st.error(str(e))

    elif tool == "📈 In Vitro Weibull Fit":
        app_header("📈 In Vitro Weibull Fit", "Fit single, double, and triple Weibull models to one or more dissolution profiles, compare AIC values, and save the best model for reuse in the current session as InVitroFit.")
        c1, c2 = st.columns([1, 6])
        with c1:
            st.button("Sample Data", key="sample_weibull_ivivc", on_click=load_weibull_sample_text, args=("weibull_input_ivivc",))
        with c2:
            fit_text = st.text_area("Dissolution table (first column = time, remaining columns = one or more profiles / replicates)", height=260, key="weibull_input_ivivc")
        u1, u2 = st.columns([1, 1])
        with u1:
            time_unit_label = st.selectbox("Input time units", ["Minutes", "Hours", "Days"], index=0)
        with u2:
            decimals = st.slider("Decimals", 1, 8, 3, key="weibull_dec_ivivc")
        st.caption("The selected input time unit is converted internally so all fitting, stored parameters, and reports are in hours. Fmax is constrained to a maximum of 100, and the Triple Weibull fit enforces f1 + f2 ≤ 1 so the implied third fraction remains valid.")

        if fit_text:
            try:
                fit_df = parse_dissolution_weibull_table(fit_text)
                t_hours_preview = fit_df["Time_input"].to_numpy(dtype=float) * TIME_UNIT_TO_HOURS[time_unit_label]
                y_preview = fit_df.iloc[:, 1:].to_numpy(dtype=float).ravel()
                y_preview = y_preview[np.isfinite(y_preview)]
                parameter_tables = {}
                with st.expander("Parameter bounds and starting values", expanded=False):
                    st.write("Edit the initial values and fitting bounds for each model. MDT values below are shown in hours, regardless of the input time unit you selected above.")
                    default_tables = build_weibull_parameter_tables(t_hours_preview, y_preview)
                    for model_name in MODEL_SPECS:
                        st.markdown(f"**{model_name}**")
                        editor = st.data_editor(
                            default_tables[model_name],
                            key=f"weibull_editor_{_slugify(model_name)}",
                            hide_index=True,
                            num_rows="fixed",
                            use_container_width=True,
                        )
                        parameter_tables[model_name] = pd.DataFrame(editor)

                fit_pack = fit_weibull_suite(fit_df, time_unit_label, parameter_tables=parameter_tables)
                summary_df = fit_pack["summary_df"]
                per_rep_df = fit_pack["per_rep_df"]
                param_df = fit_pack["param_df"]
                single_profile_detail_df = fit_pack["single_profile_detail_df"]
                mean_profile_df = fit_pack["mean_profile_df"]
                best_model = fit_pack["best_model"]

                m1, m2, m3 = st.columns(3)
                m1.metric("Best model", best_model)
                m2.metric("Best AIC (mean profile)", f"{summary_df.loc[summary_df['Model'] == best_model, 'Mean-profile AIC'].iloc[0]:.{decimals}f}")
                m3.metric("Profiles fitted", str(len(fit_pack["replicate_cols"])))

                fig_fit = plot_weibull_profile_fits(fit_df, fit_pack, time_unit_label)
                fig_aic = plot_model_comparison_aic(fit_pack)
                fig_resid = plot_residuals_best_model(fit_pack, time_unit_label)

                tab_names = ["Summary", "Per-profile results"]
                if len(fit_pack["replicate_cols"]) == 1:
                    tab_names.append("Single-profile parameter statistics")
                tab_names.append("Save model")
                tabs = st.tabs(tab_names)
                with tabs[0]:
                    input_show = fit_df.copy()
                    input_show.insert(1, "Time_h", fit_df["Time_input"] * fit_pack["time_factor"])
                    report_table(input_show, "Input dissolution data used for Weibull fitting", decimals)
                    report_table(summary_df, "Model comparison summary", decimals)
                    report_table(param_df, "Parameter estimates summary", decimals)
                    report_table(mean_profile_df, "Mean dissolution profile used for summary fitting", decimals)
                    show_figure(fig_fit, caption="Observed dissolution profile and Weibull fits")
                    show_figure(fig_aic, caption="Weibull model comparison by AIC")
                    show_figure(fig_resid, caption=f"Residual plot for best model ({best_model})")
                with tabs[1]:
                    report_table(per_rep_df, "Per-profile Weibull fit statistics", decimals)
                if len(fit_pack["replicate_cols"]) == 1:
                    with tabs[2]:
                        report_table(single_profile_detail_df, "Single-profile parameter statistics from the Jacobian-based covariance calculation", decimals)
                with tabs[-1]:
                    best_params = param_df.loc[param_df["Model"] == best_model].copy()
                    report_table(best_params, f"Parameters that will be saved for {best_model}", decimals)
                    if st.button("Save best model as InVitroFit", key="save_invitrofit_button"):
                        save_invitrofit_to_session(fit_pack, time_unit_label)
                        st.success("The best-fitting Weibull model was saved in this session as InVitroFit.")
                    if "InVitroFit" in st.session_state:
                        current = st.session_state["InVitroFit"]
                        st.info(f"Current saved in-session model: {current.get('name', 'InVitroFit')} ({current.get('model', '-')}, stored in hours).")

                table_map = {
                    "Input Dissolution Data": fit_df.assign(Time_h=fit_df["Time_input"] * fit_pack["time_factor"]),
                    "Model Comparison Summary": summary_df,
                    "Per-Profile Weibull Fit Statistics": per_rep_df,
                    "Parameter Estimates Summary": param_df,
                    "Mean Dissolution Profile": mean_profile_df,
                }
                if len(fit_pack["replicate_cols"]) == 1 and not single_profile_detail_df.empty:
                    table_map["Single-Profile Parameter Statistics"] = single_profile_detail_df
                figure_map = {
                    "Observed dissolution profile and Weibull fits": fig_to_png_bytes(fig_fit),
                    "Weibull model comparison by AIC": fig_to_png_bytes(fig_aic),
                    f"Residual plot for best model ({best_model})": fig_to_png_bytes(fig_resid),
                }
                export_results(
                    prefix="ivivc_weibull_fit",
                    report_title="Statistical Analysis Report",
                    module_name="In Vitro Weibull Fit",
                    statistical_analysis="Single-, double-, and triple-Weibull models were fitted to the dissolution profile data. The fitting now respects user-defined initial values and bounds, constrains Fmax to a maximum of 100, enforces f1 + f2 ≤ 1 for the Triple Weibull model, and stores all fitted MDT values in hours. When a single profile is fitted, parameter uncertainty is additionally summarized from a Jacobian-based covariance calculation.",
                    offer_text="This module supports in vitro dissolution profile fitting for IVIVC workflows and can store the best model in-session for later tools.",
                    python_tools="pandas, numpy, scipy.optimize.least_squares, scipy.stats, matplotlib, openpyxl, reportlab",
                    table_map=table_map,
                    figure_map=figure_map,
                    conclusion=f"Based on the current data, the best Weibull model by AIC was {best_model}. Review the parameter bounds you selected, the model comparison table, and any single-profile uncertainty table before using the saved InVitroFit model downstream.",
                    decimals=decimals,
                )
            except Exception as e:
                st.error(str(e))
