import modules.common as common
from modules.common import *
from scipy.optimize import least_squares
from scipy.integrate import solve_ivp
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

DOSE_UNIT_TO_MG = {
    "ng": 1e-6,
    "ug": 1e-3,
    "mg": 1.0,
    "g": 1e3,
}

VOLUME_UNIT_TO_L = {
    "uL": 1e-6,
    "mL": 1e-3,
    "L": 1.0,
}

CP_MG_PER_L_TO_UNIT = {
    "mg/L": 1.0,
    "ug/mL": 1.0,
    "ng/mL": 1e3,
    "ug/L": 1e3,
    "mg/mL": 1e-3,
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


MODEL_CHOICE_OPTIONS = ["All Weibull models (assess best by AIC)"] + list(MODEL_SPECS.keys())


def _resolve_model_choices(choice=None):
    if choice is None or str(choice).startswith("All Weibull models"):
        return list(MODEL_SPECS.keys())
    if choice not in MODEL_SPECS:
        raise ValueError("Unknown Weibull model selection.")
    return [choice]


def _progress_update(callback, step, total, message):
    if callback is not None:
        callback(step, total, message)



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


def _parameter_long_to_wide(model_name, df_long):
    spec = MODEL_SPECS[model_name]
    df = df_long.copy()
    if "Parameter" not in df.columns:
        df.insert(0, "Parameter", spec["display_names"])
    df["Parameter"] = spec["display_names"]
    wide = df.set_index("Parameter").T.reset_index().rename(columns={"index": "Setting"})
    desired_order = ["Setting"] + spec["display_names"]
    return wide.loc[:, [c for c in desired_order if c in wide.columns]]


def _parameter_wide_to_long(model_name, editor_df):
    spec = MODEL_SPECS[model_name]
    expected_settings = ["Initial Value", "Min (≥)", "Max (≤)"]
    df = pd.DataFrame(editor_df).copy().reset_index(drop=True)
    if "Parameter" in df.columns:
        return df
    if "Setting" not in df.columns:
        if len(df.columns) == len(spec["display_names"]) + 1:
            df = df.rename(columns={df.columns[0]: "Setting"})
        else:
            raise ValueError(f"Parameter table for {model_name} is incomplete.")
    df["Setting"] = df["Setting"].astype(str)
    long_rows = []
    for param in spec["display_names"]:
        if param not in df.columns:
            raise ValueError(f"Column '{param}' is missing from the wide parameter table for {model_name}.")
        row = {"Parameter": param}
        for setting in expected_settings:
            match = df.loc[df["Setting"] == setting, param]
            if match.empty:
                raise ValueError(f"Row '{setting}' is missing from the wide parameter table for {model_name}.")
            row[setting] = match.iloc[0]
        long_rows.append(row)
    return pd.DataFrame(long_rows)


def _wide_parameter_statistics(detail_df, model_name):
    if detail_df is None or len(detail_df) == 0:
        return pd.DataFrame()
    sub = detail_df.loc[detail_df["Model"] == model_name].copy()
    if sub.empty:
        return pd.DataFrame()
    value_cols = [c for c in sub.columns if c not in {"Model", "Coeff."}]
    wide = sub.set_index("Coeff.")[value_cols].T.reset_index().rename(columns={"index": "Statistic"})
    desired = ["Statistic"] + sub["Coeff."].tolist()
    return wide.loc[:, [c for c in desired if c in wide.columns]]


def _wide_parameter_estimate_summary(param_df):
    if param_df is None or len(param_df) == 0:
        return pd.DataFrame()
    frames = []
    for model_name, sub in param_df.groupby("Model", sort=False):
        row_est = {"Model": model_name, "Statistic": "Mean estimate"}
        row_se = {"Model": model_name, "Statistic": "SE across replicates"}
        for _, r in sub.iterrows():
            row_est[r["Parameter"]] = r["Mean estimate"]
            row_se[r["Parameter"]] = r["SE across replicates"]
        frames.extend([row_est, row_se])
    return pd.DataFrame(frames)


def _wide_saved_parameter_table(param_df, model_name):
    sub = param_df.loc[param_df["Model"] == model_name].copy()
    if sub.empty:
        return pd.DataFrame()
    row_est = {"Statistic": "Mean estimate"}
    row_se = {"Statistic": "SE across replicates"}
    for _, r in sub.iterrows():
        row_est[r["Parameter"]] = r["Mean estimate"]
        row_se[r["Parameter"]] = r["SE across replicates"]
    return pd.DataFrame([row_est, row_se])


def _slugify(s):
    return ''.join(ch.lower() if ch.isalnum() else '_' for ch in str(s)).strip('_')


def _sanitize_editor_table(model_name, editor_df):
    spec = MODEL_SPECS[model_name]
    param_names = spec["param_names"]
    display_names = spec["display_names"]
    if editor_df is None:
        raise ValueError(f"Parameter table for {model_name} is incomplete.")
    df = _parameter_wide_to_long(model_name, editor_df).copy().reset_index(drop=True)
    if len(df) != len(param_names):
        raise ValueError(f"Parameter table for {model_name} is incomplete.")
    if "Parameter" not in df.columns:
        df.insert(0, "Parameter", display_names)
    df["Parameter"] = display_names
    for col in ["Initial Value", "Min (≥)", "Max (≤)"]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' is required in the parameter table for {model_name}.")
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if df[["Initial Value", "Min (≥)", "Max (≤)"]].isna().any().any():
        raise ValueError(f"All initial, minimum, and maximum values must be numeric for {model_name}.")

    init = np.array(df["Initial Value"].to_numpy(dtype=float), dtype=float, copy=True)
    lb = np.array(df["Min (≥)"].to_numpy(dtype=float), dtype=float, copy=True)
    ub = np.array(df["Max (≤)"].to_numpy(dtype=float), dtype=float, copy=True)

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



def fit_weibull_suite(df, time_unit_label, parameter_tables=None, model_choice=None, progress_callback=None):
    factor = TIME_UNIT_TO_HOURS[time_unit_label]
    t_in = df["Time_input"].to_numpy(dtype=float)
    t_h = t_in * factor
    rep_cols = [c for c in df.columns if c != "Time_input"]
    selected_models = _resolve_model_choices(model_choice)
    results = []
    per_rep_rows = []
    param_rows = []
    single_detail_rows = []
    mean_profile = pd.DataFrame({"Time_input": t_in, "Time_h": t_h})
    mean_profile["Mean"] = df[rep_cols].mean(axis=1, skipna=True)
    mean_profile["SD"] = df[rep_cols].std(axis=1, ddof=1, skipna=True)
    mean_profile["SE"] = mean_profile["SD"] / np.sqrt(max(len(rep_cols), 1))
    mean_models = {}
    total_steps = max(1, len(selected_models) * (len(rep_cols) + 1))
    step = 0

    for model_name in selected_models:
        rep_fits = {}
        for rep in rep_cols:
            step += 1
            _progress_update(progress_callback, step, total_steps, f"Fitting {model_name} to dissolution profile '{rep}'")
            y = df[rep].to_numpy(dtype=float)
            fit = fit_weibull_model(t_h, y, model_name, parameter_tables=parameter_tables)
            rep_fits[rep] = fit
            per_rep_rows.append({"Model": model_name, "Replicate": rep, "AIC": fit["aic"], "BIC": fit["bic"], "RSS": fit["rss"], "R²": fit["r2"], "Adjusted R²": fit["adj_r2"]})
            if len(rep_cols) == 1:
                single_detail_rows.extend(_single_profile_detail_rows(model_name, fit))
        param_mat = np.vstack([rep_fits[r]["params"] for r in rep_cols])
        spec = MODEL_SPECS[model_name]
        if param_mat.shape[0] > 1:
            means = np.nanmean(param_mat, axis=0)
            ses = np.nanstd(param_mat, axis=0, ddof=1) / np.sqrt(param_mat.shape[0])
            for idx, (pname, dname) in enumerate(zip(spec["param_names"], spec["display_names"])):
                param_rows.append({"Model": model_name, "Parameter": dname, "Parameter key": pname, "Mean estimate": means[idx], "SE across replicates": ses[idx], "Profiles summarized": param_mat.shape[0]})
        else:
            one_fit = rep_fits[rep_cols[0]]
            for idx, (pname, dname) in enumerate(zip(spec["param_names"], spec["display_names"])):
                param_rows.append({"Model": model_name, "Parameter": dname, "Parameter key": pname, "Mean estimate": one_fit["params"][idx], "SE across replicates": one_fit["se"][idx], "Profiles summarized": 1})
        step += 1
        _progress_update(progress_callback, step, total_steps, f"Fitting {model_name} to the mean dissolution profile")
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

    param_df = pd.DataFrame(param_rows)
    single_profile_detail_df = pd.DataFrame(single_detail_rows)
    return {
        "summary_df": summary_df,
        "per_rep_df": pd.DataFrame(per_rep_rows),
        "param_df": param_df,
        "param_df_wide": _wide_parameter_estimate_summary(param_df),
        "single_profile_detail_df": single_profile_detail_df,
        "single_profile_detail_best_wide": _wide_parameter_statistics(single_profile_detail_df, best_model),
        "mean_profile_df": mean_profile,
        "curve_df": pd.DataFrame(curve_rows),
        "best_model": best_model,
        "mean_model_fits": mean_models,
        "time_factor": factor,
        "replicate_cols": rep_cols,
        "input_df": df.copy(),
        "model_names": selected_models,
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
    for model_name in fit_pack.get("model_names", list(MODEL_SPECS.keys())):
        sub = fit_pack["curve_df"].loc[fit_pack["curve_df"]["Model"] == model_name]
        lw = cfg["line_width"] + (0.8 if model_name == fit_pack["best_model"] else 0.0)
        alpha = 0.95 if model_name == fit_pack["best_model"] else 0.7
        ax.plot(sub["Time_input"], sub["Predicted"], linewidth=lw, alpha=alpha, label=model_name)
    apply_ax_style(ax, "Observed dissolution profile and Weibull fits", f"Time ({time_unit_label})", "% Dissolved", legend=True, plot_key="Dissolution comparison")
    return fig


def plot_best_model_profile(df, fit_pack, time_unit_label):
    cfg = safe_get_plot_cfg("Dissolution comparison")
    fig, ax = plt.subplots(figsize=(cfg["fig_w"], cfg["fig_h"]))
    time_col = df["Time_input"].to_numpy(dtype=float)
    rep_cols = fit_pack["replicate_cols"]
    for rep in rep_cols:
        ax.plot(time_col, df[rep].to_numpy(dtype=float), color=cfg["secondary_color"], alpha=0.22, linewidth=max(0.8, cfg["aux_line_width"]))
    mean_df = fit_pack["mean_profile_df"]
    ax.plot(time_col, mean_df["Mean"], marker="o", color=cfg["primary_color"], linewidth=cfg["line_width"], label="Observed mean")
    best = fit_pack["best_model"]
    sub = fit_pack["curve_df"].loc[fit_pack["curve_df"]["Model"] == best]
    ax.plot(sub["Time_input"], sub["Predicted"], linewidth=cfg["line_width"] + 0.9, color=cfg["tertiary_color"], label=f"Fitted {best}")
    apply_ax_style(ax, f"Experimental and fitted profile ({best})", f"Time ({time_unit_label})", "% Dissolved", legend=True, plot_key="Dissolution comparison")
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


PK_SYNTHETIC_SAMPLE = """Time	Concentration_1	Concentration_2	Concentration_3
0	0	0	0
1	8.94	18.2	14.1
6	12.6	18.2	16.1
12	9.08	14.4	11.4
24	11.7	14.7	12
48	7.86	12.9	9.81
96	10.4	18.3	14.5
168	11.8	21	15.3
264	8.77	12.3	10.9
360	8.57	6.26	7.48
456	3.97	1.45	5.43
552	3.51	0	3.23
648	2.49	0	1.28
744	2.21	0	0.564
840	1.66	0	0
"""


DECONV_SAMPLE_SETTINGS = {
    "time_unit_label": "Hours",
    "cp_unit": "ng/mL",
    "compartments": 3,
    "dose_value": 6666666.600,
    "dose_unit": "ng",
    "v_value": 1136.900,
    "v_unit": "mL",
    "k10": 0.770,
    "k12": 1.382,
    "k21": 1.814,
    "k13": 1.000,
    "k31": 0.000,
    "model_choice": "Triple Weibull",
}


DECONV_SAMPLE_STARTS = {
    "Single Weibull": {
        "Parameter": ["Fmax", "MDT1", "β1"],
        "Initial Value": [72.90, 265.68, 1.57],
        "Min (≥)": [1e-6, 1e-6, 1e-6],
        "Max (≤)": [100.0, 5000.0, 50.0],
        "Fix": [False, False, False],
    },
    "Double Weibull": {
        "Parameter": ["Fmax", "f1", "MDT1", "β1", "MDT2", "β2"],
        "Initial Value": [72.90, 0.04, 7.48, 0.83, 29.31, 6.75],
        "Min (≥)": [1e-6, 0.0, 1e-6, 1e-6, 1e-6, 1e-6],
        "Max (≤)": [100.0, 1.0, 5000.0, 50.0, 5000.0, 50.0],
        "Fix": [False, False, False, False, False, False],
    },
    "Triple Weibull": {
        "Parameter": ["Fmax", "f1", "f2", "MDT1", "β1", "MDT2", "β2", "MDT3", "β3"],
        "Initial Value": [72.90, 0.04, 0.02, 7.48, 0.83, 29.31, 6.75, 265.68, 1.57],
        "Min (≥)": [1e-6, 0.0, 0.0, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6],
        "Max (≤)": [100.0, 1.0, 1.0, 5000.0, 50.0, 5000.0, 50.0, 5000.0, 50.0],
        "Fix": [False, False, False, False, False, False, False, False, False],
    },
}



def _deconv_editor_state_key(model_name):
    return f"_deconv_editor_state_{_slugify(model_name)}"


def _default_deconv_editor_table(model_name):
    if model_name in DECONV_SAMPLE_STARTS:
        return pd.DataFrame(DECONV_SAMPLE_STARTS[model_name]).copy()
    fallback = build_deconv_parameter_tables(np.asarray([0.0, 1.0]), np.asarray([0.0, 1.0]))
    return fallback[model_name].copy()


def _reset_deconv_editor_state(force=False):
    for model_name in MODEL_SPECS:
        key = _deconv_editor_state_key(model_name)
        if force or key not in st.session_state:
            st.session_state[key] = _default_deconv_editor_table(model_name)


def load_pk_sample_text(state_key):
    st.session_state[state_key] = PK_SYNTHETIC_SAMPLE



def load_pk_deconv_sample():
    st.session_state["pk_input_deconv"] = PK_SYNTHETIC_SAMPLE
    st.session_state["deconv_time_units"] = DECONV_SAMPLE_SETTINGS["time_unit_label"]
    st.session_state["deconv_cp_units"] = DECONV_SAMPLE_SETTINGS["cp_unit"]
    st.session_state["deconv_compartments"] = DECONV_SAMPLE_SETTINGS["compartments"]
    st.session_state["deconv_dose_value"] = DECONV_SAMPLE_SETTINGS["dose_value"]
    st.session_state["deconv_dose_unit"] = DECONV_SAMPLE_SETTINGS["dose_unit"]
    st.session_state["deconv_v_value"] = DECONV_SAMPLE_SETTINGS["v_value"]
    st.session_state["deconv_v_unit"] = DECONV_SAMPLE_SETTINGS["v_unit"]
    st.session_state["deconv_k10"] = DECONV_SAMPLE_SETTINGS["k10"]
    st.session_state["deconv_k12"] = DECONV_SAMPLE_SETTINGS["k12"]
    st.session_state["deconv_k21"] = DECONV_SAMPLE_SETTINGS["k21"]
    st.session_state["deconv_k13"] = DECONV_SAMPLE_SETTINGS["k13"]
    st.session_state["deconv_k31"] = DECONV_SAMPLE_SETTINGS["k31"]
    st.session_state["deconv_model_choice"] = DECONV_SAMPLE_SETTINGS["model_choice"]
    st.session_state["deconv_decimals"] = 3
    st.session_state.pop("deconv_last_pack", None)
    _reset_deconv_editor_state(force=True)



def parse_pk_profile_table(text):
    df = _coerce_numeric_df(text)
    time_col = df.columns[0]
    cp_cols = [c for c in df.columns[1:] if df[c].notna().any()]
    if not cp_cols:
        raise ValueError("Please provide at least one Cp column after the time column.")
    out = df[[time_col] + cp_cols].copy().rename(columns={time_col: "Time_input"})
    out = out.sort_values("Time_input").reset_index(drop=True)
    return out



def _time_grid_from_obs(t_obs_h, density=10.0):
    t_obs_h = np.asarray(t_obs_h, dtype=float)
    tmax = float(np.nanmax(t_obs_h)) if len(t_obs_h) else 1.0
    n = int(max(400, min(3000, tmax * density + 1)))
    return np.unique(np.r_[0.0, np.linspace(0.0, tmax, n), t_obs_h])



def _auc_trap(t_h, c):
    t_h = np.asarray(t_h, dtype=float)
    c = np.asarray(c, dtype=float)
    auc = np.zeros_like(t_h, dtype=float)
    for i in range(1, len(t_h)):
        auc[i] = auc[i - 1] + 0.5 * (c[i] + c[i - 1]) * (t_h[i] - t_h[i - 1])
    return auc



def wagner_nelson_fraction(t_h, cp, kel):
    t_h = np.asarray(t_h, dtype=float)
    cp = np.clip(np.asarray(cp, dtype=float), 0.0, None)
    if len(t_h) < 2 or kel <= 0:
        return np.full_like(t_h, np.nan, dtype=float)
    auc = _auc_trap(t_h, cp)
    auc_inf = auc[-1] + cp[-1] / max(kel, 1e-12)
    denom = max(kel * auc_inf, 1e-12)
    frac = (cp + kel * auc) / denom
    return np.clip(frac, 0.0, 1.0)



def fit_pk_deconvolution_suite(pk_df, time_unit_label, disposition, parameter_tables=None, model_choice=None, progress_callback=None):
    factor = TIME_UNIT_TO_HOURS[time_unit_label]
    t_in = pk_df["Time_input"].to_numpy(dtype=float)
    t_h = t_in * factor
    cp_cols = [c for c in pk_df.columns if c != "Time_input"]
    mean_cp = pk_df[cp_cols].mean(axis=1, skipna=True).to_numpy(dtype=float)
    selected_models = _resolve_model_choices(model_choice)
    results = []
    fits = {}
    total_steps = max(1, len(selected_models) + 2)
    step = 0
    _progress_update(progress_callback, step, total_steps, "Preparing PK study summaries")
    pk_tables = build_pk_study_tables(pk_df, time_unit_label, disposition["cp_unit"])
    for model_name in selected_models:
        step += 1
        _progress_update(progress_callback, step, total_steps, f"Fitting {model_name} to the mean PK profile")
        fit = fit_pk_deconvolution_model(t_h, mean_cp, model_name, disposition, parameter_table=(None if parameter_tables is None else parameter_tables.get(model_name)))
        fits[model_name] = fit
        results.append({"Model": model_name, "AIC": fit["aic"], "BIC": fit["bic"], "RSS": fit["rss"], "R²": fit["r2"]})
    summary_df = pd.DataFrame(results).sort_values("AIC").reset_index(drop=True)
    best_model = summary_df.iloc[0]["Model"]
    best_fit = fits[best_model]
    step += 1
    _progress_update(progress_callback, step, total_steps, f"Summarizing best PK deconvolution model ({best_model})")
    if disposition["compartments"] == 1:
        wn = wagner_nelson_fraction(t_h, mean_cp, disposition["k10"])
    else:
        wn = np.full_like(t_h, np.nan, dtype=float)
    detail_rows = []
    for idx, pname in enumerate(best_fit["display_names"]):
        detail_rows.append({"Parameter": pname, "Estimate": best_fit["params"][idx], "S.E.": best_fit["se"][idx], "R.S.E (%)": best_fit["rse_pct"][idx], "t-Value": best_fit["t_value"][idx], "p-Value": best_fit["p_value"][idx], "95% LCL": best_fit["lcl"][idx], "95% UCL": best_fit["ucl"][idx]})
    disp_df = pd.DataFrame([{
        "Compartments": disposition["compartments"], "Dose": disposition["dose_value"], "Dose units": disposition["dose_unit"], "V": disposition["V_value"], "V units": disposition["V_unit"], "Cp units": disposition["cp_unit"], "BIO": disposition["bio"],
        "k10": disposition["k10"], "k12": disposition.get("k12", 0.0), "k21": disposition.get("k21", 0.0), "k13": disposition.get("k13", 0.0), "k31": disposition.get("k31", 0.0),
    }])
    return {
        "input_df": pk_df.copy(), "time_factor": factor, "time_unit_label": time_unit_label, "cp_cols": cp_cols,
        "mean_pk_df": pd.DataFrame({"Time_input": t_in, "Time_h": t_h, "Mean Cp": mean_cp}),
        "summary_df": summary_df, "best_model": best_model, "best_fit": best_fit, "parameter_df": pd.DataFrame(detail_rows),
        "wn_df": pd.DataFrame({"Time_input": t_in, "Time_h": t_h, "Wagner-Nelson fraction": wn}),
        "disposition": disposition, "disposition_df": disp_df,
        "pk_individual_df": pk_tables["individual_df"], "pk_mean_summary_df": pk_tables["mean_summary_df"], "pk_mean_profile_df": pk_tables["mean_profile_df"],
        "model_names": selected_models,
        "parameter_tables_used": {m: fits[m].get("editor_table") for m in selected_models if m in fits},
    }



def plot_deconvolution_pk_fit(pack):
    cfg = safe_get_plot_cfg("Dissolution comparison")
    fig, ax = plt.subplots(figsize=(cfg["fig_w"], cfg["fig_h"]))
    df = pack["input_df"]
    t = df["Time_input"].to_numpy(dtype=float)
    for col in pack["cp_cols"]:
        ax.plot(t, df[col].to_numpy(dtype=float), color=cfg["secondary_color"], alpha=0.25, linewidth=max(0.8, cfg["aux_line_width"]))
    mean_df = pack["mean_pk_df"]
    ax.plot(mean_df["Time_input"], mean_df["Mean Cp"], marker="o", color=cfg["primary_color"], linewidth=cfg["line_width"], label="Observed mean Cp")
    pred = pack["best_fit"]["yhat"]
    ax.plot(mean_df["Time_input"], pred, color=cfg["tertiary_color"], linewidth=cfg["line_width"] + 0.8, label=f"Fitted {pack['best_model']} Cp")
    apply_ax_style(ax, f"Observed and fitted PK profile ({pack['best_model']})", f"Time ({pack['time_unit_label']})", f"Cp ({pack['disposition']['cp_unit']})", legend=True, plot_key="Dissolution comparison")
    return fig


def plot_deconvoluted_profile(pack):
    cfg = safe_get_plot_cfg("Dissolution comparison")
    fig, ax = plt.subplots(figsize=(cfg["fig_w"], cfg["fig_h"]))
    pred_pack = pack["best_fit"]["pred_pack"]
    if pack["disposition"]["compartments"] == 1 and np.isfinite(pack["wn_df"]["Wagner-Nelson fraction"]).any():
        wn = pack["wn_df"].copy()
        ax.plot(wn["Time_input"], wn["Wagner-Nelson fraction"] * 100.0, marker="o", linestyle="None", color=cfg["primary_color"], label="Wagner–Nelson reference")
    ax.plot(pred_pack["t_grid_h"] / pack["time_factor"], pred_pack["cumfrac_grid"] * 100.0, color=cfg["tertiary_color"], linewidth=cfg["line_width"] + 0.8, label=f"Recovered {pack['best_model']} release")
    title = f"Recovered in vivo release profile ({pack['best_model']})"
    apply_ax_style(ax, title, f"Time ({pack['time_unit_label']})", "% released / absorbed", legend=True, plot_key="Dissolution comparison")
    return fig



def plot_pk_mean_profile_errorbars(pack, title="PK mean profile with error bars"):
    cfg = safe_get_plot_cfg("Dissolution comparison")
    fig, ax = plt.subplots(figsize=(cfg["fig_w"], cfg["fig_h"]))
    df = pack["pk_mean_profile_df"]
    yerr = df["SE"].to_numpy(dtype=float)
    yerr = np.where(np.isfinite(yerr), yerr, 0.0)
    ax.errorbar(df["Time_input"], df["Mean Cp"], yerr=yerr, fmt="o-", capsize=3, color=cfg["primary_color"], linewidth=cfg["line_width"], label="Mean Cp ± SE")
    apply_ax_style(ax, title, f"Time ({pack['time_unit_label']})", f"Cp ({pack['disposition']['cp_unit']})", legend=True, plot_key="Dissolution comparison")
    return fig



def plot_pk_individual_profiles(pack, title="Individual PK profiles"):
    cfg = safe_get_plot_cfg("Dissolution comparison")
    fig, ax = plt.subplots(figsize=(cfg["fig_w"], cfg["fig_h"]))
    df = pack["input_df"]
    t = df["Time_input"].to_numpy(dtype=float)
    for col in pack["cp_cols"]:
        ax.plot(t, df[col].to_numpy(dtype=float), marker="o", linewidth=max(0.9, cfg["aux_line_width"]), alpha=0.85, label=col)
    apply_ax_style(ax, title, f"Time ({pack['time_unit_label']})", f"Cp ({pack['disposition']['cp_unit']})", legend=True, plot_key="Dissolution comparison")
    return fig



def save_invivofit_to_session(pack):
    best_model = pack["best_model"]
    best_fit = pack["best_fit"]
    pred = best_fit.get("pred_pack", {})
    st.session_state["InVivoFit"] = {
        "name": "InVivoFit",
        "model": best_model,
        "stored_time_units": "Hours",
        "parameter_order": best_fit["param_names"],
        "parameter_display": best_fit["display_names"],
        "parameter_estimates": {k: float(v) for k, v in zip(best_fit["param_names"], best_fit["params"])},
        "model_comparison": pack["summary_df"].to_dict(orient="records"),
        "disposition": pack["disposition"],
        "time_h": np.asarray(pred.get("t_grid_h", []), dtype=float).tolist(),
        "cumfrac": np.asarray(pred.get("cumfrac_grid", []), dtype=float).tolist(),
        "kab": np.asarray(pred.get("kab_grid", []), dtype=float).tolist(),
        "mean_pk_time_h": np.asarray(pack.get("mean_pk_df", pd.DataFrame()).get("Time_h", []), dtype=float).tolist(),
        "mean_pk_cp": np.asarray(pack.get("mean_pk_df", pd.DataFrame()).get("Mean Cp", []), dtype=float).tolist(),
    }


def _evaluate_saved_invivo_cumfrac(t_h, saved_invivo):
    t_h = np.asarray(t_h, dtype=float)
    if not saved_invivo:
        return np.full_like(t_h, np.nan, dtype=float)
    if saved_invivo.get("time_h") and saved_invivo.get("cumfrac"):
        src_t = np.asarray(saved_invivo.get("time_h", []), dtype=float)
        src_y = np.asarray(saved_invivo.get("cumfrac", []), dtype=float)
        if len(src_t) >= 2 and len(src_t) == len(src_y):
            return np.clip(np.interp(t_h, src_t, src_y), 0.0, 1.0)
    model_name = saved_invivo.get("model")
    order = saved_invivo.get("parameter_order", [])
    param_map = saved_invivo.get("parameter_estimates", {})
    if model_name in MODEL_SPECS and order:
        params = [param_map[k] for k in order]
        return _cumfrac_weibull_fmax(model_name, t_h, params)
    return np.full_like(t_h, np.nan, dtype=float)


def _ivivc_default_bounds(t_h, use_paper_defaults=True, fit_bio=False):
    t_h = np.asarray(t_h, dtype=float)
    tmax = float(np.nanmax(t_h)) if len(t_h) else 1.0
    if use_paper_defaults:
        p0 = [1.0, 1.0]
        lb = [0.01, 0.20]
        ub = [10.0, 5.0]
        names = ["B2", "B3"]
    else:
        p0 = [0.0, 1.0, 0.0, 1.0, 1.0]
        lb = [-0.5, 0.0, -tmax, 0.01, 0.20]
        ub = [1.0, 2.0, tmax, 10.0, 5.0]
        names = ["A1", "A2", "B1", "B2", "B3"]
    if fit_bio:
        p0 += [1.0]
        lb += [0.01]
        ub += [1.5]
        names += ["BIO"]
    return np.asarray(p0, float), np.asarray(lb, float), np.asarray(ub, float), names


def _ivivc_expand_params(params, use_paper_defaults=True, fit_bio=False, fixed_bio=1.0):
    p = np.asarray(params, dtype=float)
    idx = 0
    if use_paper_defaults:
        A1, A2, B1 = 0.0, 1.0, 0.0
        B2, B3 = p[idx], p[idx + 1]
        idx += 2
    else:
        A1, A2, B1, B2, B3 = p[idx:idx + 5]
        idx += 5
    BIO = p[idx] if fit_bio else fixed_bio
    return float(A1), float(A2), float(B1), float(B2), float(B3), float(BIO)


def _simulate_pk_ode_from_cumfrac_grid(t_obs_h, t_grid_h, cumfrac_grid, disposition):
    t_obs_h = np.asarray(t_obs_h, dtype=float)
    t_grid_h = np.asarray(t_grid_h, dtype=float)
    cumfrac_grid = np.clip(np.asarray(cumfrac_grid, dtype=float), 0.0, 1.0)
    if len(t_grid_h) < 2:
        raise ValueError("Time grid for IVIVC simulation must contain at least two points.")
    kab_grid = np.gradient(cumfrac_grid, t_grid_h, edge_order=2)
    kab_grid = np.clip(kab_grid, 0.0, None)

    def _rhs_from_grid(t, y, disposition_local, t_grid_local, kab_grid_local):
        kab = float(np.interp(t, t_grid_local, kab_grid_local, left=kab_grid_local[0], right=kab_grid_local[-1]))
        input_mass = float(disposition_local["dose_mg"]) * float(disposition_local.get("bio", 1.0)) * kab
        comps = int(disposition_local["compartments"])
        k10 = float(disposition_local["k10"])
        a1 = y[0]
        if comps == 1:
            da1 = input_mass - k10 * a1
            return [da1]
        k12 = float(disposition_local.get("k12", 0.0))
        k21 = float(disposition_local.get("k21", 0.0))
        a2 = y[1]
        if comps == 2:
            da1 = input_mass - (k10 + k12) * a1 + k21 * a2
            da2 = k12 * a1 - k21 * a2
            return [da1, da2]
        k13 = float(disposition_local.get("k13", 0.0))
        k31 = float(disposition_local.get("k31", 0.0))
        a3 = y[2]
        da1 = input_mass - (k10 + k12 + k13) * a1 + k21 * a2 + k31 * a3
        da2 = k12 * a1 - k21 * a2
        da3 = k13 * a1 - k31 * a3
        return [da1, da2, da3]

    comps = int(disposition["compartments"])
    sol = solve_ivp(
        _rhs_from_grid,
        (0.0, float(t_grid_h[-1])),
        np.zeros(comps, dtype=float),
        t_eval=t_grid_h,
        args=(disposition, t_grid_h, kab_grid),
        method="LSODA",
        rtol=1e-7,
        atol=1e-9,
    )
    if not sol.success:
        raise ValueError("ODE solver failed while simulating the IVIVC PK profile.")
    a1_grid = sol.y[0]
    cp_grid = _mg_per_l_to_cp_unit(a1_grid / max(float(disposition["V_L"]), 1e-12), disposition["cp_unit"])
    cp_obs = np.interp(t_obs_h, t_grid_h, cp_grid)
    return cp_obs, cp_grid, kab_grid, sol.y


def _predict_ivivc_pk(t_h, saved_invitro, params, disposition, use_paper_defaults=True, fit_bio=False, fixed_bio=1.0):
    A1, A2, B1, B2, B3, BIO = _ivivc_expand_params(params, use_paper_defaults=use_paper_defaults, fit_bio=fit_bio, fixed_bio=fixed_bio)
    t_grid = _time_grid_from_obs(t_h)
    t_scaled = np.clip(B1 + B2 * np.power(np.clip(t_grid, 0.0, None), B3), 0.0, None)
    vitro_pct = _evaluate_saved_invitro_dissolution_percent(t_scaled, saved_invitro)
    tx1 = np.clip(1.0 - vitro_pct / 100.0, 0.0, 1.0)
    vabs = np.clip(1.0 - (A1 + A2 * tx1), 0.0, 1.0)
    disp = dict(disposition)
    disp["bio"] = BIO
    cp_obs, cp_grid, kab_grid, state_grid = _simulate_pk_ode_from_cumfrac_grid(t_h, t_grid, vabs, disp)
    return {
        "t_grid_h": t_grid,
        "t_scaled_h": t_scaled,
        "tx1_grid": tx1,
        "cumfrac_grid": vabs,
        "cp_obs": cp_obs,
        "cp_grid": cp_grid,
        "kab_grid": kab_grid,
        "state_grid": state_grid,
        "A1": A1,
        "A2": A2,
        "B1": B1,
        "B2": B2,
        "B3": B3,
        "BIO": BIO,
    }


def _ivivc_residuals(params, t_h, y, saved_invitro, disposition, use_paper_defaults=True, fit_bio=False, fixed_bio=1.0):
    pred = _predict_ivivc_pk(t_h, saved_invitro, params, disposition, use_paper_defaults=use_paper_defaults, fit_bio=fit_bio, fixed_bio=fixed_bio)["cp_obs"]
    return pred - y


def _ivivc_residuals_release(params, t_h, target_cumfrac, saved_invitro, disposition, use_paper_defaults=True, fit_bio=False, fixed_bio=1.0):
    pred_pack = _predict_ivivc_pk(t_h, saved_invitro, params, disposition, use_paper_defaults=use_paper_defaults, fit_bio=fit_bio, fixed_bio=fixed_bio)
    pred = np.interp(np.asarray(t_h, dtype=float), pred_pack["t_grid_h"], pred_pack["cumfrac_grid"])
    return pred - np.asarray(target_cumfrac, dtype=float)


def fit_ivivc_tool(pk_df, time_unit_label, saved_invitro, disposition, use_paper_defaults=True, fit_bio=False, fixed_bio=1.0, saved_invivo=None):
    factor = TIME_UNIT_TO_HOURS[time_unit_label]
    t_in = pk_df["Time_input"].to_numpy(dtype=float)
    t_h = t_in * factor
    cp_cols = [c for c in pk_df.columns if c != "Time_input"]
    mean_cp = pk_df[cp_cols].mean(axis=1, skipna=True).to_numpy(dtype=float)

    release_target = _evaluate_saved_invivo_cumfrac(t_h, saved_invivo)
    use_saved_invivo = saved_invivo is not None and np.isfinite(release_target).any()
    fit_bio_effective = bool(fit_bio) and not use_saved_invivo

    p0, lb, ub, pnames = _ivivc_default_bounds(t_h, use_paper_defaults=use_paper_defaults, fit_bio=fit_bio_effective)
    starts = [p0.copy(), np.clip(p0 * np.array([0.7] * len(p0)), lb + 1e-12, ub - 1e-12), np.clip(p0 * np.array([1.3] * len(p0)), lb + 1e-12, ub - 1e-12)]
    best = None
    for start in starts:
        try:
            if use_saved_invivo:
                res = least_squares(
                    _ivivc_residuals_release,
                    x0=start,
                    bounds=(lb, ub),
                    args=(t_h, release_target, saved_invitro, disposition, use_paper_defaults, fit_bio_effective, fixed_bio),
                    max_nfev=50000,
                    method="trf",
                )
            else:
                res = least_squares(
                    _ivivc_residuals,
                    x0=start,
                    bounds=(lb, ub),
                    args=(t_h, mean_cp, saved_invitro, disposition, use_paper_defaults, fit_bio_effective, fixed_bio),
                    max_nfev=50000,
                    method="trf",
                )
            if not res.success:
                continue
            pred_pack = _predict_ivivc_pk(t_h, saved_invitro, res.x, disposition, use_paper_defaults=use_paper_defaults, fit_bio=fit_bio_effective, fixed_bio=fixed_bio)
            if use_saved_invivo:
                yhat = np.interp(t_h, pred_pack["t_grid_h"], pred_pack["cumfrac_grid"])
                target = np.asarray(release_target, dtype=float)
                target_label = "Saved InVivoFit release profile"
            else:
                yhat = pred_pack["cp_obs"]
                target = mean_cp
                target_label = "Observed PK profile"
            rss = float(np.sum((target - yhat) ** 2))
            n = len(target)
            k = len(res.x)
            aic = n * np.log(max(rss, 1e-12) / n) + 2 * k
            bic = n * np.log(max(rss, 1e-12) / n) + k * np.log(max(n, 1))
            tss = float(np.sum((target - np.mean(target)) ** 2))
            r2 = 1.0 - rss / tss if tss > 0 else np.nan
            if (best is None) or (aic < best["aic"]):
                best = {
                    "params": np.asarray(res.x, dtype=float),
                    "pred_pack": pred_pack,
                    "rss": rss,
                    "aic": float(aic),
                    "bic": float(bic),
                    "r2": float(r2) if np.isfinite(r2) else np.nan,
                    "param_names": pnames,
                    "init": p0,
                    "lb": lb,
                    "ub": ub,
                    "fit_target": target_label,
                    "fit_target_values": np.asarray(target, dtype=float),
                    "fit_target_predictions": np.asarray(yhat, dtype=float),
                    "used_saved_invivo": use_saved_invivo,
                    "fit_bio_effective": fit_bio_effective,
                }
        except Exception:
            continue
    if best is None:
        raise ValueError("IVIVC fit did not converge for the available saved models and input PK data.")
    rows = []
    for name, est, init, lo, hi in zip(best["param_names"], best["params"], best["init"], best["lb"], best["ub"]):
        rows.append({"Parameter": name, "Estimate": est, "Initial": init, "Min": lo, "Max": hi})
    if disposition["compartments"] == 1:
        wn = wagner_nelson_fraction(t_h, mean_cp, disposition["k10"])
    else:
        wn = np.full_like(t_h, np.nan, dtype=float)
    pk_tables = build_pk_study_tables(pk_df, time_unit_label, disposition["cp_unit"])
    ref_release_df = pd.DataFrame({
        "Time_input": t_in,
        "Time_h": t_h,
        "Saved InVivoFit release (%)": release_target * 100.0 if use_saved_invivo else np.full_like(t_h, np.nan, dtype=float),
        "IVIVC transformed in vitro (%)": np.interp(t_h, best["pred_pack"]["t_grid_h"], best["pred_pack"]["cumfrac_grid"]) * 100.0,
    })
    return {
        "input_df": pk_df.copy(),
        "time_factor": factor,
        "time_unit_label": time_unit_label,
        "cp_cols": cp_cols,
        "mean_pk_df": pd.DataFrame({"Time_input": t_in, "Time_h": t_h, "Mean Cp": mean_cp}),
        "fit_stats_df": pd.DataFrame([{"Fit target": best["fit_target"], "AIC": best["aic"], "BIC": best["bic"], "RSS": best["rss"], "R²": best["r2"]}]),
        "param_df": pd.DataFrame(rows),
        "wn_df": pd.DataFrame({"Time_input": t_in, "Time_h": t_h, "Wagner-Nelson fraction": wn}),
        "reference_release_df": ref_release_df,
        "fit": best,
        "saved_invitro_model": saved_invitro["model"],
        "saved_invivo_model": None if saved_invivo is None else saved_invivo.get("model", "InVivoFit"),
        "used_saved_invivo": use_saved_invivo,
        "disposition": disposition,
        "disposition_df": pd.DataFrame([{
            "Compartments": disposition["compartments"],
            "Dose": disposition["dose_value"],
            "Dose units": disposition["dose_unit"],
            "V": disposition["V_value"],
            "V units": disposition["V_unit"],
            "Cp units": disposition["cp_unit"],
            "k10": disposition["k10"],
            "k12": disposition.get("k12", 0.0),
            "k21": disposition.get("k21", 0.0),
            "k13": disposition.get("k13", 0.0),
            "k31": disposition.get("k31", 0.0),
        }]),
        "pk_individual_df": pk_tables["individual_df"],
        "pk_mean_summary_df": pk_tables["mean_summary_df"],
        "pk_mean_profile_df": pk_tables["mean_profile_df"],
    }


def plot_ivivc_pk_fit(pack):
    cfg = safe_get_plot_cfg("Dissolution comparison")
    fig, ax = plt.subplots(figsize=(cfg["fig_w"], cfg["fig_h"]))
    df = pack["input_df"]
    t = df["Time_input"].to_numpy(dtype=float)
    for col in pack["cp_cols"]:
        ax.plot(t, df[col].to_numpy(dtype=float), color=cfg["secondary_color"], alpha=0.25, linewidth=max(0.8, cfg["aux_line_width"]))
    mean_df = pack["mean_pk_df"]
    ax.plot(mean_df["Time_input"], mean_df["Mean Cp"], marker="o", color=cfg["primary_color"], linewidth=cfg["line_width"], label="Observed mean Cp")
    ax.plot(mean_df["Time_input"], pack["fit"]["pred_pack"]["cp_obs"], color=cfg["tertiary_color"], linewidth=cfg["line_width"] + 0.8, label="IVIVC fitted Cp")
    apply_ax_style(ax, f"Observed and IVIVC-fitted PK profile ({pack['saved_invitro_model']})", f"Time ({pack['time_unit_label']})", f"Cp ({pack['disposition']['cp_unit']})", legend=True, plot_key="Dissolution comparison")
    return fig


def plot_ivivc_deconv(pack):
    cfg = safe_get_plot_cfg("Dissolution comparison")
    fig, ax = plt.subplots(figsize=(cfg["fig_w"], cfg["fig_h"]))
    pred = pack["fit"]["pred_pack"]
    if pack.get("used_saved_invivo", False):
        ref_df = pack.get("reference_release_df", pd.DataFrame())
        if not ref_df.empty:
            ax.plot(ref_df["Time_input"], ref_df["Saved InVivoFit release (%)"], marker="o", linestyle="None", color=cfg["primary_color"], label="Saved InVivoFit release")
    elif pack["disposition"]["compartments"] == 1 and np.isfinite(pack["wn_df"]["Wagner-Nelson fraction"]).any():
        wn = pack["wn_df"]
        ax.plot(wn["Time_input"], wn["Wagner-Nelson fraction"] * 100.0, marker="o", linestyle="None", color=cfg["primary_color"], label="Wagner–Nelson reference")
    ax.plot(pred["t_grid_h"] / pack["time_factor"], pred["cumfrac_grid"] * 100.0, color=cfg["tertiary_color"], linewidth=cfg["line_width"] + 0.8, label="IVIVC transformed in vitro")
    apply_ax_style(ax, "Recovered in vivo release and IVIVC fit", f"Time ({pack['time_unit_label']})", "% absorbed / released", legend=True, plot_key="Dissolution comparison")
    return fig


def save_ivivc_to_session(pack):
    fit = pack["fit"]
    pred = fit["pred_pack"]
    st.session_state["IVIVCModel"] = {
        "name": "IVIVCModel",
        "source_invitro_model": pack["saved_invitro_model"],
        "source_invivo_model": pack.get("saved_invivo_model"),
        "stored_time_units": "Hours",
        "parameter_estimates": {k: float(v) for k, v in zip(fit["param_names"], fit["params"])},
        "fit_statistics": pack["fit_stats_df"].to_dict(orient="records"),
        "expanded_parameters": {"A1": pred["A1"], "A2": pred["A2"], "B1": pred["B1"], "B2": pred["B2"], "B3": pred["B3"], "BIO": pred["BIO"]},
        "disposition": pack["disposition"],
        "used_saved_invivo": pack.get("used_saved_invivo", False),
    }


def render():
    render_display_settings()
    st.sidebar.title("💊 lm Stats")
    st.sidebar.markdown("IVIVC Suite")
    tool = st.sidebar.radio("IVIVC tool", ["💊 Dissolution Comparison (f₂)", "📈 In Vitro Weibull Fit", "🧬 Deconvolution through convolution", "🔗 IVIVC"], key="ivivc_tool")
    st.sidebar.caption("This page contains dissolution similarity, in vitro fitting, convolution-based deconvolution, and IVIVC tools for formulation and PK workflows.")

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

    elif tool == "🧬 Deconvolution through convolution":
        app_header("🧬 Deconvolution through convolution", "Fit single-, double-, and triple-Weibull in vivo release models directly to PK data through the analytical ODE workflow, choose the best model by AIC, and optionally save it in-session as InVivoFit.")
        _reset_deconv_editor_state(force=False)

        c1, c2, c3 = st.columns([1, 1, 6])
        with c1:
            st.button("Sample Data", key="sample_pk_deconv", on_click=load_pk_deconv_sample)
        with c2:
            if st.button("Reset Tables", key="reset_pk_deconv_tables"):
                _reset_deconv_editor_state(force=True)
                st.session_state.pop("deconv_last_pack", None)
        with c3:
            pk_text = st.text_area("PK table (first column = time, remaining columns = one or more Cp profiles)", height=260, key="pk_input_deconv")

        top1 = st.columns(6)
        with top1[0]:
            time_unit_label = st.selectbox("Time units", ["Minutes", "Hours", "Days"], index=1, key="deconv_time_units")
        with top1[1]:
            cp_unit = st.selectbox("Cp units", list(CP_MG_PER_L_TO_UNIT.keys()), index=2, key="deconv_cp_units")
        with top1[2]:
            compartments = st.selectbox("Compartments", [1, 2, 3], index=2, key="deconv_compartments")
        with top1[3]:
            dose_value = st.number_input("Dose", min_value=0.000001, value=6666666.600, format="%.6f", key="deconv_dose_value")
        with top1[4]:
            dose_unit = st.selectbox("Dose units", list(DOSE_UNIT_TO_MG.keys()), index=0, key="deconv_dose_unit")
        with top1[5]:
            model_choice = st.selectbox("Weibull model(s)", MODEL_CHOICE_OPTIONS, index=3, key="deconv_model_choice")

        top2 = st.columns(6)
        with top2[0]:
            v_value = st.number_input("V", min_value=0.000001, value=1136.900, format="%.6f", key="deconv_v_value")
        with top2[1]:
            v_unit = st.selectbox("V units", list(VOLUME_UNIT_TO_L.keys()), index=1, key="deconv_v_unit")
        with top2[2]:
            k10 = st.number_input("k10 (1/h)", min_value=0.000001, value=0.770000, format="%.6f", key="deconv_k10")
        with top2[3]:
            k12 = st.number_input("k12 (1/h)", min_value=0.0, value=1.382000, format="%.6f", key="deconv_k12", disabled=compartments < 2)
        with top2[4]:
            k21 = st.number_input("k21 (1/h)", min_value=0.0, value=1.814000, format="%.6f", key="deconv_k21", disabled=compartments < 2)
        with top2[5]:
            decimals = st.slider("Decimals", 1, 8, 3, key="deconv_decimals")

        top3 = st.columns(6)
        with top3[0]:
            k13 = st.number_input("k13 (1/h)", min_value=0.0, value=1.000000, format="%.6f", key="deconv_k13", disabled=compartments < 3)
        with top3[1]:
            k31 = st.number_input("k31 (1/h)", min_value=0.0, value=0.000000, format="%.6f", key="deconv_k31", disabled=compartments < 3)
        with top3[2]:
            run_deconv = st.button("Run Deconvolution Fit", type="primary", key="run_deconv_fit")
        with top3[3]:
            st.empty()
        with top3[4]:
            st.empty()
        with top3[5]:
            st.empty()

        st.caption("The selected input time unit is converted internally so all fitting and stored parameters are in hours. The PK model uses the analytical Weibull input-rate directly in the compartment ODE system. The default sample data, microconstants, dose, volume, and starting values match the values you provided for quick checking.")

        parameter_tables = {}
        with st.expander("Parameter bounds, starting values, and fixed parameters", expanded=True):
            st.write("Edit the starting values, bounds, and fixed-parameter flags for each Weibull model. MDT values are always handled in hours.")
            for model_name in MODEL_SPECS:
                st.markdown(f"**{model_name}**")
                table_key = _deconv_editor_state_key(model_name)
                current_df = st.session_state.get(table_key, build_deconv_parameter_tables(np.asarray([0.0, 1.0]), np.asarray([0.0, 1.0]))[model_name]).copy()
                edited_df = st.data_editor(
                    current_df,
                    key=f"deconv_editor_{_slugify(model_name)}",
                    hide_index=True,
                    num_rows="fixed",
                    use_container_width=True,
                    disabled=["Parameter"],
                    column_config={
                        "Parameter": st.column_config.TextColumn("Parameter", disabled=True),
                        "Initial Value": st.column_config.NumberColumn("Initial Value", format="%.6f"),
                        "Min (≥)": st.column_config.NumberColumn("Min (≥)", format="%.6f"),
                        "Max (≤)": st.column_config.NumberColumn("Max (≤)", format="%.6f"),
                        "Fix": st.column_config.CheckboxColumn("Fix"),
                    },
                )
                st.session_state[table_key] = pd.DataFrame(edited_df)
                parameter_tables[model_name] = st.session_state[table_key].copy()

        if run_deconv:
            if not pk_text or not str(pk_text).strip():
                st.warning("Please provide PK data or press 'Sample Data' first.")
                st.session_state.pop("deconv_last_pack", None)
            else:
                try:
                    pk_df = parse_pk_profile_table(pk_text)
                    disposition = _build_disposition_config(compartments, dose_value, dose_unit, v_value, v_unit, cp_unit, k10, k12, k21, k13, k31, bio=1.0)
                    progress_holder = st.empty()
                    status_holder = st.empty()
                    progress_bar = progress_holder.progress(0)

                    def _cb(step, total, msg):
                        progress_bar.progress(max(0.0, min(1.0, float(step) / max(float(total), 1.0))))
                        status_holder.caption(msg)

                    pack = fit_pk_deconvolution_suite(pk_df, time_unit_label, disposition, parameter_tables=parameter_tables, model_choice=model_choice, progress_callback=_cb)
                    st.session_state["deconv_last_pack"] = pack
                    progress_bar.progress(1.0)
                    status_holder.caption(f"Finished PK deconvolution fit. Best model: {pack['best_model']}.")
                except Exception as e:
                    st.session_state.pop("deconv_last_pack", None)
                    st.error(str(e))

        pack = st.session_state.get("deconv_last_pack")
        if pack is not None:
            fig_pk = plot_deconvolution_pk_fit(pack)
            fig_deconv = plot_deconvoluted_profile(pack)
            fig_mean_pk = plot_pk_mean_profile_errorbars(pack, title="PK mean profile with error bars")
            fig_individual_pk = plot_pk_individual_profiles(pack, title="Individual PK profiles")
            m1, m2, m3 = st.columns(3)
            m1.metric("Best model", pack["best_model"])
            m2.metric("Best AIC", f"{pack['summary_df'].iloc[0]['AIC']:.{decimals}f}")
            m3.metric("Saved model key", "InVivoFit")

            t1, t2 = st.tabs(["PK study", "Save model"])
            with t1:
                inp = pack["input_df"].copy()
                inp.insert(1, "Time_h", inp["Time_input"] * pack["time_factor"])
                report_table(inp, "Input PK data used in the convolution-through-ODE fit", decimals)
                report_table(pack["pk_individual_df"], "Individual-subject PK summary", decimals)
                report_table(pack["pk_mean_summary_df"], "Mean PK summary across profiles", decimals)
                report_table(pack["pk_mean_profile_df"], "PK mean profile with SD and SE", decimals)
                report_table(pack["disposition_df"], "Disposition system used to generate the PK profile", decimals)
                report_table(pack["summary_df"], "AIC comparison for the selected Weibull release models", decimals)
                report_table(pack["parameter_df"], f"Parameter estimates for the best model ({pack['best_model']})", decimals)
                show_figure(fig_individual_pk, caption="Individual PK profiles")
                show_figure(fig_mean_pk, caption="PK mean profile with error bars")
                show_figure(fig_pk, caption=f"Observed and fitted PK profile for the best model ({pack['best_model']})")
                show_figure(fig_deconv, caption="Recovered in vivo release profile for the best model")
            with t2:
                if st.button("Save best model as InVivoFit", key="save_invivofit_button"):
                    save_invivofit_to_session(pack)
                    st.success("The best convolution-based in vivo release model was saved in this session as InVivoFit.")
                if "InVivoFit" in st.session_state:
                    current = st.session_state["InVivoFit"]
                    st.info(f"Current saved in-session model: {current.get('name', 'InVivoFit')} ({current.get('model', '-')}, stored in hours).")

            table_map = {
                "Input PK Data": pack["input_df"].assign(Time_h=pack["input_df"]["Time_input"] * pack["time_factor"]),
                "Individual PK Summary": pack["pk_individual_df"],
                "Mean PK Summary": pack["pk_mean_summary_df"],
                "PK Mean Profile": pack["pk_mean_profile_df"],
                "Disposition System": pack["disposition_df"],
                "Model Comparison": pack["summary_df"],
                f"Best Model Parameters ({pack['best_model']})": pack["parameter_df"],
                "Recovered Release Reference": pack["wn_df"],
            }
            figure_map = {
                "Individual PK profiles": fig_to_png_bytes(fig_individual_pk),
                "PK mean profile with error bars": fig_to_png_bytes(fig_mean_pk),
                f"Observed and fitted PK profile ({pack['best_model']})": fig_to_png_bytes(fig_pk),
                "Recovered in vivo release profile": fig_to_png_bytes(fig_deconv),
            }
            export_results(
                prefix="ivivc_deconvolution_through_convolution",
                report_title="Statistical Analysis Report",
                module_name="Deconvolution through convolution",
                statistical_analysis="Single-, double-, and triple-Weibull in vivo release functions were fitted directly to the PK data. The analytical Weibull input-rate was used directly in the compartment ODE system together with the user-supplied microconstants, dose, and V.",
                offer_text="This module recovers a practical in vivo release function from PK data without first performing a separate standalone deconvolution step.",
                python_tools="pandas, numpy, scipy.optimize.least_squares, scipy.integrate.solve_ivp, scipy.stats, matplotlib, openpyxl, reportlab",
                table_map=table_map,
                figure_map=figure_map,
                conclusion=f"The best convolution-based in vivo release model by AIC was {pack['best_model']}. Review the fitted PK graph and the recovered release profile before saving the model downstream.",
                decimals=decimals,
            )
    elif tool == "🔗 IVIVC":
        app_header("🔗 IVIVC", "Apply the IVIVC time-scaling workflow to the saved InVitroFit model and, when available, use the saved InVivoFit release model as the in vivo target before back-predicting PK through the compartment ODE system.")
        if "InVitroFit" not in st.session_state:
            st.warning("Please run 'In Vitro Weibull Fit' first and save a model as InVitroFit in the current session before using this IVIVC tool.")
        else:
            saved_invitro = st.session_state["InVitroFit"]
            saved_invivo = st.session_state.get("InVivoFit")
            st.info(f"Current saved InVitroFit: {saved_invitro.get('model', '-')} (stored in hours).")
            if saved_invivo is not None:
                st.success(f"Current saved InVivoFit: {saved_invivo.get('model', '-')} (stored in hours). The IVIVC fit will use this saved in vivo release profile as the target.")
            else:
                st.warning("No saved InVivoFit was found in the current session. The IVIVC tool will fall back to fitting against the PK profile directly.")
            c1, c2 = st.columns([1, 6])
            with c1:
                st.button("Sample Data", key="sample_pk_ivivc", on_click=load_pk_sample_text, args=("pk_input_ivivc_tool",))
            with c2:
                pk_text = st.text_area("PK table (first column = time, remaining columns = one or more Cp profiles)", height=260, key="pk_input_ivivc_tool")
            u1, u2, u3 = st.columns([1, 1, 1])
            with u1:
                time_unit_label = st.selectbox("Input time units", ["Minutes", "Hours", "Days"], index=0, key="ivivc_time_units_tool")
            with u2:
                cp_unit_options = list(CP_MG_PER_L_TO_UNIT.keys())
                cp_unit_default = (saved_invivo or {}).get("disposition", {}).get("cp_unit", "ug/mL")
                cp_unit = st.selectbox("Cp units", cp_unit_options, index=(cp_unit_options.index(cp_unit_default) if cp_unit_default in cp_unit_options else 1), key="ivivc_cp_units_tool")
            with u3:
                decimals = st.slider("Decimals", 1, 8, 3, key="ivivc_decimals_tool")
            d1, d2, d3, d4 = st.columns(4)
            with d1:
                comp_options = [1, 2, 3]
                comp_default = int((saved_invivo or {}).get("disposition", {}).get("compartments", 1))
                compartments = st.selectbox("Compartments", comp_options, index=(comp_options.index(comp_default) if comp_default in comp_options else 0), key="ivivc_compartments")
            with d2:
                dose_value = st.number_input("Dose", min_value=0.000001, value=float((saved_invivo or {}).get("disposition", {}).get("dose_value", 20.0)), format="%.6f", key="ivivc_dose_value")
            with d3:
                dose_unit_options = list(DOSE_UNIT_TO_MG.keys())
                dose_unit_default = (saved_invivo or {}).get("disposition", {}).get("dose_unit", "mg")
                dose_unit = st.selectbox("Dose units", dose_unit_options, index=(dose_unit_options.index(dose_unit_default) if dose_unit_default in dose_unit_options else 2), key="ivivc_dose_unit")
            with d4:
                v_value = st.number_input("V", min_value=0.000001, value=float((saved_invivo or {}).get("disposition", {}).get("V_value", 50.0)), format="%.6f", key="ivivc_v_value")
            d5, d6, d7, d8 = st.columns(4)
            with d5:
                v_unit_options = list(VOLUME_UNIT_TO_L.keys())
                v_unit_default = (saved_invivo or {}).get("disposition", {}).get("V_unit", "L")
                v_unit = st.selectbox("V units", v_unit_options, index=(v_unit_options.index(v_unit_default) if v_unit_default in v_unit_options else 2), key="ivivc_v_unit")
            with d6:
                k10 = st.number_input("k10 (1/h)", min_value=0.000001, value=float((saved_invivo or {}).get("disposition", {}).get("k10", 0.254000)), format="%.6f", key="ivivc_k10")
            with d7:
                k12 = st.number_input("k12 (1/h)", min_value=0.0, value=float((saved_invivo or {}).get("disposition", {}).get("k12", 0.100000)), format="%.6f", key="ivivc_k12", disabled=compartments < 2)
            with d8:
                k21 = st.number_input("k21 (1/h)", min_value=0.0, value=float((saved_invivo or {}).get("disposition", {}).get("k21", 0.050000)), format="%.6f", key="ivivc_k21", disabled=compartments < 2)
            d9, d10 = st.columns(2)
            with d9:
                k13 = st.number_input("k13 (1/h)", min_value=0.0, value=float((saved_invivo or {}).get("disposition", {}).get("k13", 0.050000)), format="%.6f", key="ivivc_k13", disabled=compartments < 3)
            with d10:
                k31 = st.number_input("k31 (1/h)", min_value=0.0, value=float((saved_invivo or {}).get("disposition", {}).get("k31", 0.020000)), format="%.6f", key="ivivc_k31", disabled=compartments < 3)
            o1, o2, o3 = st.columns([1.3, 1, 1])
            with o1:
                use_paper_defaults = st.checkbox("Use paper/code defaults (A1 = 0, A2 = 1, B1 = 0)", value=True)
            with o2:
                fit_bio = st.checkbox("Fit BIO", value=True)
            with o3:
                fixed_bio = st.number_input("Fixed BIO", min_value=0.000001, value=1.000000, format="%.6f", key="ivivc_fixed_bio", disabled=fit_bio)
            if saved_invivo is not None:
                st.caption("The saved in vitro Weibull model is transformed through the IVIVC time-scaling function t'' = B1 + B2·t^B3 and is now fitted against the saved InVivoFit release profile. The PK table and the disposition settings are still used to back-predict the PK profile for validation and reporting.")
            else:
                st.caption("The saved in vitro Weibull model is transformed through the paper-style time-scaling function t'' = B1 + B2·t^B3. Because no saved InVivoFit is currently available, the transformed profile is fitted directly against the PK profile through the compartment ODE system.")

            if saved_invivo is not None and fit_bio:
                st.info("Fit BIO is disabled when the objective is the saved InVivoFit release profile, because BIO is not identifiable from the release-vs-release IVIVC mapping alone. The current Fixed BIO value is used only for PK back-prediction.")

            if pk_text:
                try:
                    pk_df = parse_pk_profile_table(pk_text)
                    disposition = _build_disposition_config(compartments, dose_value, dose_unit, v_value, v_unit, cp_unit, k10, k12, k21, k13, k31, bio=(1.0 if fit_bio else fixed_bio))
                    pack = fit_ivivc_tool(pk_df, time_unit_label, saved_invitro, disposition, use_paper_defaults=use_paper_defaults, fit_bio=fit_bio, fixed_bio=fixed_bio, saved_invivo=saved_invivo)
                    fig_pk = plot_ivivc_pk_fit(pack)
                    fig_rel = plot_ivivc_deconv(pack)
                    fig_mean_pk = plot_pk_mean_profile_errorbars(pack, title="PK mean profile with error bars")
                    fig_individual_pk = plot_pk_individual_profiles(pack, title="Individual PK profiles")
                    fit_stats = pack["fit_stats_df"].iloc[0]
                    m1, m2, m3 = st.columns(3)
                    m1.metric("InVitroFit model", pack["saved_invitro_model"])
                    m2.metric("IVIVC target", "Saved InVivoFit" if pack.get("used_saved_invivo", False) else "PK profile")
                    m3.metric("IVIVC R²", f"{fit_stats['R²']:.{decimals}f}")

                    t1, t2 = st.tabs(["PK study", "Save model"])
                    with t1:
                        inp = pk_df.copy()
                        inp.insert(1, "Time_h", pk_df["Time_input"] * pack["time_factor"])
                        report_table(inp, "Input PK data used in the IVIVC fit", decimals)
                        report_table(pack["pk_individual_df"], "Individual-subject PK summary", decimals)
                        report_table(pack["pk_mean_summary_df"], "Mean PK summary across profiles", decimals)
                        report_table(pack["pk_mean_profile_df"], "PK mean profile with SD and SE", decimals)
                        report_table(pack["disposition_df"], "Disposition system used in the IVIVC ODE fit", decimals)
                        report_table(pack["fit_stats_df"], "IVIVC fit statistics", decimals)
                        if pack.get("used_saved_invivo", False):
                            report_table(pack["reference_release_df"], "Saved InVivoFit release profile and IVIVC transformed in vitro release", decimals)
                        report_table(pack["param_df"], "Estimated IVIVC parameters", decimals)
                        show_figure(fig_individual_pk, caption="Individual PK profiles")
                        show_figure(fig_mean_pk, caption="PK mean profile with error bars")
                        show_figure(fig_pk, caption="Observed PK profile and IVIVC-fitted PK profile")
                        show_figure(fig_rel, caption="Recovered in vivo release and transformed in vitro release from the IVIVC fit")
                    with t2:
                        if st.button("Save IVIVC model", key="save_ivivc_model_button"):
                            save_ivivc_to_session(pack)
                            st.success("The IVIVC model was saved in this session as IVIVCModel.")
                        if "IVIVCModel" in st.session_state:
                            current = st.session_state["IVIVCModel"]
                            invivo_src = current.get('source_invivo_model', '-')
                            st.info(f"Current saved in-session model: {current.get('name', 'IVIVCModel')} (source InVitroFit: {current.get('source_invitro_model', '-')}, source InVivoFit: {invivo_src}).")

                    table_map = {
                        "Input PK Data": pk_df.assign(Time_h=pk_df["Time_input"] * pack["time_factor"]),
                        "Individual PK Summary": pack["pk_individual_df"],
                        "Mean PK Summary": pack["pk_mean_summary_df"],
                        "PK Mean Profile": pack["pk_mean_profile_df"],
                        "Disposition System": pack["disposition_df"],
                        "IVIVC Fit Statistics": pack["fit_stats_df"],
                        "Estimated IVIVC Parameters": pack["param_df"],
                        "Recovered Release Reference": pack["wn_df"],
                    }
                    if pack.get("used_saved_invivo", False):
                        table_map["Saved InVivoFit Release vs IVIVC Transformed In Vitro"] = pack["reference_release_df"]
                    figure_map = {
                        "Individual PK profiles": fig_to_png_bytes(fig_individual_pk),
                        "PK mean profile with error bars": fig_to_png_bytes(fig_mean_pk),
                        "Observed and IVIVC-fitted PK profile": fig_to_png_bytes(fig_pk),
                        "Recovered in vivo release and IVIVC fit": fig_to_png_bytes(fig_rel),
                    }
                    export_results(
                        prefix="ivivc_convolution_framework",
                        report_title="Statistical Analysis Report",
                        module_name="IVIVC",
                        statistical_analysis="The saved InVitroFit Weibull model was used as the in vitro dissolution input, transformed through the paper-style time-scaling function, converted to KAB by finite difference, and linked to PK data through the compartment ODE system with user-supplied microconstants, dose, and V.",
                        offer_text="This implementation follows the paper/code logic by fixing the in vitro dissolution model first and then estimating the IVIVC transformation on the PK profile.",
                        python_tools="pandas, numpy, scipy.optimize.least_squares, scipy.integrate.solve_ivp, scipy.stats, matplotlib, openpyxl, reportlab",
                        table_map=table_map,
                        figure_map=figure_map,
                        conclusion="Review the estimated IVIVC parameters, the fitted PK profile, and the recovered-release comparison before using this model for downstream simulations.",
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
        u1, u2, u3 = st.columns([1, 1, 1.4])
        with u1:
            time_unit_label = st.selectbox("Input time units", ["Minutes", "Hours", "Days"], index=0)
        with u2:
            decimals = st.slider("Decimals", 1, 8, 3, key="weibull_dec_ivivc")
        with u3:
            model_choice = st.selectbox("Weibull model(s) to fit", MODEL_CHOICE_OPTIONS, index=0, key="weibull_fit_model_choice")
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
                            _parameter_long_to_wide(model_name, default_tables[model_name]),
                            key=f"weibull_editor_{_slugify(model_name)}",
                            hide_index=True,
                            num_rows="fixed",
                            width='stretch',
                        )
                        parameter_tables[model_name] = _parameter_wide_to_long(model_name, pd.DataFrame(editor))

                progress_holder = st.empty()
                status_holder = st.empty()
                progress_bar = progress_holder.progress(0)
                def _cb(step, total, msg):
                    progress_bar.progress(max(0.0, min(1.0, float(step) / max(float(total), 1.0))))
                    status_holder.caption(msg)
                fit_pack = fit_weibull_suite(fit_df, time_unit_label, parameter_tables=parameter_tables, model_choice=model_choice, progress_callback=_cb)
                progress_bar.progress(1.0)
                status_holder.caption(f"Finished Weibull fitting. Best model: {fit_pack['best_model']}.")
                summary_df = fit_pack["summary_df"]
                per_rep_df = fit_pack["per_rep_df"]
                param_df = fit_pack["param_df"]
                param_df_wide = fit_pack["param_df_wide"]
                single_profile_detail_df = fit_pack["single_profile_detail_df"]
                single_profile_detail_best_wide = fit_pack["single_profile_detail_best_wide"]
                mean_profile_df = fit_pack["mean_profile_df"]
                best_model = fit_pack["best_model"]

                m1, m2, m3 = st.columns(3)
                m1.metric("Best model", best_model)
                m2.metric("Best AIC (mean profile)", f"{summary_df.loc[summary_df['Model'] == best_model, 'Mean-profile AIC'].iloc[0]:.{decimals}f}")
                m3.metric("Profiles fitted", str(len(fit_pack["replicate_cols"])))

                fig_fit = plot_weibull_profile_fits(fit_df, fit_pack, time_unit_label)
                fig_best = plot_best_model_profile(fit_df, fit_pack, time_unit_label)
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
                    report_table(param_df_wide, "Parameter estimates summary", decimals)
                    report_table(mean_profile_df, "Mean dissolution profile used for summary fitting", decimals)
                    show_figure(fig_fit, caption="Observed dissolution profile and Weibull fits")
                    show_figure(fig_aic, caption="Weibull model comparison by AIC")
                    show_figure(fig_resid, caption=f"Residual plot for best model ({best_model})")
                with tabs[1]:
                    report_table(per_rep_df, "Per-profile Weibull fit statistics", decimals)
                if len(fit_pack["replicate_cols"]) == 1:
                    with tabs[2]:
                        report_table(single_profile_detail_best_wide, f"Single-profile parameter statistics from the Jacobian-based covariance calculation for the best model ({best_model})", decimals)
                with tabs[-1]:
                    best_params = _wide_saved_parameter_table(param_df, best_model)
                    report_table(best_params, f"Parameters that will be saved for {best_model}", decimals)
                    show_figure(fig_best, caption=f"Experimental and fitted profile ({best_model})")
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
                    "Parameter Estimates Summary": param_df_wide,
                    "Mean Dissolution Profile": mean_profile_df,
                    f"Parameters Saved for {best_model}": _wide_saved_parameter_table(param_df, best_model),
                }
                if len(fit_pack["replicate_cols"]) == 1 and not single_profile_detail_best_wide.empty:
                    table_map["Single-Profile Parameter Statistics"] = single_profile_detail_best_wide
                figure_map = {
                    "Observed dissolution profile and Weibull fits": fig_to_png_bytes(fig_fit),
                    f"Experimental and fitted profile ({best_model})": fig_to_png_bytes(fig_best),
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
