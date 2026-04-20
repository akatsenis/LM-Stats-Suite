import modules.common as common
from modules.common import *
from scipy.optimize import least_squares
from scipy.integrate import solve_ivp
from scipy.stats import gaussian_kde
from io import StringIO
import numpy as np
import pandas as pd

st = common.st
pd = common.pd
np = common.np
plt = common.plt

# --- Constants ---
TIME_UNIT_TO_HOURS = {"Minutes": 1 / 60.0, "Hours": 1.0, "Days": 24.0}
DOSE_UNIT_TO_MG = {"ng": 1e-6, "ug": 1e-3, "mg": 1.0, "g": 1e3}
VOLUME_UNIT_TO_L = {"uL": 1e-6, "mL": 1e-3, "L": 1.0}
CP_MG_PER_L_TO_UNIT = {"mg/L": 1.0, "ug/mL": 1.0, "ng/mL": 1e3, "ug/L": 1e3, "mg/mL": 1e-3}

PK_SYNTHETIC_SAMPLE = """Time\tCp1\tCp2\tCp3
0\t0\t0\t0
1\t8.94\t18.2\t14.1
6\t12.6\t18.2\t16.1
12\t9.08\t14.4\t11.4
24\t11.7\t14.7\t12
48\t7.86\t12.9\t9.81
96\t10.4\t18.3\t14.5
168\t11.8\t21\t15.3
264\t8.77\t12.3\t10.9
360\t8.57\t6.26\t7.48
456\t3.97\t1.45\t5.43
552\t3.51\t0\t3.23
648\t2.49\t0\t1.28
744\t2.21\t0\t0.564
840\t1.66\t0\t0"""

# --- Analytical Weibull Math ---
def _weibull_rate_unit(t, MDT, b):
    t_safe = np.maximum(t, 1e-12)
    MDT_s = np.maximum(MDT, 1e-12)
    b_s = np.maximum(b, 1e-12)
    return (b_s / MDT_s) * np.power(t_safe / MDT_s, b_s - 1) * np.exp(-np.power(t_safe / MDT_s, b_s))

def get_kab_rate(t, model_name, p):
    Fmax_f = np.clip(p.get('Fmax', 100.0), 0, 100) / 100.0
    if model_name == "Single Weibull":
        return Fmax_f * _weibull_rate_unit(t, p['MDT1'], p['b1'])
    elif model_name == "Double Weibull":
        f1 = np.clip(p.get('f1', 0.5), 0, 1)
        return Fmax_f * (f1 * _weibull_rate_unit(t, p['MDT1'], p['b1']) + (1 - f1) * _weibull_rate_unit(t, p['MDT2'], p['b2']))
    else: # Triple
        f1, f2 = np.clip(p.get('f1', 0.3), 0, 1), np.clip(p.get('f2', 0.3), 0, 1)
        if (f1 + f2) > 1.0:
            s = f1 + f2
            f1 /= s; f2 /= s
        f3 = 1.0 - f1 - f2
        return Fmax_f * (f1 * _weibull_rate_unit(t, p['MDT1'], p['b1']) + f2 * _weibull_rate_unit(t, p['MDT2'], p['b2']) + f3 * _weibull_rate_unit(t, p['MDT3'], p['b3']))

# --- ODE Engine ---
def pk_ode_rhs(t, y, model_name, p_dict, disp):
    kab = get_kab_rate(t, model_name, p_dict)
    input_mass = disp['dose_mg'] * disp['bio'] * kab
    A1 = y[0]
    da1 = input_mass - disp['k10'] * A1
    if disp['compartments'] == 1: return [da1]
    A2 = y[1]
    da1 += disp['k21'] * A2 - disp['k12'] * A1
    da2 = disp['k12'] * A1 - disp['k21'] * A2
    if disp['compartments'] == 2: return [da1, da2]
    A3 = y[2]
    da1 += disp['k31'] * A3 - disp['k13'] * A1
    da3 = disp['k13'] * A1 - disp['k31'] * A3
    return [da1, da2, da3]

# --- Parsing & NCA Helpers ---
def load_pk_sample_text(state_key):
    st.session_state[state_key] = PK_SYNTHETIC_SAMPLE

def parse_pk_profile_table(text):
    df = pd.read_csv(StringIO(text.strip()), sep=r"[\t,; ]+", engine="python")
    for c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=[df.columns[0]])
    time_col = df.columns[0]
    cp_cols = [c for c in df.columns[1:] if df[c].notna().any()]
    out = df[[time_col] + cp_cols].copy().rename(columns={time_col: "Time_input"})
    return out.sort_values("Time_input").reset_index(drop=True)

def build_pk_study_tables(pk_df, time_unit_label, cp_unit):
    # Simplified NCA summary for the report
    t_in = pk_df["Time_input"].values
    cp_cols = [c for c in pk_df.columns if c != "Time_input"]
    mean_cp = pk_df[cp_cols].mean(axis=1)
    sd_cp = pk_df[cp_cols].std(axis=1)
    return pd.DataFrame({"Time": t_in, "Mean": mean_cp, "SD": sd_cp})

# --- Main Render ---
def render():
    render_display_settings()
    st.sidebar.title("💊 lm Stats")
    tool = st.sidebar.radio("IVIVC tool", ["🧬 Deconvolution through convolution", "📈 In Vitro Weibull Fit"])

    if tool == "🧬 Deconvolution through convolution":
        app_header("🧬 Deconvolution through convolution", "Direct ODE Fitting to PK data.")
        
        # UI - Disposition
        with st.expander("1. Disposition & Dose Settings", expanded=True):
            r1c1, r1c2, r1c3, r1c4 = st.columns(4)
            comps = r1c1.selectbox("Compartments", [1, 2, 3], index=1)
            dose_val = r1c2.number_input("Dose", value=6666666.6)
            dose_u = r1c3.selectbox("Dose units", list(DOSE_UNIT_TO_MG.keys()), index=0)
            v_val = r1c4.number_input("V", value=1136.9)
            
            r2c1, r2c2, r2c3, r2c4 = st.columns(4)
            v_u = r2c1.selectbox("V units", list(VOLUME_UNIT_TO_L.keys()), index=1)
            k10 = r2c2.number_input("k10 (1/h)", value=0.77)
            k12 = r2c3.number_input("k12 (1/h)", value=1.38, disabled=comps < 2)
            k21 = r2c4.number_input("k21 (1/h)", value=1.81, disabled=comps < 2)
            
            r3c1, r3c2, r3c3 = st.columns([1, 1, 2])
            k13 = r3c1.number_input("k13 (1/h)", value=0.0, disabled=comps < 3)
            k31 = r3c2.number_input("k31 (1/h)", value=0.0, disabled=comps < 3)
            bio = r3c3.slider("Bioavailability (F)", 0.0, 1.0, 1.0)

        # UI - Data
        c_smpl, c_txt = st.columns([1, 5])
        with c_smpl: st.button("Sample Data", on_click=load_pk_sample_text, args=("pk_input_deconv",))
        pk_text = c_txt.text_area("PK table", height=150, key="pk_input_deconv")
        
        # UI - Weibull Logic
        model_choice = st.selectbox("Select Weibull model", ["Single Weibull", "Double Weibull", "Triple Weibull"], index=2)
        
        param_sets = {
            "Single Weibull": ["MDT1", "b1", "Fmax"],
            "Double Weibull": ["MDT1", "b1", "f1", "MDT2", "b2", "Fmax"],
            "Triple Weibull": ["MDT1", "b1", "f1", "MDT2", "b2", "f2", "MDT3", "b3", "Fmax"]
        }
        active_p = param_sets[model_choice]
        
        # Pre-filled values from your image
        img_vals = {"MDT1":7.48, "b1":0.83, "f1":0.04, "MDT2":29.31, "b2":6.75, "f2":0.02, "MDT3":265.68, "b3":1.57, "Fmax":72.90}
        
        df_params = pd.DataFrame(index=["Initial Value", "Min", "Max", "Fix"], columns=active_p)
        for p in active_p:
            df_params.at["Initial Value", p] = img_vals.get(p, 10.0)
            df_params.at["Min", p] = 0.0
            df_params.at["Max", p] = 1.0 if "f" in p else (100.0 if "Fmax" in p else 2000.0)
            df_params.at["Fix", p] = False

        st.write("Edit Parameter Starting Values & Constraints")
        # Showing in wide format as requested
        edited_df = st.data_editor(df_params, use_container_width=True)

        if st.button("Run Deconvolution Fit") and pk_text:
            try:
                pk_df = parse_pk_profile_table(pk_text)
                t_obs = pk_df["Time_input"].values.astype(float)
                rep_cols = [c for c in pk_df.columns if c != "Time_input"]
                y_obs = pk_df[rep_cols].mean(axis=1).values.astype(float)
                
                disp = {
                    "compartments": comps, "k10": k10, "k12": k12, "k21": k21, "k13": k13, "k31": k31,
                    "dose_mg": dose_val * DOSE_UNIT_TO_MG[dose_u], "V_L": v_val * VOLUME_UNIT_TO_L[v_u], "bio": bio
                }

                # Solve Params
                init_vals = edited_df.loc["Initial Value"].values.astype(float)
                lbs = edited_df.loc["Min"].values.astype(float)
                ubs = edited_df.loc["Max"].values.astype(float)
                fix_mask = edited_df.loc["Fix"].values.astype(bool)
                opt_idx = [i for i, f in enumerate(fix_mask) if not f]

                def objective(x_opt):
                    p_curr = init_vals.copy()
                    p_curr[opt_idx] = x_opt
                    p_dict = dict(zip(active_p, p_curr))
                    # LSODA for stiffness
                    sol = solve_ivp(pk_ode_rhs, (0, t_obs[-1]), [0.0]*comps, t_eval=t_obs, 
                                    args=(model_choice, p_dict, disp), method='LSODA', rtol=1e-7)
                    if not sol.success: return np.ones_like(y_obs) * 1e6
                    # Unit conversion to match original (ng/mL)
                    cp_pred = (sol.y[0] / disp['V_L']) * 1000.0
                    return cp_pred - y_obs

                res = least_squares(objective, init_vals[opt_idx], bounds=(lbs[opt_idx], ubs[opt_idx]), ftol=1e-7)
                
                # Update Final
                final_vals = init_vals.copy()
                final_vals[opt_idx] = res.x
                res_dict = dict(zip(active_p, final_vals))
                
                # --- RESULTS TABS (Merging your original logic) ---
                st.success("Fit Successful")
                t1, t2, t3 = st.tabs(["PK Results", "Parameter Estimates", "Save Model"])
                
                with t1:
                    # Individual Plots
                    fig, ax = plt.subplots()
                    for col in rep_cols: ax.plot(t_obs, pk_df[col], alpha=0.3, label=col)
                    ax.scatter(t_obs, y_obs, color='black', label='Observed Mean')
                    
                    t_fine = np.linspace(0, t_obs[-1], 400)
                    sol_f = solve_ivp(pk_ode_rhs, (0, t_fine[-1]), [0.0]*comps, t_eval=t_fine, 
                                      args=(model_choice, res_dict, disp), method='LSODA')
                    ax.plot(t_fine, (sol_f.y[0]/disp['V_L'])*1000.0, color='red', linewidth=2, label='Convolution Fit')
                    ax.set_ylabel("Concentration (ng/mL)")
                    ax.legend()
                    st.pyplot(fig)
                    
                    # Summary Data Table
                    st.write("Mean PK Summary Table")
                    st.dataframe(build_pk_study_tables(pk_df, "h", "ng/mL"))

                with t2:
                    # AIC / Fit Stats
                    n = len(y_obs)
                    rss = np.sum(res.fun**2)
                    k = len(opt_idx)
                    aic = n * np.log(rss/n) + 2*k
                    
                    col_a, col_b = st.columns(2)
                    col_a.metric("AIC", f"{aic:.2f}")
                    col_b.metric("RSS", f"{rss:.4f}")
                    
                    st.write("Fitted Parameters")
                    st.table(pd.DataFrame([res_dict], index=["Estimate"]))

                with t3:
                    if st.button("Save as InVivoFit"):
                        st.session_state["InVivoFit"] = {
                            "model": model_choice,
                            "parameter_estimates": res_dict,
                            "disposition": disp
                        }
                        st.success("Saved to session.")

            except Exception as e:
                st.error(f"Error during fit: {e}")

if __name__ == "__main__":
    render()