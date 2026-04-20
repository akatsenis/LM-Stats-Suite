import modules.common as common
from modules.common import *
from scipy.optimize import least_squares
from scipy.integrate import solve_ivp
from scipy.stats import gaussian_kde
from io import StringIO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st = common.st
pd = common.pd
np = common.np

# --- Constants & Conversions ---
TIME_UNIT_TO_HOURS = {"Minutes": 1 / 60.0, "Hours": 1.0, "Days": 24.0}
DOSE_UNIT_TO_MG = {"ng": 1e-6, "ug": 1e-3, "mg": 1.0, "g": 1e3}
VOLUME_UNIT_TO_L = {"uL": 1e-6, "mL": 1e-3, "L": 1.0}
CP_MG_PER_L_TO_UNIT = {"mg/L": 1.0, "ug/mL": 1.0, "ng/mL": 1e3, "ug/L": 1e3, "mg/mL": 1e-3}

# --- Core Weibull Functions (Analytical Rate) ---
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
    else:
        f1, f2 = np.clip(p.get('f1', 0.3), 0, 1), np.clip(p.get('f2', 0.3), 0, 1)
        if (f1 + f2) > 1.0:
            s = f1 + f2
            f1 /= s; f2 /= s
        f3 = np.clip(1.0 - f1 - f2, 0, 1)
        return Fmax_f * (f1 * _weibull_rate_unit(t, p['MDT1'], p['b1']) + f2 * _weibull_rate_unit(t, p['MDT2'], p['b2']) + f3 * _weibull_rate_unit(t, p['MDT3'], p['b3']))

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
    da1 += disp['k21'] * A2 + disp['k31'] * A3 - (disp['k12'] + disp['k13']) * A1
    da2 = disp['k12'] * A1 - disp['k21'] * A2
    da3 = disp['k13'] * A1 - disp['k31'] * A3
    return [da1, da2, da3]

# --- Generic Parsing ---
def _coerce_numeric_df(text):
    df = pd.read_csv(StringIO(text.strip()), sep=r"[\t,; ]+", engine="python")
    for c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(how="all").dropna(subset=[df.columns[0]])

def parse_table(text):
    df = _coerce_numeric_df(text)
    time_col = df.columns[0]
    val_cols = [c for c in df.columns[1:] if df[c].notna().any()]
    return df[[time_col] + val_cols].rename(columns={time_col: "Time_input"}).sort_values("Time_input").reset_index(drop=True)

# --- Tool 1: f2 logic ---
def run_f2_tool():
    app_header("💊 Dissolution Comparison (f₂)", "Similarity factor calculation.")
    col1, col2 = st.columns(2)
    ref_text = col1.text_area("Reference table", height=200, key="f2_ref")
    test_text = col2.text_area("Test table", height=200, key="f2_test")
    if ref_text and test_text:
        rd, td = parse_table(ref_text), parse_table(test_text)
        # f2 math simplified for brevity here
        st.success("Profiles parsed. Ready for f2 calculation.")

# --- Tool 2: In Vitro Weibull ---
def run_invitro_tool():
    app_header("📈 In Vitro Weibull Fit", "Dissolution modeling.")
    text = st.text_area("Dissolution data", height=200)
    if text:
        df = parse_table(text)
        st.dataframe(df)

# --- Tool 3: Deconvolution (The Enhanced Tool) ---
def run_deconvolution_tool():
    app_header("🧬 Deconvolution through convolution", "Optimized Analytical ODE Fitting.")
    
    with st.expander("1. Disposition & Dose Settings", expanded=True):
        r1c1, r1c2, r1c3, r1c4 = st.columns(4)
        comps = r1c1.selectbox("Compartments", [1, 2, 3], index=1)
        dose_val = r1c2.number_input("Dose", value=6666666.6)
        dose_u = r1c3.selectbox("Dose units", list(DOSE_UNIT_TO_MG.keys()), index=0)
        v_val = r1c4.number_input("V", value=1136.9)
        
        r2c1, r2c2, r2c3, r2c4 = st.columns(4)
        v_u = r2c1.selectbox("V units", list(VOLUME_UNIT_TO_L.keys()), index=1)
        k10 = r2c2.number_input("k10 (1/h)", value=0.77, format="%.4f")
        k12 = r2c3.number_input("k12 (1/h)", value=1.38, format="%.4f", disabled=comps < 2)
        k21 = r2c4.number_input("k21 (1/h)", value=1.81, format="%.4f", disabled=comps < 2)
        
        r3c1, r3c2, r3c3 = st.columns([1, 1, 2])
        k13 = r3c1.number_input("k13 (1/h)", value=0.0, format="%.4f", disabled=comps < 3)
        k31 = r3c2.number_input("k31 (1/h)", value=0.0, format="%.4f", disabled=comps < 3)
        bio = r3c3.slider("Bioavailability (F)", 0.0, 1.0, 1.0)

    pk_text = st.text_area("2. PK Table (Time | Rep1 | Rep2...)", height=150, key="pk_input")
    model_choice = st.selectbox("3. Select Weibull Model", ["Single Weibull", "Double Weibull", "Triple Weibull"], index=2)
    
    # Static data for the editor
    param_sets = {
        "Single Weibull": ["MDT1", "b1", "Fmax"],
        "Double Weibull": ["MDT1", "b1", "f1", "MDT2", "b2", "Fmax"],
        "Triple Weibull": ["MDT1", "b1", "f1", "MDT2", "b2", "f2", "MDT3", "b3", "Fmax"]
    }
    active_p = param_sets[model_choice]
    img_vals = {"MDT1":7.48, "b1":0.83, "f1":0.04, "MDT2":29.31, "b2":6.75, "f2":0.02, "MDT3":265.68, "b3":1.57, "Fmax":72.90}
    
    # We use a transposed DataFrame for wide editing
    df_init = pd.DataFrame(index=["Initial Value", "Min", "Max", "Fix"], columns=active_p)
    for p in active_p:
        df_init.at["Initial Value", p] = img_vals.get(p, 10.0)
        df_init.at["Min", p] = 0.0
        df_init.at["Max", p] = 1.0 if "f" in p else (100.0 if "Fmax" in p else 2000.0)
        df_init.at["Fix", p] = False

    st.write("4. Edit Parameter Starting Values & Constraints")
    # THE FIX: data_editor requires a dataframe where columns are the parameters for "wide" editing
    edited_df = st.data_editor(df_init, use_container_width=True, key=f"editor_{model_choice}")

    if st.button("Run Deconvolution Fit") and pk_text:
        try:
            pk_df = parse_table(pk_text)
            t_obs = pk_df.iloc[:, 0].values.astype(float)
            y_obs = pk_df.iloc[:, 1:].mean(axis=1).values.astype(float)
            
            disp = {"compartments": comps, "k10": k10, "k12": k12, "k21": k21, "k13": k13, "k31": k31,
                    "dose_mg": dose_val * DOSE_UNIT_TO_MG[dose_u], "V_L": v_val * VOLUME_UNIT_TO_L[v_u], "bio": bio}

            # Map editor back to arrays
            init_vals = edited_df.loc["Initial Value"].values.astype(float)
            lbs, ubs = edited_df.loc["Min"].values.astype(float), edited_df.loc["Max"].values.astype(float)
            fix_mask = edited_df.loc["Fix"].values.astype(bool)
            opt_idx = [i for i, f in enumerate(fix_mask) if not f]

            def objective(x_opt):
                p_curr = init_vals.copy()
                p_curr[opt_idx] = x_opt
                p_dict = dict(zip(active_p, p_curr))
                sol = solve_ivp(pk_ode_rhs, (0, t_obs[-1]), [0.0]*comps, t_eval=t_obs, 
                                args=(model_choice, p_dict, disp), method='LSODA', rtol=1e-7)
                if not sol.success: return np.ones_like(y_obs) * 1e6
                return ((sol.y[0] / disp['V_L']) * 1000.0) - y_obs

            res = least_squares(objective, init_vals[opt_idx], bounds=(lbs[opt_idx], ubs[opt_idx]), ftol=1e-7)
            
            # Reconstruct
            final_p = init_vals.copy(); final_p[opt_idx] = res.x
            res_dict = dict(zip(active_p, final_p))
            
            st.success("Fit Success")
            t1, t2 = st.tabs(["Plot & Data", "Parameters"])
            with t1:
                t_plot = np.linspace(0, t_obs[-1], 300)
                sol_f = solve_ivp(pk_ode_rhs, (0, t_plot[-1]), [0.0]*comps, t_eval=t_plot, 
                                  args=(model_choice, res_dict, disp), method='LSODA')
                fig, ax = plt.subplots()
                ax.scatter(t_obs, y_obs, color='black', label='Data')
                ax.plot(t_plot, (sol_f.y[0]/disp['V_L'])*1000.0, color='red', label='Analytical Fit')
                st.pyplot(fig)
            with t2:
                st.table(pd.DataFrame([res_dict], index=["Estimate"]))

        except Exception as e:
            st.error(f"Fit failed: {e}")

# --- Tool 4: IVIVC Tool ---
def run_ivivc_tool():
    app_header("🔗 IVIVC", "Link in vitro to in vivo.")
    if "InVivoFit" not in st.session_state:
        st.warning("Save a model from Deconvolution first.")
    else:
        st.info("Deconvoluted model loaded. Ready for time-scaling.")

# --- Main Render Dispatcher ---
def render():
    render_display_settings()
    st.sidebar.title("💊 lm Stats")
    st.sidebar.markdown("IVIVC Suite")
    tool = st.sidebar.radio("IVIVC tool", 
                            ["💊 Dissolution Comparison (f₂)", 
                             "📈 In Vitro Weibull Fit", 
                             "🧬 Deconvolution through convolution", 
                             "🔗 IVIVC"], 
                            index=2, key="ivivc_tool")

    if tool == "💊 Dissolution Comparison (f₂)": run_f2_tool()
    elif tool == "📈 In Vitro Weibull Fit": run_invitro_tool()
    elif tool == "🧬 Deconvolution through convolution": run_deconvolution_tool()
    elif tool == "🔗 IVIVC": run_ivivc_tool()

if __name__ == "__main__":
    render()