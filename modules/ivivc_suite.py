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

# --- Constants & Helpers ---
TIME_UNIT_TO_HOURS = {"Minutes": 1 / 60.0, "Hours": 1.0, "Days": 24.0}
DOSE_UNIT_TO_MG = {"ng": 1e-6, "ug": 1e-3, "mg": 1.0, "g": 1e3}
VOLUME_UNIT_TO_L = {"uL": 1e-6, "mL": 1e-3, "L": 1.0}
CP_MG_PER_L_TO_UNIT = {"mg/L": 1.0, "ug/mL": 1.0, "ng/mL": 1e3, "ug/L": 1e3, "mg/mL": 1e-3}

# --- Core Weibull Rate Functions (Analytical Derivatives) ---
def _weibull_rate_unit(t, MDT, b):
    """Instantaneous rate (PDF) of a single Weibull unit."""
    t_safe = np.maximum(t, 1e-12)
    MDT_s = np.maximum(MDT, 1e-12)
    b_s = np.maximum(b, 1e-12)
    return (b_s / MDT_s) * np.power(t_safe / MDT_s, b_s - 1) * np.exp(-np.power(t_safe / MDT_s, b_s))

def get_kab_rate(t, model_name, p):
    """Calculates total kab for Single, Double, or Triple Weibull."""
    # p is a dict or array depending on context. Here we use a dict for clarity in ODE.
    Fmax_f = p['Fmax'] / 100.0
    if model_name == "Single Weibull":
        return Fmax_f * _weibull_rate_unit(t, p['MDT1'], p['b1'])
    elif model_name == "Double Weibull":
        f1 = np.clip(p['f1'], 0, 1)
        return Fmax_f * (f1 * _weibull_rate_unit(t, p['MDT1'], p['b1']) + 
                         (1 - f1) * _weibull_rate_unit(t, p['MDT2'], p['b2']))
    else: # Triple
        f1, f2 = np.clip(p['f1'], 0, 1), np.clip(p['f2'], 0, 1)
        if f1 + f2 > 1.0:
            tot = f1 + f2
            f1, f2 = f1/tot, f2/tot
        f3 = 1.0 - f1 - f2
        return Fmax_f * (f1 * _weibull_rate_unit(t, p['MDT1'], p['b1']) + 
                         f2 * _weibull_rate_unit(t, p['MDT2'], p['b2']) +
                         f3 * _weibull_rate_unit(t, p['MDT3'], p['b3']))

# --- ODE System ---
def pk_ode_rhs(t, y, model_name, p_dict, disp):
    kab = get_kab_rate(t, model_name, p_dict)
    input_mass = disp['dose_mg'] * disp['bio'] * kab
    A1 = y[0]
    da1 = input_mass - disp['k10'] * A1
    
    if disp['compartments'] == 1:
        return [da1]
    if disp['compartments'] == 2:
        A2 = y[1]
        da1 += disp['k21'] * A2 - disp['k12'] * A1
        da2 = disp['k12'] * A1 - disp['k21'] * A2
        return [da1, da2]
    if disp['compartments'] == 3:
        A2, A3 = y[1], y[2]
        da1 += disp['k21'] * A2 + disp['k31'] * A3 - (disp['k12'] + disp['k13']) * A1
        da2 = disp['k12'] * A1 - disp['k21'] * A2
        da3 = disp['k13'] * A1 - disp['k31'] * A3
        return [da1, da2, da3]

# --- UI & Streamlit Render ---
def render():
    render_display_settings()
    st.sidebar.title("💊 PK Suite")
    tool = st.sidebar.radio("Tool", ["🧬 Deconvolution through convolution", "🔗 IVIVC"])

    if tool == "🧬 Deconvolution through convolution":
        app_header("🧬 Deconvolution through convolution", "Optimized ODE-based Weibull fitting.")
        
        # Compact Disposition Inputs
        with st.expander("1. Disposition & Dose (Compact)", expanded=True):
            r1c1, r1c2, r1c3, r1c4 = st.columns(4)
            comps = r1c1.selectbox("Compartments", [1, 2, 3], index=1)
            dose_val = r1c2.number_input("Dose", value=6666666.6)
            dose_u = r1c3.selectbox("Dose units", list(DOSE_UNIT_TO_MG.keys()))
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

        # PK Data Input
        pk_text = st.text_area("2. Paste PK Table (Time | Cp1 | Cp2 ...)", height=150)
        
        # Model Selection
        model_choice = st.selectbox("3. Select Weibull Model", ["Single Weibull", "Double Weibull", "Triple Weibull"], index=2)
        
        # Parameter Editor Table (Wide)
        st.write("4. Initial Parameters & Constraints")
        param_names = {
            "Single Weibull": ["Fmax", "MDT1", "b1"],
            "Double Weibull": ["Fmax", "f1", "MDT1", "b1", "MDT2", "b2"],
            "Triple Weibull": ["Fmax", "f1", "f2", "MDT1", "b1", "MDT2", "b2", "MDT3", "b3"]
        }[model_choice]

        # Build initial DataFrame for wide format
        default_vals = {"Fmax": 100.0, "f1": 0.5, "f2": 0.3, "MDT1": 5.0, "b1": 1.2, "MDT2": 20.0, "b2": 1.5, "MDT3": 50.0, "b3": 2.0}
        df_init = pd.DataFrame(index=["Initial Value", "Min", "Max", "Fix"], columns=param_names)
        for p in param_names:
            df_init.at["Initial Value", p] = default_vals.get(p, 10.0)
            df_init.at["Min", p] = 0.0 if "f" in p or "Fmax" in p else 0.01
            df_init.at["Max", p] = 1.0 if "f" in p else (120.0 if "Fmax" in p else 1000.0)
            df_init.at["Fix", p] = False

        edited_df = st.data_editor(df_init.T.reset_index().rename(columns={'index': 'Parameter'}), hide_index=True, use_container_width=True)
        
        if st.button("Run Deconvolution Fit") and pk_text:
            try:
                # 1. Parse Data
                pk_df = parse_pk_profile_table(pk_text)
                t_obs = pk_df.iloc[:, 0].values.astype(float)
                cp_obs = pk_df.iloc[:, 1:].mean(axis=1).values.astype(float)
                
                disp = {
                    "compartments": comps, "k10": k10, "k12": k12, "k21": k21, "k13": k13, "k31": k31,
                    "dose_mg": dose_val * DOSE_UNIT_TO_MG[dose_u], "V_L": v_val * VOLUME_UNIT_TO_L[v_u],
                    "bio": bio, "cp_unit_factor": 1.0 # Assume factor handled by user for now
                }
                
                # 2. Setup Solver with "Fixed" Logic
                p_names = edited_df['Parameter'].tolist()
                init_vals = edited_df['Initial Value'].values.astype(float)
                lbs = edited_df['Min'].values.astype(float)
                ubs = edited_df['Max'].values.astype(float)
                fix_mask = edited_df['Fix'].values.astype(bool)
                
                opt_idx = [i for i, fixed in enumerate(fix_mask) if not fixed]
                x0 = init_vals[opt_idx]
                bounds = (lbs[opt_idx], ubs[opt_idx])

                def objective(x_try):
                    curr_p = init_vals.copy()
                    curr_p[opt_idx] = x_try
                    p_dict = dict(zip(p_names, curr_p))
                    
                    sol = solve_ivp(pk_ode_rhs, (0, t_obs[-1]), [0]*comps, t_eval=t_obs, 
                                    args=(model_choice, p_dict, disp), method='RK45')
                    if not sol.success: return np.ones_like(cp_obs) * 1e6
                    
                    cp_pred = sol.y[0] / (disp['V_L'] / VOLUME_UNIT_TO_L[v_u])
                    return cp_pred - cp_obs

                # 3. Fit
                res = least_squares(objective, x0, bounds=bounds, method='trf')
                
                # 4. Result Display
                final_params = init_vals.copy()
                final_params[opt_idx] = res.x
                st.success("Fit Converged!")
                st.write("Final Estimates:", dict(zip(p_names, final_params)))
                
                # Simple Plot
                final_p_dict = dict(zip(p_names, final_params))
                sol_final = solve_ivp(pk_ode_rhs, (0, t_obs[-1]), [0]*comps, t_eval=np.linspace(0, t_obs[-1], 200), 
                                      args=(model_choice, final_p_dict, disp))
                
                fig, ax = plt.subplots()
                ax.scatter(t_obs, cp_obs, label="Observed")
                ax.plot(sol_final.t, sol_final.y[0] / (disp['V_L'] / VOLUME_UNIT_TO_L[v_u]), color='red', label="Fit")
                ax.set_xlabel("Time")
                ax.set_ylabel("Concentration")
                ax.legend()
                st.pyplot(fig)

            except Exception as e:
                st.error(f"Error: {e}")

if __name__ == "__main__":
    render()