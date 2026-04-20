import modules.common as common
from modules.common import *
from scipy.optimize import least_squares
from scipy.integrate import solve_ivp
from io import StringIO
import numpy as np
import pandas as pd

st = common.st
pd = common.pd
np = common.np
plt = common.plt

# --- SAMPLE DATA (Hardcoded for quick test) ---
DEFAULT_PK_DATA = """0\t0\t0\t0
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

# --- Constants ---
DOSE_UNIT_TO_MG = {"ng": 1e-6, "ug": 1e-3, "mg": 1.0, "g": 1e3}
VOLUME_UNIT_TO_L = {"mL": 1e-3, "L": 1.0}

# --- Core Math ---
def _weibull_rate_unit(t, MDT, b):
    t_safe = np.maximum(t, 1e-12)
    MDT_s = np.maximum(MDT, 1e-12)
    b_s = np.maximum(b, 1e-12)
    # The derivative of (1 - exp(-(t/MDT)^b))
    return (b_s / MDT_s) * np.power(t_safe / MDT_s, b_s - 1) * np.exp(-np.power(t_safe / MDT_s, b_s))

def get_kab_rate(t, model_name, p):
    # Fmax is the total % absorbed (e.g. 72.9)
    Fmax_f = np.clip(p.get('Fmax', 100.0), 0, 100) / 100.0
    
    if model_name == "Single Weibull":
        return Fmax_f * _weibull_rate_unit(t, p['MDT1'], p['b1'])
    elif model_name == "Double Weibull":
        f1 = np.clip(p.get('f1', 0.5), 0, 1)
        return Fmax_f * (f1 * _weibull_rate_unit(t, p['MDT1'], p['b1']) + (1-f1) * _weibull_rate_unit(t, p['MDT2'], p['b2']))
    else: # Triple
        f1, f2 = np.clip(p.get('f1', 0.3), 0, 1), np.clip(p.get('f2', 0.3), 0, 1)
        if (f1 + f2) > 1.0:
            s = f1 + f2
            f1 /= s; f2 /= s
        f3 = 1.0 - f1 - f2
        return Fmax_f * (f1 * _weibull_rate_unit(t, p['MDT1'], p['b1']) + 
                         f2 * _weibull_rate_unit(t, p['MDT2'], p['b2']) +
                         f3 * _weibull_rate_unit(t, p['MDT3'], p['b3']))

def pk_ode_rhs(t, y, model_name, p_dict, disp):
    kab = get_kab_rate(t, model_name, p_dict)
    # Input is Dose * Fraction_Rate_Per_Hour
    input_mass = disp['dose_mg'] * kab 
    A1 = y[0]
    da1 = input_mass - disp['k10'] * A1
    
    if disp['compartments'] == 1:
        return [da1]
    
    A2 = y[1]
    da1 += disp['k21'] * A2 - disp['k12'] * A1
    da2 = disp['k12'] * A1 - disp['k21'] * A2
    
    if disp['compartments'] == 2:
        return [da1, da2]
    
    A3 = y[2]
    da1 += disp['k31'] * A3 - disp['k13'] * A1
    da3 = disp['k13'] * A1 - disp['k31'] * A3
    return [da1, da2, da3]

def render():
    render_display_settings()
    st.header("🧬 Deconvolution through Convolution")

    # 1. DISPOSITION (Compact)
    with st.expander("1. Disposition Settings", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        comps = c1.selectbox("Compartments", [1, 2, 3], index=1)
        # Your Image Values
        dose_val = c2.number_input("Dose", value=6666666.6)
        dose_u = c3.selectbox("Dose Units", ["ng", "mg"], index=0)
        v_val = c4.number_input("V", value=1136.9)
        
        c5, c6, c7, c8 = st.columns(4)
        v_u = c5.selectbox("V Units", ["mL", "L"], index=0)
        k10 = c6.number_input("k10 (1/h)", value=0.77, format="%.4f")
        k12 = c7.number_input("k12 (1/h)", value=1.38, format="%.4f")
        k21 = c8.number_input("k21 (1/h)", value=1.81, format="%.4f")

    # 2. PK DATA (Pre-filled with your table)
    pk_text = st.text_area("2. PK Table (Time | Rep1 | Rep2...)", value=DEFAULT_PK_DATA, height=150)
    
    # 3. MODEL & PARAMS
    model_choice = "Triple Weibull"
    st.subheader("3. Weibull Parameters (Wide Table)")
    
    # Your Exact Values from image_ec10bf.png
    # MDT1=7.48, b1=0.83, f1=0.04, MDT2=29.31, b2=6.75, f2=0.02, MDT3=265.68, b3=1.57, Fmax=72.90
    param_names = ["MDT1", "b1", "f1", "MDT2", "b2", "f2", "MDT3", "b3", "Fmax"]
    val_list = [7.48, 0.83, 0.04, 29.31, 6.75, 0.02, 265.68, 1.57, 72.90]
    
    df_params = pd.DataFrame({
        "Parameter": param_names,
        "Initial Value": val_list,
        "Min": [0.1, 0.1, 0.0, 0.1, 0.1, 0.0, 0.1, 0.1, 0.0],
        "Max": [1000.0, 20.0, 1.0, 1000.0, 20.0, 1.0, 2000.0, 20.0, 100.0],
        "Fix": [False] * 9
    })
    
    edited_df = st.data_editor(df_params.T, hide_index=False, use_container_width=True)
    # Transpose back for logic
    final_edit = edited_df.T 

    if st.button("Run Deconvolution Fit"):
        try:
            # Parse Data
            df = pd.read_csv(StringIO(pk_text), sep=r"\s+", header=None)
            t_obs = df[0].values.astype(float)
            cp_obs = df.iloc[:, 1:].mean(axis=1).values.astype(float)
            
            # Setup Disposition
            disp = {
                "compartments": comps, "k10": k10, "k12": k12, "k21": k21, "k13": 0.0, "k31": 0.0,
                "dose_mg": dose_val * DOSE_UNIT_TO_MG[dose_u], 
                "V_L": v_val * VOLUME_UNIT_TO_L[v_u]
            }

            p_names = final_edit['Parameter'].tolist()
            init_vals = final_edit['Initial Value'].values.astype(float)
            lbs = final_edit['Min'].values.astype(float)
            ubs = final_edit['Max'].values.astype(float)
            fix_mask = final_edit['Fix'].values.astype(bool)
            
            opt_idx = [i for i, fixed in enumerate(fix_mask) if not fixed]
            
            def objective(x_opt):
                p_curr = init_vals.copy()
                p_curr[opt_idx] = x_opt
                p_dict = dict(zip(p_names, p_curr))
                
                # Use a smaller rtol for better accuracy in complex triple Weibulls
                sol = solve_ivp(pk_ode_rhs, (0, t_obs[-1]), [0.0]*comps, t_eval=t_obs, 
                                args=(model_choice, p_dict, disp), method='LSODA', rtol=1e-7, atol=1e-9)
                
                if not sol.success: return np.ones_like(cp_obs) * 1e6
                
                # Convert Amount (mg) to Concentration (ng/mL if Dose was ng and V was mL)
                # Amount is in mg. V_L is in L. mg/L = ug/mL. 
                # To match ng/mL, we multiply by 1000.
                cp_pred = (sol.y[0] / disp['V_L']) * 1000.0
                return cp_pred - cp_obs

            # Run Fitting
            res = least_squares(objective, init_vals[opt_idx], bounds=(lbs[opt_idx], ubs[opt_idx]), 
                                xtol=1e-8, ftol=1e-8, method='trf')
            
            # Display
            final_p = init_vals.copy()
            final_p[opt_idx] = res.x
            res_dict = dict(zip(p_names, final_p))
            st.success("Fitting Finished")
            st.json(res_dict)
            
            # Final Plot
            t_fine = np.linspace(0, t_obs[-1], 400)
            sol_final = solve_ivp(pk_ode_rhs, (0, t_fine[-1]), [0.0]*comps, t_eval=t_fine, 
                                  args=(model_choice, res_dict, disp), method='LSODA')
            
            fig, ax = plt.subplots()
            ax.scatter(t_obs, cp_obs, color='black', label='Observed (Mean)')
            ax.plot(t_fine, (sol_final.y[0]/disp['V_L'])*1000.0, color='red', label='Convolution Fit')
            ax.set_xlabel("Time (h)")
            ax.set_ylabel("Cp (ng/mL)")
            ax.legend()
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Error: {e}")

if __name__ == "__main__":
    render()