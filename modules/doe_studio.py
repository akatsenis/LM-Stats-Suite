import modules.common as common
from modules.common import *


def _safe_factor_prefix(i):
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return alphabet[i] if i < len(alphabet) else f"F{i+1}"


def _build_factorial_design(factor_names, low_levels, high_levels, blocks=1, center_points=0, replicates=1, randomize=True, seed=123):
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
            for cp in range(center_points):
                row = {"Block": block, "Replicate": rep, "RunType": "Center"}
                for i, name in enumerate(factor_names):
                    row[f"{_safe_factor_prefix(i)} (coded)"] = 0
                    row[name] = (low_levels[i] + high_levels[i]) / 2
                runs.append(row)
    design = pd.DataFrame(runs)
    if randomize and len(design) > 0:
        rng = np.random.default_rng(seed)
        parts = []
        for blk, sub in design.groupby("Block", sort=True):
            sub = sub.sample(frac=1, random_state=int(rng.integers(0, 1_000_000))).reset_index(drop=True)
            parts.append(sub)
        design = pd.concat(parts, ignore_index=True)
    design.insert(0, "Run", np.arange(1, len(design) + 1))
    return design


def render():
    render_display_settings()
    st.sidebar.title("🧪 DoE Studio")
    st.sidebar.markdown("Design of Experiments")
    tabs = st.tabs(["Design Builder", "Response Analysis"])
    with tabs[0]:
        app_header("🧪 DoE Studio - Design Builder", "Create a basic 2-level full-factorial design with blocks, center points, replicates, and randomization.")
        c1, c2 = st.columns([1,1])
        with c1:
            n_factors = st.number_input("Number of factors", min_value=2, max_value=8, value=3, step=1)
            blocks = st.number_input("Blocks", min_value=1, max_value=10, value=1, step=1)
            replicates = st.number_input("Replicates", min_value=1, max_value=10, value=1, step=1)
        with c2:
            center_points = st.number_input("Center points per block", min_value=0, max_value=20, value=0, step=1)
            randomize = st.checkbox("Randomize within block", value=True)
            seed = st.number_input("Random seed", min_value=1, max_value=999999, value=123, step=1)
        st.markdown("### Factor definitions")
        names=[]; lows=[]; highs=[]
        for i in range(int(n_factors)):
            cols = st.columns([1.3,1,1])
            names.append(cols[0].text_input(f"Factor {i+1} name", value=f"Factor {i+1}", key=f"doe_name_{i}"))
            lows.append(cols[1].number_input(f"Low level {i+1}", value=0.0, key=f"doe_low_{i}"))
            highs.append(cols[2].number_input(f"High level {i+1}", value=1.0, key=f"doe_high_{i}"))
        if st.button("Generate design", type="primary", key="gen_design"):
            design = _build_factorial_design(names, lows, highs, int(blocks), int(center_points), int(replicates), randomize, int(seed))
            st.session_state["doe_generated_design"] = design
        if "doe_generated_design" in st.session_state:
            design = st.session_state["doe_generated_design"]
            st.success(f"Generated design with {len(design)} runs")
            st.dataframe(design, use_container_width=True)
            excel_bytes = make_excel_bytes({"Design": design})
            st.download_button("Download design workbook", excel_bytes, file_name="doe_design.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            st.download_button("Download design CSV", design.to_csv(index=False).encode("utf-8"), file_name="doe_design.csv", mime="text/csv")
    with tabs[1]:
            app_header("🧪 App 09 - DoE / Response Surfaces", "Build a full-factorial design, assign blocks, and fit linear / interaction / quadratic response-surface models.")
            tab_build, tab_analyze = st.tabs(["Build design", "Analyze responses"])
        
            with tab_build:
                st.markdown("### Build a full-factorial DoE")
                st.markdown("Enter factor names and low / high levels. You can add blocks, center points, replicates, and randomize the run order within block.")
                c1, c2, c3, c4 = st.columns([0.8, 0.9, 0.9, 1])
                with c1:
                    n_factors = st.number_input("Number of factors", min_value=2, max_value=8, value=2, step=1)
                with c2:
                    n_blocks = st.number_input("Number of blocks", min_value=1, max_value=20, value=1, step=1)
                with c3:
                    replicates = st.number_input("Replicates / treatment / block", min_value=1, max_value=20, value=1, step=1)
                with c4:
                    center_points = st.number_input("Center points / block", min_value=0, max_value=20, value=0, step=1)
        
                factor_names_txt = st.text_input(
                    "Factor names (comma-separated)",
                    value=", ".join([chr(65 + i) for i in range(n_factors)])
                )
                factor_names = [f.strip() for f in factor_names_txt.split(",") if f.strip()]
                if len(factor_names) != n_factors:
                    st.warning(f"Please provide exactly {n_factors} factor names.")
                else:
                    st.markdown("**Low / high levels**")
                    cols = st.columns(n_factors)
                    low_vals, high_vals = {}, {}
                    for i, f in enumerate(factor_names):
                        with cols[i]:
                            low_vals[f] = st.text_input(f"{f} low", value="-1", key=f"low_{f}")
                            high_vals[f] = st.text_input(f"{f} high", value="1", key=f"high_{f}")
        
                    randomize = st.checkbox("Randomize runs within each block", value=True)
                    random_seed = st.number_input("Random seed", min_value=0, max_value=100000, value=42, step=1) if randomize else 42
        
                    try:
                        low_numeric = {f: float(low_vals[f]) for f in factor_names}
                        high_numeric = {f: float(high_vals[f]) for f in factor_names}
                        for f in factor_names:
                            if high_numeric[f] == low_numeric[f]:
                                raise ValueError(f"{f}: high level must differ from low level.")
        
                        coded_combos = list(product([-1, 1], repeat=n_factors))
                        rows = []
                        rng = np.random.default_rng(int(random_seed))
                        for block in range(1, int(n_blocks) + 1):
                            block_rows = []
                            for rep in range(1, int(replicates) + 1):
                                for combo in coded_combos:
                                    row = {"Block": block, "Replicate": rep, "Center point": "No"}
                                    for i, f in enumerate(factor_names):
                                        coded = combo[i]
                                        actual = low_numeric[f] if coded == -1 else high_numeric[f]
                                        row[f"{f} (coded)"] = coded
                                        row[f] = actual
                                    block_rows.append(row)
                            for cp in range(1, int(center_points) + 1):
                                row = {"Block": block, "Replicate": np.nan, "Center point": f"CP{cp}"}
                                for f in factor_names:
                                    row[f"{f} (coded)"] = 0
                                    row[f] = (low_numeric[f] + high_numeric[f]) / 2
                                block_rows.append(row)
                            if randomize:
                                rng.shuffle(block_rows)
                            for run_idx, row in enumerate(block_rows, start=1):
                                row["Run"] = run_idx
                                rows.append(row)
        
                        design_df = pd.DataFrame(rows)
                        ordered_cols = ["Block", "Run", "Replicate", "Center point"]
                        for f in factor_names:
                            ordered_cols += [f"{f} (coded)", f]
                        design_df = design_df[ordered_cols]
                        report_table(design_df, "Generated DoE design", 3)
        
                        c_csv, c_xlsx = st.columns(2)
                        with c_csv:
                            st.download_button(
                                "⬇️ Export design as CSV",
                                data=design_df.to_csv(index=False).encode("utf-8"),
                                file_name="doe_design.csv",
                                mime="text/csv",
                            )
                        with c_xlsx:
                            st.download_button(
                                "⬇️ Export design as Excel",
                                data=make_excel_bytes({"DoE Design": design_df}),
                                file_name="doe_design.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            )
                    except Exception as e:
                        st.error(str(e))
        
            with tab_analyze:
                st.markdown("### Analyze responses and build response surfaces")
                data_input = st.text_area("Paste completed DoE data with headers", height=240)
                decimals = st.slider("Decimals", 1, 8, DEFAULT_DECIMALS, key="doe_dec")
                if data_input:
                    try:
                        df = parse_pasted_table(data_input, header=True)
                        num_cols = get_numeric_columns(df)
                        all_cols = list(df.columns)
        
                        c1, c2, c3, c4 = st.columns([1.35, 1, 1, 1])
                        with c1:
                            factors = st.multiselect("Numeric factors", num_cols, default=num_cols[: min(2, len(num_cols))])
                        with c2:
                            response = st.selectbox("Response", [c for c in num_cols if c not in factors] or num_cols)
                        with c3:
                            model_type = st.selectbox("Model type", ["linear", "interaction", "quadratic"])
                        with c4:
                            block_col = st.selectbox("Block column (optional)", ["(None)"] + [c for c in all_cols if c not in factors + [response]])
        
                        if len(factors) >= 2:
                            use_cols = factors + [response] + ([block_col] if block_col != "(None)" else [])
                            d = df[use_cols].copy()
                            for c in factors + [response]:
                                d[c] = to_numeric(d[c])
                            d = d.dropna()
        
                            safe_factor_names = [f"F{i+1}" for i in range(len(factors))]
                            rename_map = {orig: safe for orig, safe in zip(factors, safe_factor_names)}
                            inv_map = {v: k for k, v in rename_map.items()}
        
                            safe_df = d.rename(columns=rename_map).rename(columns={response: "Response"})
                            if block_col != "(None)":
                                safe_df["Block"] = d[block_col].astype(str).values
        
                            formula = doe_formula(safe_factor_names, model_type=model_type)
                            if block_col != "(None)":
                                formula += " + C(Block)"
        
                            model = smf.ols(formula, data=safe_df).fit()
                            anova = anova_lm(model, typ=2).reset_index().rename(
                                columns={
                                    "index": "Source",
                                    "sum_sq": "Sum of Squares",
                                    "df": "df",
                                    "F": "F-Statistic",
                                    "PR(>F)": "P-Value",
                                }
                            )
                            anova["Mean Square"] = anova["Sum of Squares"] / anova["df"]
                            anova["SS (%)"] = anova["Sum of Squares"] / anova["Sum of Squares"].sum() * 100
        
                            def pretty_term(term):
                                term = str(term)
                                if term == "Residual":
                                    return "Error"
                                if term == "Intercept":
                                    return "Intercept"
                                if term.startswith("C(Block)"):
                                    return "Block"
                                term = term.replace(":", " × ")
                                term = term.replace("I(", "").replace(" ** 2)", "²")
                                for safe, orig in inv_map.items():
                                    term = term.replace(safe, orig)
                                return term
        
                            anova["Source"] = anova["Source"].map(pretty_term)
                            coef = pd.DataFrame({
                                "Term": [pretty_term(t) for t in model.params.index],
                                "Coefficient": model.params.values,
                                "P-Value": model.pvalues.values,
                            })
        
                            report_table(anova[["Source", "df", "Sum of Squares", "Mean Square", "F-Statistic", "P-Value", "SS (%)"]], f"DoE ANOVA ({model_type} model)", decimals)
                            report_table(coef, "Model coefficients", decimals)
        
                            xfac = st.selectbox("X-axis factor", factors, index=0)
                            yfac = st.selectbox("Y-axis factor", [f for f in factors if f != xfac], index=0)
                            other_factors = [f for f in factors if f not in [xfac, yfac]]
                            fixed_vals = {}
                            if other_factors:
                                st.markdown("**Fixed levels for remaining factors**")
                                cols = st.columns(len(other_factors))
                                for i, f in enumerate(other_factors):
                                    fixed_vals[f] = cols[i].number_input(f, value=float(d[f].mean()))
        
                            x_vals = np.linspace(d[xfac].min(), d[xfac].max(), 60)
                            y_vals = np.linspace(d[yfac].min(), d[yfac].max(), 60)
                            xx, yy = np.meshgrid(x_vals, y_vals)
                            grid = pd.DataFrame({xfac: xx.ravel(), yfac: yy.ravel()})
                            for f in other_factors:
                                grid[f] = fixed_vals[f]
                            if block_col != "(None)":
                                block_default = str(d[block_col].iloc[0])
                                selected_block = st.selectbox("Block level for prediction grid", sorted(d[block_col].astype(str).unique()))
                                grid["Block"] = selected_block
                            safe_grid = grid.rename(columns=rename_map)
                            zz = model.predict(safe_grid).to_numpy().reshape(xx.shape)
        
                            contour_cfg = common.safe_get_plot_cfg("DoE contour")
                            fig_contour, ax = plt.subplots(figsize=(contour_cfg["fig_w"], contour_cfg["fig_h"]))
                            cs = ax.contourf(xx, yy, zz, levels=20, cmap="viridis")
                            ax.contour(xx, yy, zz, levels=10, colors=contour_cfg["primary_color"], linewidths=max(0.6, contour_cfg["aux_line_width"] * 0.7))
                            fig_contour.colorbar(cs, ax=ax, label=response)
                            ax.scatter(
                                d[xfac],
                                d[yfac],
                                c="white",
                                edgecolor=contour_cfg["primary_color"],
                                s=contour_cfg["marker_size"],
                            )
                            apply_ax_style(ax, f"Contour plot for {response}", xfac, yfac, plot_key="DoE contour")
                            st.pyplot(fig_contour)
        
                            fig_surface = plt.figure(figsize=(contour_cfg["fig_w"], contour_cfg["fig_h"] + 0.5))
                            ax3 = fig_surface.add_subplot(111, projection="3d")
                            surf = ax3.plot_surface(xx, yy, zz, cmap="viridis", edgecolor="none", alpha=0.88)
                            ax3.scatter(d[xfac], d[yfac], d[response], c="black", s=max(12, contour_cfg["marker_size"] * 0.45))
                            ax3.set_xlabel(xfac)
                            ax3.set_ylabel(yfac)
                            ax3.set_zlabel(response)
                            ax3.set_title(f"Response surface for {response}")
                            fig_surface.colorbar(surf, ax=ax3, shrink=0.68, aspect=12)
                            st.pyplot(fig_surface)
        
                            fig_res = residual_plot(model.fittedvalues, model.resid, xlabel="Fitted values", ylabel="Residuals", title="Residuals vs fitted")
                            st.pyplot(fig_res)
                            fig_qq = qq_plot(model.resid, title="Normal probability plot of DoE residuals")
                            st.pyplot(fig_qq)
        
                            export_results(
                                prefix="doe_response_surfaces",
                                report_title="Statistical Analysis Report",
                                module_name="DoE / Response Surfaces",
                                statistical_analysis="A design-of-experiments style regression model was fitted to the selected numeric response using the chosen numeric factors. Depending on the selected option, the model included linear terms only, linear plus interactions, or a quadratic response-surface structure. When a block column was supplied, block was included as a categorical effect. ANOVA, model coefficients, contour plots, surface plots, and residual diagnostics were generated from the fitted model.",
                                offer_text="This analysis offers a practical way to build simple factorial designs, quantify factor effects, inspect interactions or curvature, account for blocks, and visualize the response surface over two selected factors while fixing any remaining factors at chosen values.",
                                python_tools="Python tools used here include pandas and numpy for design construction and factor selection, itertools.product for generating full-factorial combinations, statsmodels for model fitting and ANOVA, matplotlib for contour, 3D surface, and residual diagnostic plots, openpyxl for Excel export, and reportlab for the PDF-style report.",
                                table_map={"DoE ANOVA": anova, "Coefficients": coef},
                                figure_map={
                                    "Contour plot": fig_to_png_bytes(fig_contour),
                                    "Response surface": fig_to_png_bytes(fig_surface),
                                    "Residuals vs fitted": fig_to_png_bytes(fig_res),
                                    "Normal probability plot": fig_to_png_bytes(fig_qq),
                                },
                                conclusion="The fitted DoE model can be used to assess influential factors, detect interactions or curvature, evaluate block effects, and visualize predicted response behavior across the chosen design space.",
                                decimals=decimals,
                            )
                    except Exception as e:
                        st.error(str(e))
