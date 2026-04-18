import modules.common as common
from modules.common import *

st = common.st
pd = common.pd
np = common.np
plt = common.plt


def load_dual_sample_text(state_key_a, sample_key_a, state_key_b, sample_key_b):
    from modules.stats_suite import SAMPLE_DATA
    st.session_state[state_key_a] = SAMPLE_DATA[sample_key_a]
    st.session_state[state_key_b] = SAMPLE_DATA[sample_key_b]


def render():
    render_display_settings()
    st.sidebar.title("💊 lm Stats")
    st.sidebar.markdown("IVIVC Suite")
    tool = st.sidebar.radio("IVIVC tool", ["💊 Dissolution Comparison (f₂)"], key="ivivc_tool")
    st.sidebar.caption("This page currently contains dissolution similarity tools. Additional IVIVC tools can be added here later.")

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
