import re
from io import StringIO, BytesIO

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy import stats
from scipy.stats import t, norm, gaussian_kde, nct

import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.diagnostic import normal_ad
from sklearn.decomposition import PCA

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak

DEFAULT_DECIMALS = 3
FIG_W = 9
FIG_H = 5
SHOW_LEGEND = True
LEGEND_LOC = "best"
PRIMARY_COLOR = "#1f77b4"
SECONDARY_COLOR = "#ff7f0e"
TERTIARY_COLOR = "#2ca02c"
BAND_COLOR = "#93c5fd"
GRID_ALPHA = 0.0
MARKER_SIZE = 20
FIT_LINE_COLOR = PRIMARY_COLOR
FIT_LINE_STYLE = "-"
LINE_WIDTH = 1.0
AREA_ALPHA = 0.18
CI_LINE_STYLE = "--"
PI_LINE_STYLE = "--"
SPEC_LINE_STYLE = "--"
ARROW_SIZE = 0.03

PLOT_STYLE_KEYS = [
    "All graphs", "Descriptive summary", "Regression Analysis", "Shelf life",
    "Shelf life residual plot", "Shelf life Q-Q plot", "Dissolution comparison",
    "Dissolution bootstrap distribution", "Two-sample box plot", "Two-sample density plot",
    "Two-way ANOVA interaction", "Two-way ANOVA residual plot", "Two-way ANOVA Q-Q plot",
    "Tolerance/CI box plot", "PCA score plot", "PCA loading plot", "DoE contour",
    "DoE surface", "DoE residual plot", "DoE Q-Q plot", "DoE interaction", "DoE residual diagnostics",
    "DoE predicted vs observed", "DoE overlay contour", "Residual plot", "Q-Q plot",
]
LINE_STYLE_MAP = {"Solid": "-", "Dash": "--", "Dot": ":", "Dash-dot": "-."}
DEFAULT_STYLE_CFG = {
    "fig_w": 9, "fig_h": 5, "show_legend": True, "legend_loc": "best",
    "primary_color": PRIMARY_COLOR, "secondary_color": SECONDARY_COLOR, "tertiary_color": TERTIARY_COLOR,
    "band_color": BAND_COLOR, "marker_color": PRIMARY_COLOR, "line_color": PRIMARY_COLOR,
    "border_color": "#111827", "font_color": "#111827", "grid_alpha": 0.0,
    "line_style": "-", "aux_line_style": "--", "line_width": 1.0, "aux_line_width": 1.0,
    "marker_size": 20, "marker_style": "o", "tick_dir": "out", "tick_len": 4, "border_width": 1.0,
    "show_top": True, "show_right": True, "axis_type": "standard", "font_family": "DejaVu Sans",
    "font_size": 10, "title_size": 12, "label_size": 10, "tick_label_size": 9,
    "x_min": None, "x_max": None, "y_min": None, "y_max": None, "arrow_size": 0.025,
}


def init_page(page_title="lm Stats"):
    st.set_page_config(page_title=page_title, page_icon="🔬", layout="wide")
    inject_css()


def inject_css():
    st.markdown(
        """
        <style>
        .block-container {padding-top: 0.9rem; padding-bottom: 2rem;}
        .app-header {border:1px solid #e2e8f0; border-radius:14px; padding:16px 20px;
            background: linear-gradient(90deg, #f8fafc 0%, #ffffff 100%); margin-bottom: 1rem;
            box-shadow: 0 1px 4px rgba(15,23,42,0.05);}
        .app-title {font-size: 1.75rem; font-weight: 700; margin-bottom: 0.2rem; color:#0f172a;}
        .app-sub {font-size: 0.96rem; color:#475569;}
        .report-table table {width:100%; border-collapse:collapse; background:white; font-size:0.95rem;}
        .report-table caption {text-align:left; font-weight:700; font-size:1rem; color:#111827; margin-bottom:0.55rem;}
        .report-table thead th {border-top:2px solid #111827; border-bottom:1px solid #111827;
            padding:8px 12px; text-align:center; background:#f8fafc; color:#111827;}
        .report-table tbody td {padding:8px 12px; text-align:center; border:none;}
        .report-table tbody tr:last-child td {border-bottom:2px solid #111827;}
        .report-caption {font-size:0.85rem; color:#475569; margin-top:-0.5rem; margin-bottom:0.75rem;}
        div[data-testid='stMetric'] {border:1px solid #e2e8f0; border-radius:12px; padding:10px 12px; background:#fff;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _parse_style_float(val):
    if val in [None, "", "None"]:
        return None
    try:
        return float(str(val).strip())
    except Exception:
        return None


def get_plot_cfg(plot_key="All graphs"):
    cfg_map = st.session_state.get("plot_style_cfg", {})
    base = DEFAULT_STYLE_CFG.copy()
    base.update(cfg_map.get("All graphs", {}))
    if plot_key != "All graphs":
        base.update(cfg_map.get(plot_key, {}))
    for k in ["x_min", "x_max", "y_min", "y_max", "fig_w", "fig_h", "line_width", "aux_line_width", "border_width", "grid_alpha", "arrow_size"]:
        if k in base:
            parsed = _parse_style_float(base.get(k))
            if parsed is not None or k in ["x_min", "x_max", "y_min", "y_max"]:
                base[k] = parsed
    for k in ["marker_size", "tick_len", "font_size", "title_size", "label_size", "tick_label_size"]:
        try:
            base[k] = int(float(base.get(k, DEFAULT_STYLE_CFG[k])))
        except Exception:
            base[k] = DEFAULT_STYLE_CFG[k]
    return base


def safe_get_plot_cfg(plot_key="All graphs"):
    try:
        return get_plot_cfg(plot_key)
    except Exception:
        return DEFAULT_STYLE_CFG.copy()


def render_display_settings():
    global DEFAULT_DECIMALS, FIG_W, FIG_H, SHOW_LEGEND, LEGEND_LOC, PRIMARY_COLOR, SECONDARY_COLOR, TERTIARY_COLOR, BAND_COLOR, GRID_ALPHA, MARKER_SIZE, FIT_LINE_COLOR, FIT_LINE_STYLE, LINE_WIDTH, AREA_ALPHA, CI_LINE_STYLE, PI_LINE_STYLE, SPEC_LINE_STYLE, ARROW_SIZE
    if "plot_style_cfg" not in st.session_state:
        st.session_state["plot_style_cfg"] = {k: {} for k in PLOT_STYLE_KEYS}
        st.session_state["plot_style_cfg"]["All graphs"] = DEFAULT_STYLE_CFG.copy()

    with st.sidebar.expander("Display & export settings", expanded=False):
        DEFAULT_DECIMALS = st.number_input("Default decimals", min_value=1, max_value=8, value=int(st.session_state.get("default_decimals", 3)), step=1, key="global_default_decimals")
        st.session_state["default_decimals"] = DEFAULT_DECIMALS
        target_graph = st.selectbox("Graph to customize", PLOT_STYLE_KEYS, index=0, key="target_graph_style")
        current_cfg = safe_get_plot_cfg(target_graph)
        st.caption("Direct click-to-style on a matplotlib figure is not reliable in Streamlit. Use the graph selector above to apply formatting to one graph at a time.")

        st.markdown("**Sizes**")
        s1, s2, s3 = st.columns(3)
        with s1:
            fig_w = st.number_input("Figure width", min_value=3.0, max_value=16.0, value=float(current_cfg.get("fig_w", 6.8)), step=0.2, key=f"{target_graph}_fig_w")
            marker_size = st.number_input("Marker size", min_value=1, max_value=200, value=int(current_cfg.get("marker_size", 32)), step=1, key=f"{target_graph}_ms")
            font_size = st.number_input("Font size", min_value=6, max_value=24, value=int(current_cfg.get("font_size", 10)), step=1, key=f"{target_graph}_font_size")
        with s2:
            fig_h = st.number_input("Figure height", min_value=2.5, max_value=12.0, value=float(current_cfg.get("fig_h", 4.3)), step=0.2, key=f"{target_graph}_fig_h")
            line_width = st.number_input("Line width", min_value=0.2, max_value=6.0, value=float(current_cfg.get("line_width", 1.8)), step=0.1, key=f"{target_graph}_lw")
            label_size = st.number_input("Axis label size", min_value=6, max_value=28, value=int(current_cfg.get("label_size", 10)), step=1, key=f"{target_graph}_label_size")
        with s3:
            aux_line_width = st.number_input("Aux line width", min_value=0.2, max_value=6.0, value=float(current_cfg.get("aux_line_width", 1.2)), step=0.1, key=f"{target_graph}_alw")
            border_width = st.number_input("Border size", min_value=0.2, max_value=5.0, value=float(current_cfg.get("border_width", 1.0)), step=0.1, key=f"{target_graph}_bw")
            title_size = st.number_input("Title size", min_value=6, max_value=30, value=int(current_cfg.get("title_size", 12)), step=1, key=f"{target_graph}_title_size")

        st.markdown("**Types**")
        t1, t2, t3, t4 = st.columns(4)
        marker_options = ["o", "s", "^", "D", "v", "P", "X", "*", "+", "x"]
        line_names = list(LINE_STYLE_MAP.keys())
        axis_type_options = ["standard", "boxed", "left-bottom only"]
        font_opts = ["DejaVu Sans", "Arial", "Helvetica", "Times New Roman", "Courier New"]
        with t1:
            marker_style = st.selectbox("Marker type", marker_options, index=marker_options.index(current_cfg.get("marker_style", "o")) if current_cfg.get("marker_style", "o") in marker_options else 0, key=f"{target_graph}_marker_style")
        with t2:
            line_style_name = st.selectbox("Line type", line_names, index=line_names.index(next((k for k,v in LINE_STYLE_MAP.items() if v == current_cfg.get("line_style", "-")), "Solid")), key=f"{target_graph}_line_style")
        with t3:
            axis_type = st.selectbox("Axis type", axis_type_options, index=axis_type_options.index(current_cfg.get("axis_type", "standard")) if current_cfg.get("axis_type", "standard") in axis_type_options else 0, key=f"{target_graph}_axis_type")
        with t4:
            font_family = st.selectbox("Font type", font_opts, index=font_opts.index(current_cfg.get("font_family", "DejaVu Sans")) if current_cfg.get("font_family", "DejaVu Sans") in font_opts else 0, key=f"{target_graph}_font_family")

        st.markdown("**Colors**")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            marker_color = st.color_picker("Marker color", value=current_cfg.get("marker_color", current_cfg.get("primary_color", "#1f77b4")), key=f"{target_graph}_marker_color")
        with c2:
            line_color = st.color_picker("Line color", value=current_cfg.get("line_color", current_cfg.get("primary_color", "#1f77b4")), key=f"{target_graph}_line_color")
        with c3:
            border_color = st.color_picker("Border color", value=current_cfg.get("border_color", "#111827"), key=f"{target_graph}_border_color")
        with c4:
            font_color = st.color_picker("Font color", value=current_cfg.get("font_color", "#111827"), key=f"{target_graph}_font_color")
        c5, c6, c7 = st.columns(3)
        with c5:
            band_color = st.color_picker("Band / fill color", value=current_cfg.get("band_color", "#93c5fd"), key=f"{target_graph}_band_color")
        with c6:
            secondary_color = st.color_picker("Secondary color", value=current_cfg.get("secondary_color", "#ff7f0e"), key=f"{target_graph}_secondary_color")
        with c7:
            tertiary_color = st.color_picker("Tertiary color", value=current_cfg.get("tertiary_color", "#2ca02c"), key=f"{target_graph}_tertiary_color")

        st.markdown("**Axes / legend**")
        a1, a2, a3, a4 = st.columns(4)
        with a1:
            tick_dir = st.selectbox("Tick direction", ["out", "in", "inout"], index=["out", "in", "inout"].index(current_cfg.get("tick_dir", "out")), key=f"{target_graph}_tick_dir")
            tick_len = st.number_input("Tick size", min_value=0, max_value=20, value=int(current_cfg.get("tick_len", 4)), step=1, key=f"{target_graph}_tick_len")
        with a2:
            show_legend = st.checkbox("Show legend", value=bool(current_cfg.get("show_legend", True)), key=f"{target_graph}_show_legend")
            legend_opts = ["best", "upper right", "upper left", "lower right", "lower left", "center left", "center right", "lower center", "upper center"]
            legend_loc = st.selectbox("Legend location", legend_opts, index=legend_opts.index(current_cfg.get("legend_loc", "best")) if current_cfg.get("legend_loc", "best") in legend_opts else 0, key=f"{target_graph}_legend_loc")
        with a3:
            grid_alpha = st.number_input("Grid transparency", min_value=0.0, max_value=1.0, value=float(current_cfg.get("grid_alpha", 0.00)), step=0.05, key=f"{target_graph}_ga")
            show_top = st.checkbox("Show top border", value=bool(current_cfg.get("show_top", True)), key=f"{target_graph}_top")
        with a4:
            arrow_size = st.number_input("Arrow size", min_value=0.001, max_value=0.3, value=float(current_cfg.get("arrow_size", 0.025)), step=0.005, key=f"{target_graph}_arrow")
            show_right = st.checkbox("Show right border", value=bool(current_cfg.get("show_right", True)), key=f"{target_graph}_right")

        st.markdown("**Axis limits (leave blank for automatic)**")
        x1, x2, y1, y2 = st.columns(4)
        x_min_cfg = x1.text_input("X min", value="" if current_cfg.get("x_min") in [None, ""] else str(current_cfg.get("x_min")), key=f"{target_graph}_xmin")
        x_max_cfg = x2.text_input("X max", value="" if current_cfg.get("x_max") in [None, ""] else str(current_cfg.get("x_max")), key=f"{target_graph}_xmax")
        y_min_cfg = y1.text_input("Y min", value="" if current_cfg.get("y_min") in [None, ""] else str(current_cfg.get("y_min")), key=f"{target_graph}_ymin")
        y_max_cfg = y2.text_input("Y max", value="" if current_cfg.get("y_max") in [None, ""] else str(current_cfg.get("y_max")), key=f"{target_graph}_ymax")

        st.session_state["plot_style_cfg"][target_graph] = {
            "fig_w": fig_w, "fig_h": fig_h,
            "show_legend": show_legend, "legend_loc": legend_loc,
            "primary_color": line_color, "secondary_color": secondary_color, "tertiary_color": tertiary_color,
            "marker_color": marker_color, "line_color": line_color, "band_color": band_color,
            "border_color": border_color, "font_color": font_color,
            "grid_alpha": grid_alpha,
            "line_style": LINE_STYLE_MAP[line_style_name], "aux_line_style": current_cfg.get("aux_line_style", "--"),
            "line_width": line_width, "aux_line_width": aux_line_width,
            "marker_size": marker_size, "marker_style": marker_style,
            "tick_dir": tick_dir, "tick_len": tick_len, "border_width": border_width,
            "show_top": show_top, "show_right": show_right, "axis_type": axis_type,
            "font_family": font_family, "font_size": font_size, "title_size": title_size, "label_size": label_size,
            "tick_label_size": max(6, font_size - 1),
            "x_min": x_min_cfg.strip(), "x_max": x_max_cfg.strip(), "y_min": y_min_cfg.strip(), "y_max": y_max_cfg.strip(),
            "arrow_size": arrow_size,
        }
        r1, r2 = st.columns(2)
        with r1:
            if st.button("Reset selected graph style", key=f"reset_style_{target_graph}"):
                st.session_state["plot_style_cfg"][target_graph] = {} if target_graph != "All graphs" else DEFAULT_STYLE_CFG.copy()
                st.rerun()
        with r2:
            if st.button("Reset all graph styles", key="reset_all_graph_styles"):
                st.session_state["plot_style_cfg"] = {k: {} for k in PLOT_STYLE_KEYS}
                st.session_state["plot_style_cfg"]["All graphs"] = DEFAULT_STYLE_CFG.copy()
                st.rerun()

    _all_cfg = safe_get_plot_cfg("All graphs")
    FIG_W = _all_cfg["fig_w"]; FIG_H = _all_cfg["fig_h"]; SHOW_LEGEND = _all_cfg["show_legend"]; LEGEND_LOC = _all_cfg["legend_loc"]
    PRIMARY_COLOR = _all_cfg["line_color"]; SECONDARY_COLOR = _all_cfg["secondary_color"]; TERTIARY_COLOR = _all_cfg["tertiary_color"]
    BAND_COLOR = _all_cfg["band_color"]; GRID_ALPHA = _all_cfg["grid_alpha"]; MARKER_SIZE = _all_cfg["marker_size"]
    FIT_LINE_COLOR = _all_cfg["line_color"]; FIT_LINE_STYLE = _all_cfg["line_style"]; LINE_WIDTH = _all_cfg["line_width"]; AREA_ALPHA = 0.16
    CI_LINE_STYLE = _all_cfg["aux_line_style"]; PI_LINE_STYLE = _all_cfg["aux_line_style"]; SPEC_LINE_STYLE = _all_cfg["aux_line_style"]; ARROW_SIZE = _all_cfg["arrow_size"]


def fig_to_png_bytes(fig):
    bio = BytesIO()
    fig.savefig(bio, format="png", dpi=220, bbox_inches="tight", facecolor="white")
    bio.seek(0)
    return bio.getvalue()


def app_header(title, subtitle=""):
    st.markdown(f"<div class='app-header'><div class='app-title'>{title}</div><div class='app-sub'>{subtitle}</div></div>", unsafe_allow_html=True)


def info_box(text):
    st.markdown(f"<div class='report-caption'>{text}</div>", unsafe_allow_html=True)


def to_numeric(series):
    return pd.to_numeric(series.astype(str).str.strip().str.replace("%", "", regex=False), errors="coerce")


def parse_pasted_table(text, header=True):
    text = str(text).strip()
    if not text:
        return None
    parsers = [
        lambda s: pd.read_csv(StringIO(s), sep="\t", engine="python", header=0 if header else None),
        lambda s: pd.read_csv(StringIO(s), sep=",", engine="python", header=0 if header else None),
        lambda s: pd.read_csv(StringIO(s), sep=";", engine="python", header=0 if header else None),
        lambda s: pd.read_csv(StringIO(s), sep=r"\s+", engine="python", header=0 if header else None),
    ]
    for parser in parsers:
        try:
            df = parser(text)
            if df is not None and df.shape[1] >= 1:
                if header:
                    df.columns = [str(c).strip() for c in df.columns]
                return df.dropna(how="all").reset_index(drop=True)
        except Exception:
            continue
    return None


def parse_xy(text):
    raw = parse_pasted_table(text, header=False)
    if raw is None or raw.shape[1] < 2:
        raise ValueError("Paste at least two columns from Excel.")
    raw = raw.iloc[:, :2].copy()
    x_label, y_label = "X", "Y"
    first_row = raw.iloc[0].astype(str)
    first_row_numeric = pd.to_numeric(first_row.str.replace("%", "", regex=False), errors="coerce")
    if first_row_numeric.isna().any():
        x_label = str(raw.iloc[0, 0]).strip() or "X"
        y_label = str(raw.iloc[0, 1]).strip() or "Y"
        raw = raw.iloc[1:].reset_index(drop=True)
    raw.columns = ["x", "y"]
    raw["x"] = to_numeric(raw["x"])
    raw["y"] = to_numeric(raw["y"])
    raw = raw.dropna().sort_values("x").reset_index(drop=True)
    if len(raw) < 3 or raw["x"].nunique() < 2:
        raise ValueError("At least 3 valid rows and 2 unique X values are required.")
    return raw, x_label, y_label


def parse_x_values(text):
    text = str(text).strip()
    if not text:
        return np.array([])
    vals = []
    for part in re.split(r"[\s,;\t]+", text):
        if part:
            vals.append(float(part))
    return np.array(vals, dtype=float)


def parse_optional_float(txt):
    txt = str(txt).strip()
    return None if txt == "" else float(txt)


def get_numeric_columns(df, min_nonempty=2, required_numeric_ratio=0.95):
    out = []
    for col in df.columns:
        raw = df[col]
        raw_str = raw.astype(str).str.strip()
        nonempty_mask = raw.notna() & raw_str.ne("") & raw_str.str.lower().ne("nan")
        if nonempty_mask.sum() < min_nonempty:
            continue
        converted = pd.to_numeric(raw_str.str.replace("%", "", regex=False), errors="coerce")
        if converted[nonempty_mask].notna().mean() >= required_numeric_ratio:
            out.append(col)
    return out


def fmt_p(p):
    if pd.isna(p):
        return "-"
    return "<0.001" if p < 0.001 else f"{p:.3f}"

def plot_regression_advanced(
    data_df,
    model,
    grid_df,
    confidence=0.95,
    interval="pi",
    side="upper",
    title="",
    xlabel="Time",
    ylabel="Response",
    point_label="Data",
    y_suffix="%",
    spec_enabled=False,
    spec_limit=None,
    spec_label="US",
    crossing_on="auto",
):
    cfg = get_plot_cfg("Regression Analysis")
    x = data_df["x"].to_numpy()
    y = data_df["y"].to_numpy()
    fig, ax = plt.subplots(figsize=(cfg["fig_w"], cfg["fig_h"]))
    main_color = cfg.get("line_color", cfg["primary_color"])
    point_color = cfg.get("marker_color", cfg["primary_color"])
    pi_color = cfg["secondary_color"]
    ci_color = cfg["band_color"]
    lw = cfg["line_width"]
    ls = cfg["line_style"]
    aux_ls = cfg["aux_line_style"]
    ms = cfg["marker_size"]
    area_alpha = 0.18

    ax.scatter(x, y, color=point_color, s=ms, alpha=0.85, label=point_label, zorder=3, marker=cfg.get("marker_style", "o"))
    ax.plot(grid_df["x"], grid_df["fit"], color=main_color, lw=lw, ls=ls, label="Fitted Line")

    if interval in ["ci", "both"]:
        if side == "two-sided":
            ax.fill_between(grid_df["x"], grid_df["ci_lower"], grid_df["ci_upper"], color=ci_color, alpha=area_alpha, label="Confidence Interval (CI)")
            ax.plot(grid_df["x"], grid_df["ci_upper"], color=ci_color, ls=aux_ls, lw=cfg["aux_line_width"])
            ax.plot(grid_df["x"], grid_df["ci_lower"], color=ci_color, ls=aux_ls, lw=cfg["aux_line_width"])
        elif side == "upper":
            ax.fill_between(grid_df["x"], grid_df["fit"], grid_df["ci_upper"], color=ci_color, alpha=area_alpha, label="Upper CI")
            ax.plot(grid_df["x"], grid_df["ci_upper"], color=ci_color, ls=aux_ls, lw=cfg["aux_line_width"], label="_nolegend_")
        else:
            ax.fill_between(grid_df["x"], grid_df["ci_lower"], grid_df["fit"], color=ci_color, alpha=area_alpha, label="Lower CI")
            ax.plot(grid_df["x"], grid_df["ci_lower"], color=ci_color, ls=aux_ls, lw=cfg["aux_line_width"], label="_nolegend_")

    if interval in ["pi", "both"]:
        pa = max(area_alpha - 0.05, 0.05)
        if side == "two-sided":
            ax.fill_between(grid_df["x"], grid_df["pi_lower"], grid_df["pi_upper"], color=pi_color, alpha=pa, label="Prediction Interval (PI)")
            ax.plot(grid_df["x"], grid_df["pi_upper"], color=pi_color, ls=aux_ls, lw=cfg["aux_line_width"])
            ax.plot(grid_df["x"], grid_df["pi_lower"], color=pi_color, ls=aux_ls, lw=cfg["aux_line_width"])
        elif side == "upper":
            ax.fill_between(grid_df["x"], grid_df["fit"], grid_df["pi_upper"], color=pi_color, alpha=pa, label="Upper PI")
            ax.plot(grid_df["x"], grid_df["pi_upper"], color=pi_color, ls=aux_ls, lw=cfg["aux_line_width"], label="_nolegend_")
        else:
            ax.fill_between(grid_df["x"], grid_df["pi_lower"], grid_df["fit"], color=pi_color, alpha=pa, label="Lower PI")
            ax.plot(grid_df["x"], grid_df["pi_lower"], color=pi_color, ls=aux_ls, lw=cfg["aux_line_width"], label="_nolegend_")

    crossing_x = None
    if spec_enabled and spec_limit is not None:
        ax.axhline(spec_limit, color=cfg["tertiary_color"], ls=aux_ls, lw=lw, label=f"Limit ({spec_label})")
        curve_map = {
            "fit": grid_df["fit"].to_numpy(),
            "ci_upper": grid_df["ci_upper"].to_numpy(),
            "ci_lower": grid_df["ci_lower"].to_numpy(),
            "pi_upper": grid_df["pi_upper"].to_numpy(),
            "pi_lower": grid_df["pi_lower"].to_numpy(),
        }
        if crossing_on == "auto":
            if interval in ["both", "pi"]:
                crossing_on = "pi_upper" if side == "upper" else "pi_lower" if side == "lower" else "pi_upper"
            else:
                crossing_on = "ci_upper" if side == "upper" else "ci_lower" if side == "lower" else "ci_upper"
        if crossing_on in curve_map:
            crossing_x = reg_find_crossing(grid_df["x"].to_numpy(), curve_map[crossing_on], spec_limit)
            if crossing_x is not None:
                ax.axvline(crossing_x, color=cfg["tertiary_color"], ls=aux_ls, lw=cfg["aux_line_width"])

        xmin = float(grid_df["x"].min())
        xmax = float(grid_df["x"].max())
        ymax_data = max(float(grid_df["fit"].max()), float(grid_df["ci_upper"].max()), float(grid_df["pi_upper"].max()), float(y.max()))
        ymin_data = min(float(grid_df["fit"].min()), float(grid_df["ci_lower"].min()), float(grid_df["pi_lower"].min()), float(y.min()))
        pad = 0.02 * (ymax_data - ymin_data if ymax_data > ymin_data else 1)
        suffix = y_suffix or ""
        ax.text(xmin + (xmax - xmin) * 0.02, spec_limit + pad, f"{spec_label} = {spec_limit:.1f}{suffix}",
                ha="left", va="bottom", fontsize=11, color=cfg["tertiary_color"], weight="bold",
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=2))
        if crossing_x is not None:
            ax.text(crossing_x, ymin_data + pad, f" {crossing_x:.2f}",
                    color=cfg["tertiary_color"], ha="left", va="bottom", fontsize=11, weight="bold",
                    bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=2))

    if y_suffix:
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, pos: f"{v:.1f}{y_suffix}"))

    if not title.strip():
        s1 = {"upper": "Upper One-Sided", "lower": "Lower One-Sided", "two-sided": "Two-Sided"}[side]
        s2 = {"ci": "Confidence Intervals", "pi": "Prediction Intervals", "both": "Confidence and Prediction Intervals"}[interval]
        title = f"{s1} {s2} ({confidence:.0%})"

    apply_ax_style(ax, title, xlabel, ylabel, legend=True, plot_key="Regression Analysis")
    return fig, crossing_x

def _auto_explanation_text(label, kind="table"):
    label = str(label or "").strip()
    if not label:
        return f"The {kind} below summarizes the current analysis output."
    lower = label.rstrip(":.")
    prefix = "The table below" if kind == "table" else "The figure below"
    return f"{prefix} presents {lower.lower()} and is intended to support interpretation of the current analysis."


def show_figure(fig, caption="", explanation=None):
    info_box(explanation or _auto_explanation_text(caption or "current graph", kind="figure"))
    st.pyplot(fig)


def report_table(df, caption="", decimals=None):
    decimals = DEFAULT_DECIMALS if decimals is None else decimals
    info_box(_auto_explanation_text(caption or "current table", kind="table"))
    styled = df.style.hide(axis="index").set_caption(caption).set_table_styles([
        {"selector": "caption", "props": [("text-align", "left"), ("font-size", "1rem"), ("font-weight", "700"), ("margin-bottom", "0.55rem")]},
        {"selector": "thead th", "props": [("border-top", "2px solid #111827"), ("border-bottom", "1px solid #111827"), ("padding", "8px 12px"), ("text-align", "center"), ("background-color", "#f8fafc")]},
        {"selector": "tbody td", "props": [("padding", "8px 12px"), ("text-align", "center")]},
        {"selector": "tbody tr:last-child td", "props": [("border-bottom", "2px solid #111827")]},
    ]).format(precision=decimals, na_rep="-")
    st.markdown(f"<div class='report-table'>{styled.to_html()}</div>", unsafe_allow_html=True)


def make_excel_bytes(sheet_map):
    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        for sheet_name, df in sheet_map.items():
            safe = re.sub(r"[^A-Za-z0-9 _-]", "", sheet_name)[:31] or "Sheet1"
            df.copy().to_excel(writer, sheet_name=safe, index=False)
            ws = writer.sheets[safe]
            for col_cells in ws.columns:
                max_len = max((len(str(cell.value)) if cell.value is not None else 0 for cell in col_cells), default=0)
                ws.column_dimensions[col_cells[0].column_letter].width = min(max_len + 2, 28)
    bio.seek(0)
    return bio.getvalue()


def _pdf_table(df, styles, title, decimals=3, max_rows=40):
    story = [Paragraph(title, styles["Heading3"])]
    if len(df) > max_rows:
        story.append(Paragraph(f"Table truncated to first {max_rows} rows for compact reporting.", styles["BodyText"]))
        df = df.head(max_rows)
    fmt_df = df.copy()
    for c in fmt_df.columns:
        if pd.api.types.is_numeric_dtype(fmt_df[c]):
            fmt_df[c] = fmt_df[c].map(lambda x: "-" if pd.isna(x) else f"{x:.{decimals}f}")
        else:
            fmt_df[c] = fmt_df[c].fillna("-").astype(str)
    data = [list(fmt_df.columns)] + fmt_df.values.tolist()
    ncols = max(1, len(fmt_df.columns))
    tbl = Table(data, repeatRows=1, colWidths=[27.5 * cm / ncols] * ncols)
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F8FAFC")), ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8), ("LEADING", (0, 0), (-1, -1), 10), ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LINEABOVE", (0, 0), (-1, 0), 1.2, colors.black), ("LINEBELOW", (0, 0), (-1, 0), 0.8, colors.black),
        ("LINEBELOW", (0, -1), (-1, -1), 1.2, colors.black),
    ]))
    story.extend([tbl, Spacer(1, 0.35 * cm)])
    return story


def make_pdf_report(report_title, module_name, statistical_analysis, offer_text, python_tools, tables, figures, conclusion=None, decimals=3):
    bio = BytesIO()
    doc = SimpleDocTemplate(bio, pagesize=landscape(A4), leftMargin=1.2 * cm, rightMargin=1.2 * cm, topMargin=1.2 * cm, bottomMargin=1.1 * cm)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="SmallBody", parent=styles["BodyText"], fontSize=9, leading=12, alignment=TA_LEFT))
    story = [Paragraph(report_title, styles["Title"]), Spacer(1, 0.15 * cm), Paragraph(f"Module: <b>{module_name}</b>", styles["Heading2"]), Spacer(1, 0.15 * cm), Paragraph("Statistical Analysis", styles["Heading2"]), Paragraph(statistical_analysis, styles["SmallBody"]), Spacer(1, 0.15 * cm), Paragraph("What this analysis offers", styles["Heading2"]), Paragraph(offer_text, styles["SmallBody"]), Spacer(1, 0.15 * cm), Paragraph("Python tools used", styles["Heading2"]), Paragraph(python_tools, styles["SmallBody"])]
    if conclusion:
        story.extend([Spacer(1, 0.15 * cm), Paragraph("Conclusion", styles["Heading2"]), Paragraph(conclusion, styles["SmallBody"])])
    story.append(Spacer(1, 0.2 * cm))
    if tables:
        story.append(Paragraph("Tables", styles["Heading2"]))
        for caption, df in tables:
            story.extend(_pdf_table(df, styles, caption, decimals=decimals))
    if figures:
        story.extend([PageBreak(), Paragraph("Figures", styles["Heading2"])])
        for caption, fig_bytes in figures:
            story.extend([Paragraph(caption, styles["Heading3"]), Image(BytesIO(fig_bytes))])
            story[-1]._restrictSize(24.5 * cm, 13.5 * cm)
            story.append(Spacer(1, 0.3 * cm))
    doc.build(story)
    bio.seek(0)
    return bio.getvalue()


def export_results(prefix, report_title, module_name, statistical_analysis, offer_text, python_tools, table_map, figure_map=None, conclusion=None, decimals=None):
    decimals = DEFAULT_DECIMALS if decimals is None else decimals
    figure_map = figure_map or {}
    c1, c2 = st.columns(2)
    excel_bytes = make_excel_bytes(table_map)
    pdf_bytes = make_pdf_report(report_title, module_name, statistical_analysis, offer_text, python_tools, list(table_map.items()), list(figure_map.items()), conclusion=conclusion, decimals=decimals)
    with c1:
        st.download_button("Download Excel workbook", excel_bytes, file_name=f"{prefix}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    with c2:
        st.download_button("Download PDF-style report", pdf_bytes, file_name=f"{prefix}.pdf", mime="application/pdf")


def apply_ax_style(ax, title, xlabel, ylabel, legend=None, plot_key="All graphs"):
    cfg = safe_get_plot_cfg(plot_key)
    ax.set_title(title, fontsize=cfg["title_size"], color=cfg["font_color"], fontfamily=cfg["font_family"])
    ax.set_xlabel(xlabel, fontsize=cfg["label_size"], color=cfg["font_color"], fontfamily=cfg["font_family"])
    ax.set_ylabel(ylabel, fontsize=cfg["label_size"], color=cfg["font_color"], fontfamily=cfg["font_family"])
    ax.grid(True, alpha=cfg["grid_alpha"])
    ax.tick_params(direction=cfg["tick_dir"], length=cfg["tick_len"], width=cfg["border_width"], colors=cfg["font_color"], labelsize=cfg["tick_label_size"])
    for spine in ax.spines.values():
        spine.set_linewidth(cfg["border_width"])
        spine.set_color(cfg["border_color"])
    if cfg.get("axis_type") == "left-bottom only":
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    elif cfg.get("axis_type") == "boxed":
        ax.spines["top"].set_visible(True); ax.spines["right"].set_visible(True)
    else:
        ax.spines["top"].set_visible(cfg.get("show_top", True)); ax.spines["right"].set_visible(cfg.get("show_right", True))
    if cfg["x_min"] is not None or cfg["x_max"] is not None:
        ax.set_xlim(left=cfg["x_min"], right=cfg["x_max"])
    if cfg["y_min"] is not None or cfg["y_max"] is not None:
        ax.set_ylim(bottom=cfg["y_min"], top=cfg["y_max"])
    if legend is None:
        legend = cfg["show_legend"]
    if cfg["show_legend"] and legend:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(frameon=False, loc=cfg["legend_loc"])
    ax.figure.tight_layout(pad=1.0)


def residual_plot(fitted, residuals, xlabel="Fitted", ylabel="Residuals", title="Residuals vs fitted"):
    cfg = safe_get_plot_cfg("Residual plot")
    fig, ax = plt.subplots(figsize=(cfg["fig_w"], cfg["fig_h"]))
    ax.scatter(fitted, residuals, color=cfg.get("marker_color", cfg["primary_color"]), s=cfg["marker_size"], marker=cfg.get("marker_style", "o"))
    ax.axhline(0, color="#111827", ls=cfg["aux_line_style"], lw=cfg["aux_line_width"])
    apply_ax_style(ax, title, xlabel, ylabel, plot_key="Residual plot")
    return fig


def qq_plot(residuals, title="Normal probability plot of residuals"):
    cfg = safe_get_plot_cfg("Q-Q plot")
    fig, ax = plt.subplots(figsize=(cfg["fig_w"], cfg["fig_h"]))
    stats.probplot(residuals, dist="norm", plot=ax)
    if len(ax.lines) >= 2:
        ax.lines[0].set_marker(cfg.get("marker_style", "o")); ax.lines[0].set_linestyle("None"); ax.lines[0].set_color(cfg["primary_color"]); ax.lines[0].set_markersize(max(3, cfg["marker_size"] / 12))
        ax.lines[1].set_color(cfg["secondary_color"]); ax.lines[1].set_linestyle(cfg["aux_line_style"]); ax.lines[1].set_linewidth(cfg["aux_line_width"])
    apply_ax_style(ax, title, "Theoretical quantiles", "Ordered residuals", plot_key="Q-Q plot")
    return fig


def tol_interval_normal(x, coverage=0.99, confidence=0.95):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return np.nan, np.nan, np.nan
    mean = np.mean(x); sd = np.std(x, ddof=1); nu = len(x) - 1
    z_p = norm.ppf((1 + coverage) / 2); chi = stats.chi2.ppf(confidence, nu)
    if chi <= 0:
        return mean, np.nan, np.nan
    k = z_p * np.sqrt(nu * (1 + 1 / len(x)) / chi)
    return mean, mean - k * sd, mean + k * sd


def tolerance_interval_normal(data, p=0.95, conf=0.95, two_sided=True):
    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]
    if len(data) < 2:
        return np.nan, np.nan, np.nan
    if two_sided:
        return tol_interval_normal(data, coverage=p, confidence=conf)
    n = len(data); mean = np.mean(data); sd = np.std(data, ddof=1); zp = norm.ppf(p)
    k = nct.ppf(conf, n - 1, np.sqrt(n) * zp) / np.sqrt(n)
    return mean, mean - k * sd, mean + k * sd


def fit_linear(x, y):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    n = len(x); X = np.column_stack([np.ones(n), x]); XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y; intercept, slope = beta; fitted = X @ beta; resid = y - fitted
    df = n - 2; s = np.sqrt(np.sum(resid ** 2) / df); ss_tot = np.sum((y - y.mean()) ** 2); ss_res = np.sum(resid ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return {"intercept": intercept, "slope": slope, "XtX_inv": XtX_inv, "fitted": fitted, "resid": resid, "df": df, "s": s, "r2": r2}


def reg_fit_linear_model(x, y):
    model = fit_linear(x, y)
    model["x"] = np.asarray(x, dtype=float).ravel(); model["y"] = np.asarray(y, dtype=float).ravel(); model["y_fit"] = model["fitted"]
    return model


def reg_predict_with_intervals(model, x_values, confidence=0.95, side="upper"):
    x_values = np.asarray(x_values, dtype=float).ravel(); Xg = np.column_stack([np.ones(len(x_values)), x_values])
    beta = np.array([model["intercept"], model["slope"]]); yhat = Xg @ beta
    h = np.einsum("ij,jk,ik->i", Xg, model["XtX_inv"], Xg); se_mean = model["s"] * np.sqrt(h); se_pred = model["s"] * np.sqrt(1 + h)
    alpha = 1 - confidence; tcrit = t.ppf(1 - alpha / 2, model["df"]) if side == "two-sided" else t.ppf(confidence, model["df"])
    return pd.DataFrame({"x": x_values, "fit": yhat, "ci_lower": yhat - tcrit * se_mean, "ci_upper": yhat + tcrit * se_mean, "pi_lower": yhat - tcrit * se_pred, "pi_upper": yhat + tcrit * se_pred})


def reg_find_crossing(xv, yv, limit):
    d = yv - limit; idx = np.where(d[:-1] * d[1:] <= 0)[0]
    if len(idx) == 0:
        return None
    i = idx[0]; x1, x2 = xv[i], xv[i + 1]; y1, y2 = yv[i], yv[i + 1]
    return x1 if y2 == y1 else x1 + (limit - y1) * (x2 - x1) / (y2 - y1)

def plot_regression_advanced(
    data_df,
    model,
    grid_df,
    confidence=0.95,
    interval="pi",
    side="upper",
    title="",
    xlabel="Time",
    ylabel="Response",
    point_label="Data",
    y_suffix="%",
    spec_enabled=False,
    spec_limit=None,
    spec_label="US",
    crossing_on="auto",
):
    cfg = get_plot_cfg("Regression Analysis")
    x = data_df["x"].to_numpy()
    y = data_df["y"].to_numpy()
    fig, ax = plt.subplots(figsize=(cfg["fig_w"], cfg["fig_h"]))
    main_color = cfg.get("line_color", cfg["primary_color"])
    point_color = cfg.get("marker_color", cfg["primary_color"])
    pi_color = cfg["secondary_color"]
    ci_color = cfg["band_color"]
    lw = cfg["line_width"]
    ls = cfg["line_style"]
    aux_ls = cfg["aux_line_style"]
    ms = cfg["marker_size"]
    area_alpha = 0.18

    ax.scatter(x, y, color=point_color, s=ms, alpha=0.85, label=point_label, zorder=3, marker=cfg.get("marker_style", "o"))
    ax.plot(grid_df["x"], grid_df["fit"], color=main_color, lw=lw, ls=ls, label="Fitted Line")

    if interval in ["ci", "both"]:
        if side == "two-sided":
            ax.fill_between(grid_df["x"], grid_df["ci_lower"], grid_df["ci_upper"], color=ci_color, alpha=area_alpha, label="Confidence Interval (CI)")
            ax.plot(grid_df["x"], grid_df["ci_upper"], color=ci_color, ls=aux_ls, lw=cfg["aux_line_width"])
            ax.plot(grid_df["x"], grid_df["ci_lower"], color=ci_color, ls=aux_ls, lw=cfg["aux_line_width"])
        elif side == "upper":
            ax.fill_between(grid_df["x"], grid_df["fit"], grid_df["ci_upper"], color=ci_color, alpha=area_alpha, label="Upper CI")
            ax.plot(grid_df["x"], grid_df["ci_upper"], color=ci_color, ls=aux_ls, lw=cfg["aux_line_width"], label="_nolegend_")
        else:
            ax.fill_between(grid_df["x"], grid_df["ci_lower"], grid_df["fit"], color=ci_color, alpha=area_alpha, label="Lower CI")
            ax.plot(grid_df["x"], grid_df["ci_lower"], color=ci_color, ls=aux_ls, lw=cfg["aux_line_width"], label="_nolegend_")

    if interval in ["pi", "both"]:
        pa = max(area_alpha - 0.05, 0.05)
        if side == "two-sided":
            ax.fill_between(grid_df["x"], grid_df["pi_lower"], grid_df["pi_upper"], color=pi_color, alpha=pa, label="Prediction Interval (PI)")
            ax.plot(grid_df["x"], grid_df["pi_upper"], color=pi_color, ls=aux_ls, lw=cfg["aux_line_width"])
            ax.plot(grid_df["x"], grid_df["pi_lower"], color=pi_color, ls=aux_ls, lw=cfg["aux_line_width"])
        elif side == "upper":
            ax.fill_between(grid_df["x"], grid_df["fit"], grid_df["pi_upper"], color=pi_color, alpha=pa, label="Upper PI")
            ax.plot(grid_df["x"], grid_df["pi_upper"], color=pi_color, ls=aux_ls, lw=cfg["aux_line_width"], label="_nolegend_")
        else:
            ax.fill_between(grid_df["x"], grid_df["pi_lower"], grid_df["fit"], color=pi_color, alpha=pa, label="Lower PI")
            ax.plot(grid_df["x"], grid_df["pi_lower"], color=pi_color, ls=aux_ls, lw=cfg["aux_line_width"], label="_nolegend_")

    crossing_x = None
    if spec_enabled and spec_limit is not None:
        ax.axhline(spec_limit, color=cfg["tertiary_color"], ls=aux_ls, lw=lw, label=f"Limit ({spec_label})")
        curve_map = {
            "fit": grid_df["fit"].to_numpy(),
            "ci_upper": grid_df["ci_upper"].to_numpy(),
            "ci_lower": grid_df["ci_lower"].to_numpy(),
            "pi_upper": grid_df["pi_upper"].to_numpy(),
            "pi_lower": grid_df["pi_lower"].to_numpy(),
        }
        if crossing_on == "auto":
            if interval in ["both", "pi"]:
                crossing_on = "pi_upper" if side == "upper" else "pi_lower" if side == "lower" else "pi_upper"
            else:
                crossing_on = "ci_upper" if side == "upper" else "ci_lower" if side == "lower" else "ci_upper"
        if crossing_on in curve_map:
            crossing_x = reg_find_crossing(grid_df["x"].to_numpy(), curve_map[crossing_on], spec_limit)
            if crossing_x is not None:
                ax.axvline(crossing_x, color=cfg["tertiary_color"], ls=aux_ls, lw=cfg["aux_line_width"])

        xmin = float(grid_df["x"].min())
        xmax = float(grid_df["x"].max())
        ymax_data = max(float(grid_df["fit"].max()), float(grid_df["ci_upper"].max()), float(grid_df["pi_upper"].max()), float(y.max()))
        ymin_data = min(float(grid_df["fit"].min()), float(grid_df["ci_lower"].min()), float(grid_df["pi_lower"].min()), float(y.min()))
        pad = 0.02 * (ymax_data - ymin_data if ymax_data > ymin_data else 1)
        suffix = y_suffix or ""
        ax.text(xmin + (xmax - xmin) * 0.02, spec_limit + pad, f"{spec_label} = {spec_limit:.1f}{suffix}",
                ha="left", va="bottom", fontsize=11, color=cfg["tertiary_color"], weight="bold",
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=2))
        if crossing_x is not None:
            ax.text(crossing_x, ymin_data + pad, f" {crossing_x:.2f}",
                    color=cfg["tertiary_color"], ha="left", va="bottom", fontsize=11, weight="bold",
                    bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=2))

    if y_suffix:
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, pos: f"{v:.1f}{y_suffix}"))

    if not title.strip():
        s1 = {"upper": "Upper One-Sided", "lower": "Lower One-Sided", "two-sided": "Two-Sided"}[side]
        s2 = {"ci": "Confidence Intervals", "pi": "Prediction Intervals", "both": "Confidence and Prediction Intervals"}[interval]
        title = f"{s1} {s2} ({confidence:.0%})"

    apply_ax_style(ax, title, xlabel, ylabel, legend=True, plot_key="Regression Analysis")
    return fig, crossing_x


def plot_regression_advanced(data_df, model, grid_df, confidence=0.95, interval="pi", side="upper", title="", xlabel="Time", ylabel="Response", point_label="Data", y_suffix="%", spec_enabled=False, spec_limit=None, spec_label="US", crossing_on="auto"):
    cfg = safe_get_plot_cfg("Regression Analysis")
    x = data_df["x"].to_numpy(); y = data_df["y"].to_numpy()
    fig, ax = plt.subplots(figsize=(cfg["fig_w"], cfg["fig_h"]))
    main_color = cfg["primary_color"]; pi_color = cfg["secondary_color"]; ci_color = cfg["band_color"]
    ax.scatter(x, y, color=main_color, s=cfg["marker_size"], alpha=0.85, label=point_label, zorder=3)
    ax.plot(grid_df["x"], grid_df["fit"], color=main_color, lw=cfg["line_width"], ls=cfg["line_style"], label="Fitted Line")
    if interval in ["ci", "both"]:
        if side == "two-sided":
            ax.fill_between(grid_df["x"], grid_df["ci_lower"], grid_df["ci_upper"], color=ci_color, alpha=0.18, label="Confidence Interval (CI)")
            ax.plot(grid_df["x"], grid_df["ci_upper"], color=ci_color, ls=cfg["aux_line_style"], lw=cfg["aux_line_width"])
            ax.plot(grid_df["x"], grid_df["ci_lower"], color=ci_color, ls=cfg["aux_line_style"], lw=cfg["aux_line_width"])
        elif side == "upper":
            ax.fill_between(grid_df["x"], grid_df["fit"], grid_df["ci_upper"], color=ci_color, alpha=0.18, label="Upper CI")
            ax.plot(grid_df["x"], grid_df["ci_upper"], color=ci_color, ls=cfg["aux_line_style"], lw=cfg["aux_line_width"])
        else:
            ax.fill_between(grid_df["x"], grid_df["ci_lower"], grid_df["fit"], color=ci_color, alpha=0.18, label="Lower CI")
            ax.plot(grid_df["x"], grid_df["ci_lower"], color=ci_color, ls=cfg["aux_line_style"], lw=cfg["aux_line_width"])
    if interval in ["pi", "both"]:
        if side == "two-sided":
            ax.fill_between(grid_df["x"], grid_df["pi_lower"], grid_df["pi_upper"], color=pi_color, alpha=0.12, label="Prediction Interval (PI)")
            ax.plot(grid_df["x"], grid_df["pi_upper"], color=pi_color, ls=cfg["aux_line_style"], lw=cfg["aux_line_width"])
            ax.plot(grid_df["x"], grid_df["pi_lower"], color=pi_color, ls=cfg["aux_line_style"], lw=cfg["aux_line_width"])
        elif side == "upper":
            ax.fill_between(grid_df["x"], grid_df["fit"], grid_df["pi_upper"], color=pi_color, alpha=0.12, label="Upper PI")
            ax.plot(grid_df["x"], grid_df["pi_upper"], color=pi_color, ls=cfg["aux_line_style"], lw=cfg["aux_line_width"])
        else:
            ax.fill_between(grid_df["x"], grid_df["pi_lower"], grid_df["fit"], color=pi_color, alpha=0.12, label="Lower PI")
            ax.plot(grid_df["x"], grid_df["pi_lower"], color=pi_color, ls=cfg["aux_line_style"], lw=cfg["aux_line_width"])
    crossing_x = None
    if spec_enabled and spec_limit is not None:
        ax.axhline(spec_limit, color=cfg["tertiary_color"], ls=cfg["aux_line_style"], lw=cfg["line_width"], label=f"Limit ({spec_label})")
        curve_map = {"fit": grid_df["fit"].to_numpy(), "ci_upper": grid_df["ci_upper"].to_numpy(), "ci_lower": grid_df["ci_lower"].to_numpy(), "pi_upper": grid_df["pi_upper"].to_numpy(), "pi_lower": grid_df["pi_lower"].to_numpy()}
        if crossing_on == "auto":
            crossing_on = "pi_upper" if interval in ["both", "pi"] and side != "lower" else "pi_lower" if interval in ["both", "pi"] else "ci_upper" if side != "lower" else "ci_lower"
        crossing_x = reg_find_crossing(grid_df["x"].to_numpy(), curve_map[crossing_on], spec_limit) if crossing_on in curve_map else None
        if crossing_x is not None:
            ax.axvline(crossing_x, color=cfg["tertiary_color"], ls=cfg["aux_line_style"], lw=cfg["aux_line_width"])
    if y_suffix:
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, pos: f"{v:.1f}{y_suffix}"))
    if not str(title).strip():
        title = f"{ {'upper':'Upper','lower':'Lower','two-sided':'Two-Sided'}[side] } { {'ci':'Confidence Intervals','pi':'Prediction Intervals','both':'Confidence and Prediction Intervals'}[interval] } ({confidence:.0%})"
    apply_ax_style(ax, title, xlabel, ylabel, legend=True, plot_key="Regression Analysis")
    return fig, crossing_x



def doe_formula(safe_factors, model_type="interaction"):
    terms = list(safe_factors)
    if model_type in ["interaction", "quadratic"]:
        for i in range(len(safe_factors)):
            for j in range(i + 1, len(safe_factors)):
                terms.append(f"{safe_factors[i]}:{safe_factors[j]}")
    if model_type == "quadratic":
        for f in safe_factors:
            terms.append(f"I({f}**2)")
    return "Response ~ " + " + ".join(terms)

def draw_conf_ellipse(scores, ax, edgecolor=PRIMARY_COLOR, facecolor=None, plot_key="PCA score plot"):
    scores = np.asarray(scores, dtype=float)
    if scores.shape[0] < 3:
        return
    cov = np.cov(scores.T); vals, vecs = np.linalg.eigh(cov); order = vals.argsort()[::-1]; vals, vecs = vals[order], vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1])); q = stats.chi2.ppf(0.95, 2); width, height = 2 * np.sqrt(vals * q)
    cfg = safe_get_plot_cfg(plot_key)
    ell = Ellipse(xy=np.mean(scores, axis=0), width=width, height=height, angle=theta, edgecolor=edgecolor, facecolor=facecolor if facecolor is not None else edgecolor, alpha=0.12, lw=max(0.8, cfg["aux_line_width"] * 0.9), ls=cfg["aux_line_style"])
    ax.add_patch(ell)

# Dissolution helpers

def dis_make_unique(names):
    out = []; seen = {}
    for i, n in enumerate(names):
        n = str(n).strip()
        if n == "" or n.lower() == "nan":
            n = f"Col{i+1}"
        if n in seen:
            seen[n] += 1; n = f"{n}_{seen[n]}"
        else:
            seen[n] = 1
        out.append(n)
    return out


def dis_parse_profile_table(text):
    text = str(text).strip()
    if not text:
        raise ValueError("Paste a dissolution table.")
    parsers = [
        lambda s: pd.read_csv(StringIO(s), sep="\t", header=None, engine="python"),
        lambda s: pd.read_csv(StringIO(s), sep=",", header=None, engine="python"),
        lambda s: pd.read_csv(StringIO(s), sep=";", header=None, engine="python"),
    ]
    df_raw = None
    for parser in parsers:
        try:
            trial = parser(text)
            if trial.shape[1] >= 2:
                df_raw = trial.copy(); break
        except Exception:
            pass
    if df_raw is None or df_raw.shape[1] < 2:
        raise ValueError("Could not read the pasted table. Use at least 2 columns: Time and one or more units.")
    df_raw = df_raw.dropna(how="all").reset_index(drop=True)
    first_row = df_raw.iloc[0].astype(str).str.strip(); first_row_numeric = pd.to_numeric(first_row, errors="coerce").notna().all()
    if first_row_numeric:
        df = df_raw.copy(); df.columns = ["Time"] + [f"Unit{i}" for i in range(1, df.shape[1])]
    else:
        df = df_raw.iloc[1:].reset_index(drop=True).copy(); header = dis_make_unique(first_row.tolist()); df.columns = header; df.rename(columns={df.columns[0]: "Time"}, inplace=True)
    df.columns = dis_make_unique(df.columns); df["Time"] = to_numeric(df["Time"])
    unit_cols = [c for c in df.columns if c != "Time"]
    for c in unit_cols:
        df[c] = to_numeric(df[c])
    df = df.dropna(subset=["Time"]).copy(); df = df.loc[df[unit_cols].notna().any(axis=1)].copy(); df = df.sort_values("Time").reset_index(drop=True)
    if len(df) == 0:
        raise ValueError("No valid dissolution rows found after cleaning.")
    return df


def dis_get_unit_cols(df):
    return [c for c in df.columns if c != "Time"]


def dis_profile_summary(df):
    unit_cols = dis_get_unit_cols(df); out = pd.DataFrame({"Time": df["Time"].to_numpy()}); values = df[unit_cols].to_numpy(dtype=float)
    out["n_units"] = np.sum(np.isfinite(values), axis=1); out["mean"] = np.nanmean(values, axis=1); out["sd"] = np.nanstd(values, axis=1, ddof=1)
    out.loc[out["n_units"] <= 1, "sd"] = np.nan; out["cv_pct"] = 100 * out["sd"] / out["mean"]; out.loc[out["mean"] == 0, "cv_pct"] = np.nan
    return out


def dis_merge_profiles(ref_summary, test_summary):
    merged = ref_summary.merge(test_summary, on="Time", how="inner", suffixes=("_ref", "_test")).sort_values("Time").reset_index(drop=True)
    if len(merged) == 0:
        raise ValueError("Reference and Test have no common timepoints.")
    return merged


def dis_select_points(merged, include_zero=True, cutoff_mode="all", threshold=85.0):
    use = merged.copy()
    if not include_zero:
        use = use.loc[use["Time"] != 0].copy()
    if len(use) == 0:
        raise ValueError("No timepoints left after filtering.")
    first_both_ge_idx = None
    for i in range(len(use)):
        if use.loc[i, "mean_ref"] >= threshold and use.loc[i, "mean_test"] >= threshold:
            first_both_ge_idx = i; break
    if cutoff_mode == "apply_85" and first_both_ge_idx is not None:
        use = use.iloc[:first_both_ge_idx + 1].copy()
    if len(use) < 3:
        raise ValueError("At least 3 selected timepoints are needed to calculate f2.")
    return use.reset_index(drop=True), first_both_ge_idx


def dis_calc_f2(ref_means, test_means):
    ref_means = np.asarray(ref_means, dtype=float); test_means = np.asarray(test_means, dtype=float)
    if len(ref_means) < 1:
        return np.nan
    return 50 * np.log10(100 / np.sqrt(1 + np.mean((ref_means - test_means) ** 2)))


def dis_get_selected_matrix(df, selected_times):
    sub = df[df["Time"].isin(selected_times)].copy().sort_values("Time").reset_index(drop=True); times_sorted = np.sort(np.asarray(selected_times, dtype=float))
    if len(sub) != len(times_sorted) or not np.allclose(sub["Time"].to_numpy(dtype=float), times_sorted):
        raise ValueError("Selected timepoints could not be aligned back to the original profile table.")
    return sub[dis_get_unit_cols(df)].to_numpy(dtype=float), dis_get_unit_cols(df)


def dis_fda_checks(ref_df, test_df, selected, threshold=85.0, include_zero=False):
    ref_units = len(dis_get_unit_cols(ref_df)); test_units = len(dis_get_unit_cols(test_df)); same_original_times = np.array_equal(np.sort(ref_df["Time"].to_numpy(dtype=float)), np.sort(test_df["Time"].to_numpy(dtype=float)))
    at_least_12 = (ref_units >= 12) and (test_units >= 12); at_least_3_points = len(selected) >= 3; both_ge = (selected["mean_ref"] >= threshold) & (selected["mean_test"] >= threshold); n_post85_kept = int(both_ge.sum()); one_post85_ok = n_post85_kept <= 1
    selected_nonzero = selected.copy() if include_zero else selected[selected["Time"] > 0].copy()
    early_cv_ref = early_cv_test = later_max_cv_ref = later_max_cv_test = np.nan; cv_ok = True
    if len(selected_nonzero) > 0:
        early_cv_ref = selected_nonzero.iloc[0]["cv_pct_ref"]; early_cv_test = selected_nonzero.iloc[0]["cv_pct_test"]
        later_ref = selected_nonzero.iloc[1:]["cv_pct_ref"].dropna(); later_test = selected_nonzero.iloc[1:]["cv_pct_test"].dropna(); later_max_cv_ref = later_ref.max() if len(later_ref) > 0 else np.nan; later_max_cv_test = later_test.max() if len(later_test) > 0 else np.nan
        for v, lim in [(early_cv_ref, 20), (early_cv_test, 20), (later_max_cv_ref, 10), (later_max_cv_test, 10)]:
            if pd.notna(v) and v > lim:
                cv_ok = False
    fda_tbl = pd.DataFrame([
        {"Criterion": "Same original timepoints in both profiles", "Pass": "Yes" if same_original_times else "No"},
        {"Criterion": "At least 12 units in Reference and Test", "Pass": "Yes" if at_least_12 else "No"},
        {"Criterion": "At least 3 selected timepoints for f2", "Pass": "Yes" if at_least_3_points else "No"},
        {"Criterion": "No more than one selected point after both are ≥ threshold", "Pass": "Yes" if one_post85_ok else "No"},
        {"Criterion": "CV at earlier selected timepoint ≤ 20% and later ≤ 10%", "Pass": "Yes" if cv_ok else "No"},
    ])
    detail_tbl = pd.DataFrame([{ "Reference units": ref_units, "Test units": test_units, "Selected timepoints": len(selected), "Selected points where both ≥ threshold": n_post85_kept, "Earlier CV ref": early_cv_ref, "Earlier CV test": early_cv_test, "Later max CV ref": later_max_cv_ref, "Later max CV test": later_max_cv_test }])
    return fda_tbl, detail_tbl, bool((fda_tbl["Pass"] == "Yes").all())


def dis_bootstrap_f2(ref_mat, test_mat, n_boot=2000, seed=123):
    rng = np.random.default_rng(seed); n_ref = ref_mat.shape[1]; n_test = test_mat.shape[1]; out = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx_ref = rng.integers(0, n_ref, size=n_ref); idx_test = rng.integers(0, n_test, size=n_test)
        out[b] = dis_calc_f2(np.nanmean(ref_mat[:, idx_ref], axis=1), np.nanmean(test_mat[:, idx_test], axis=1))
    return out


def dis_jackknife_f2(ref_mat, test_mat):
    vals = []; n_ref = ref_mat.shape[1]; n_test = test_mat.shape[1]
    for j in range(n_ref):
        keep = [i for i in range(n_ref) if i != j]
        if keep:
            vals.append(dis_calc_f2(np.nanmean(ref_mat[:, keep], axis=1), np.nanmean(test_mat, axis=1)))
    for j in range(n_test):
        keep = [i for i in range(n_test) if i != j]
        if keep:
            vals.append(dis_calc_f2(np.nanmean(ref_mat, axis=1), np.nanmean(test_mat[:, keep], axis=1)))
    return np.asarray(vals, dtype=float)


def dis_bca_interval(theta_hat, boot_vals, jack_vals, conf=0.90):
    boot_vals = np.asarray(boot_vals, dtype=float); boot_vals = boot_vals[np.isfinite(boot_vals)]; jack_vals = np.asarray(jack_vals, dtype=float); jack_vals = jack_vals[np.isfinite(jack_vals)]
    if len(boot_vals) < 10:
        return np.nan, np.nan, np.nan, np.nan
    alpha = 1 - conf; prop_less = np.mean(boot_vals < theta_hat); eps = 1 / (2 * len(boot_vals)); prop_less = np.clip(prop_less, eps, 1 - eps); z0 = norm.ppf(prop_less)
    if len(jack_vals) < 3:
        a = 0.0
    else:
        jack_mean = np.mean(jack_vals); num = np.sum((jack_mean - jack_vals) ** 3); den = 6 * (np.sum((jack_mean - jack_vals) ** 2) ** 1.5); a = num / den if den > 0 else 0.0
    z_low = norm.ppf(alpha / 2); z_high = norm.ppf(1 - alpha / 2)
    adj_low = norm.cdf(z0 + (z0 + z_low) / (1 - a * (z0 + z_low))); adj_high = norm.cdf(z0 + (z0 + z_high) / (1 - a * (z0 + z_high)))
    return np.quantile(boot_vals, adj_low), np.quantile(boot_vals, adj_high), z0, a


def dis_percentile_interval(boot_vals, conf=0.90):
    alpha = 1 - conf
    return np.quantile(boot_vals, alpha / 2), np.quantile(boot_vals, 1 - alpha / 2)


def dis_plot_profiles(ref_df, test_df, ref_summary, test_summary, selected, show_units=True, title="Dissolution Profiles", ylabel="% Dissolved"):
    cfg = safe_get_plot_cfg("Dissolution comparison")
    fig, ax = plt.subplots(figsize=(cfg["fig_w"], cfg["fig_h"]))
    c_ref = cfg["primary_color"]; c_test = cfg["secondary_color"]; sel_ms = max(40, int(cfg["marker_size"] * 1.8))
    if show_units:
        for c in dis_get_unit_cols(ref_df):
            ax.plot(ref_df["Time"], ref_df[c], color=c_ref, alpha=0.15, linewidth=max(0.8, cfg["aux_line_width"]))
        for c in dis_get_unit_cols(test_df):
            ax.plot(test_df["Time"], test_df[c], color=c_test, alpha=0.15, linewidth=max(0.8, cfg["aux_line_width"]))
    ax.plot(ref_summary["Time"], ref_summary["mean"], marker="o", color=c_ref, linewidth=cfg["line_width"], markersize=max(4, int(cfg["marker_size"] ** 0.5)), linestyle=cfg["line_style"], label="Reference Mean")
    ax.plot(test_summary["Time"], test_summary["mean"], marker="o", color=c_test, linewidth=cfg["line_width"], markersize=max(4, int(cfg["marker_size"] ** 0.5)), linestyle=cfg["line_style"], label="Test Mean")
    ax.scatter(selected["Time"], selected["mean_ref"], marker="s", edgecolor="black", facecolor="none", s=sel_ms, linewidth=1.2, label="Selected Ref Points", zorder=4)
    ax.scatter(selected["Time"], selected["mean_test"], marker="s", edgecolor="black", facecolor="none", s=sel_ms, linewidth=1.2, label="Selected Test Points", zorder=4)
    apply_ax_style(ax, title, "Time", ylabel, legend=True, plot_key="Dissolution comparison")
    return fig


def dis_plot_bootstrap_f2_distribution(boot_vals, observed_f2, ci_low=None, ci_high=None, ci_label="90% CI", title="Distribution of f2 Similarity Factor", x_min=50, x_max=100):
    cfg = safe_get_plot_cfg("Dissolution comparison")
    boot_vals = np.asarray(boot_vals, dtype=float); boot_vals = boot_vals[np.isfinite(boot_vals)]
    if len(boot_vals) < 5:
        return None
    fig, ax = plt.subplots(figsize=(cfg["fig_w"], cfg["fig_h"]))
    mean_boot = float(np.mean(boot_vals))
    sd_boot = np.std(boot_vals, ddof=1)
    if sd_boot > 0:
        kde = gaussian_kde(boot_vals); x_lo = min(x_min, np.min(boot_vals) - 2 * sd_boot, observed_f2 - 5, mean_boot - 5); x_hi = max(x_max, np.max(boot_vals) + 2 * sd_boot, observed_f2 + 5, mean_boot + 5)
        if ci_low is not None: x_lo = min(x_lo, ci_low - 3)
        if ci_high is not None: x_hi = max(x_hi, ci_high + 3)
        x_grid = np.linspace(x_lo, x_hi, 600); y_grid = kde(x_grid); y_grid = np.asarray(y_grid, dtype=float); y_grid[0] = 0.0; y_grid[-1] = 0.0
        ax.fill_between(x_grid, y_grid, color=cfg["band_color"], alpha=0.15)
        ax.plot(x_grid, y_grid, color=cfg["primary_color"], linewidth=cfg["line_width"], linestyle=cfg["line_style"])
    else:
        ax.axvline(mean_boot, color=cfg["primary_color"], linewidth=cfg["line_width"])
    ax.axvline(observed_f2, color=cfg["primary_color"], linestyle=cfg["aux_line_style"], linewidth=cfg["aux_line_width"], alpha=0.65)
    ax.axvline(mean_boot, color=cfg["primary_color"], linewidth=max(cfg["line_width"], cfg["aux_line_width"] + 0.6))
    if ci_low is not None: ax.axvline(ci_low, color=cfg["tertiary_color"], linestyle=cfg["aux_line_style"], linewidth=cfg["line_width"])
    if ci_high is not None: ax.axvline(ci_high, color=cfg["tertiary_color"], linestyle=cfg["aux_line_style"], linewidth=cfg["line_width"])
    apply_ax_style(ax, title, "f2 values", "Density", legend=False, plot_key="Dissolution comparison"); ax.set_xlim(x_min, x_max)
    y_top = float(ax.get_ylim()[1]) if ax.get_ylim()[1] > 0 else 1.0
    text_kw = dict(rotation=90, va="top", fontsize=9, clip_on=False)
    ax.text(mean_boot, y_top * 0.98, f"Mean = {mean_boot:.2f}", ha="right", color=cfg["primary_color"], bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor=cfg["primary_color"], alpha=0.65), **text_kw)
    if ci_low is not None:
        ax.text(ci_low, y_top * 0.96, f"Lower CI = {ci_low:.2f}", ha="right", color=cfg["tertiary_color"], bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor=cfg["tertiary_color"], alpha=0.65), **text_kw)
    if ci_high is not None:
        ax.text(ci_high, y_top * 0.94, f"Upper CI = {ci_high:.2f}", ha="left", color=cfg["tertiary_color"], bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor=cfg["tertiary_color"], alpha=0.65), **text_kw)
    return fig

__all__ = [name for name in globals() if not name.startswith('_')]
