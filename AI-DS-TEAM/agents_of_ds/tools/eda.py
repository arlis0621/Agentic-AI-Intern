# eda.py

import os
import tempfile
import base64
import json
from io import BytesIO
from typing import Dict, Tuple, Union
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt

from agents_of_ds.tools.dataframe import get_dataframe_summary


def explain_data(
    data_raw: dict,
    n_sample: int = 30,
    skip_stats: bool = False,
) -> str:
    """
    Provides an extensive, narrative summary of a DataFrame including shape, dtypes,
    missing-value percentages, sample rows, and optional descriptive statistics.
    """
    print("    * Tool: explain_data")
    df = pd.DataFrame(data_raw)
    return get_dataframe_summary(df, n_sample=n_sample, skip_stats=skip_stats)


def describe_dataset(
    data_raw: dict,
) -> Tuple[str, Dict]:
    """
    Computes pandas .describe() stats, returning a brief text plus the full stats dict.
    """
    print("    * Tool: describe_dataset")
    df = pd.DataFrame(data_raw)
    description_df = df.describe(include="all")
    content = "Summary statistics computed using pandas describe()."
    artifact = { "describe_df": description_df.to_dict() }
    return content, artifact


def visualize_missing(
    data_raw: dict,
    n_sample: int = None
) -> Tuple[str, Dict]:
    """
    Generates missing-value matrix, bar, and heatmap plots (base64-encoded PNGs).
    """
    print("    * Tool: visualize_missing")
    try:
        import missingno as msno
    except ImportError:
        raise ImportError("Install missingno: pip install missingno")

    df = pd.DataFrame(data_raw)
    if n_sample is not None:
        df = df.sample(n=n_sample, random_state=42)

    def encode_plot(plot_func):
        plt.figure(figsize=(8, 6))
        plot_func(df)
        plt.tight_layout()
        buf = BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        buf.seek(0)
        return base64.b64encode(buf.getvalue()).decode()

    encoded_plots = {
        "matrix_plot":  encode_plot(msno.matrix),
        "bar_plot":     encode_plot(msno.bar),
        "heatmap_plot": encode_plot(msno.heatmap),
    }

    content = "Missing data visualizations generated."
    return content, encoded_plots


def generate_correlation_funnel(
    data_raw: dict,
    target: str,
    target_bin_index: Union[int, str] = -1,
    corr_method: str = "pearson",
    n_bins: int = 4,
    thresh_infreq: float = 0.01,
    name_infreq: str = "-OTHER",
) -> Tuple[str, Dict]:
    """
    Builds a correlation funnel vs. a target column; returns content plus artifacts:
      - correlation_data: dict of correlations
      - plot_image: base64‐encoded PNG
      - plotly_figure: Plotly JSON
    """
    print("    * Tool: generate_correlation_funnel")
    try:
        import pytimetk as tk
    except ImportError:
        raise ImportError("Install pytimetk: pip install pytimetk")

    df = pd.DataFrame(data_raw)
    df_binarized = df.binarize(
        n_bins=n_bins,
        thresh_infreq=thresh_infreq,
        name_infreq=name_infreq,
        one_hot=True,
    )

    # Resolve the full target column name
    matching = [c for c in df_binarized.columns if c.startswith(f"{target}__")]
    if matching:
        if isinstance(target_bin_index, str):
            full_target = f"{target}__{target_bin_index}"
            if full_target not in matching:
                full_target = matching[-1]
        else:
            idx = target_bin_index if target_bin_index >= 0 else len(matching) + target_bin_index
            full_target = matching[idx] if 0 <= idx < len(matching) else matching[-1]
    else:
        full_target = target

    df_corr = df_binarized.correlate(target=full_target, method=corr_method)

    # Static PNG via plotnine or pandas plotting
    try:
        fig = df_corr.plot_correlation_funnel(engine="plotnine", height=600)
        buf = BytesIO()
        fig.save(buf, format="png")
        plt.close()
        buf.seek(0)
        img_enc = base64.b64encode(buf.getvalue()).decode()
    except Exception as e:
        img_enc = {"error": str(e)}

    # Interactive Plotly JSON
    try:
        import plotly.io as pio
        fig2 = df_corr.plot_correlation_funnel(engine="plotly", base_size=14)
        fig_dict = json.loads(pio.to_json(fig2))
    except Exception as e:
        fig_dict = {"error": str(e)}

    content = (
        f"Correlation funnel computed with method='{corr_method}' "
        f"for target='{full_target}'."
    )
    artifact = {
        "correlation_data": df_corr.to_dict(orient="list"),
        "plot_image": img_enc,
        "plotly_figure": fig_dict,
    }
    return content, artifact


def generate_sweetviz_report(
    data_raw: dict,
    target: str = None,
    report_name: str = "sweetviz_report.html",
    report_directory: str = None,
    open_browser: bool = False,
) -> Tuple[str, Dict]:
    """
    Generates a Sweetviz HTML report; returns its file path and (optional) HTML content.
    """
    print("    * Tool: generate_sweetviz_report")

    # -- Patch numpy so Sweetviz can reference VisibleDeprecationWarning
    import numpy as np
    if not hasattr(np, "VisibleDeprecationWarning"):
        np.VisibleDeprecationWarning = DeprecationWarning

    try:
        import sweetviz as sv
    except ImportError:
        raise ImportError("Install sweetviz: pip install sweetviz")

    # Ensure the usual imports are available

    df = pd.DataFrame(data_raw)
    report_directory = report_directory or tempfile.mkdtemp()
    os.makedirs(report_directory, exist_ok=True)

    report = sv.analyze(df, target_feat=target)
    full_path = os.path.join(report_directory, report_name)
    report.show_html(filepath=full_path, open_browser=open_browser)

    try:
        with open(full_path, "r", encoding="utf-8") as f:
            html_content = f.read()
    except Exception:
        html_content = None

    content = f"Sweetviz report saved to '{full_path}'."
    artifact = {"report_file": full_path, "report_html": html_content}
    return content, artifact



def generate_dtale_report(
    data_raw: dict,
    host: str = "localhost",
    port: int = 40000,
    open_browser: bool = False,
) -> Tuple[str, Dict]:
    """
    Launches a D-Tale server and returns its URL in the artifact.
    """
    print("    * Tool: generate_dtale_report")
    try:
        import dtale
    except ImportError:
        raise ImportError("Install dtale: pip install dtale")

    df = pd.DataFrame(data_raw)
    dt = dtale.show(df, host=host, port=port, open_browser=open_browser)

    content = f"D-Tale report available at: {dt.main_url()}"
    artifact = {"dtale_url": dt.main_url()}
    return content, artifact
