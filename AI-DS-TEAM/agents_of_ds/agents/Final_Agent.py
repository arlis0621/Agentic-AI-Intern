import os
import re
import pandas as pd

from agents_of_ds.agents.Data_Clean import ADKModelWrapper, DataCleaningAgent
from agents_of_ds.agents.EDA_Agent import EDAToolsAgent
from agents_of_ds.agents.Feature_engineer import FeatureEngineeringAgent
from agents_of_ds.tools.eda import (
    generate_sweetviz_report,
    visualize_missing,
    generate_correlation_funnel,
)


def _unwrap_sweetviz_path(result):
    if isinstance(result, (list, tuple)) and isinstance(result[0], str):
        return result[0]
    if isinstance(result, str):
        m = re.search(r"'([^']+\.html)'", result)
        if m:
            return m.group(1)
        return result
    raise ValueError(f"Cannot extract path: {result!r}")


def _fig_to_base64(fig):
    from io import BytesIO
    import base64

    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    data = base64.b64encode(buf.read()).decode("utf-8")
    return f"<img src='data:image/png;base64,{data}' style='width:100%;max-width:800px;'/>"


def generate_full_report(df: pd.DataFrame, target: str) -> str:
    gemini_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GOOGLE_ADK_API_KEY")
    if not gemini_key:
        raise KeyError("Set GOOGLE_API_KEY (or GOOGLE_ADK_API_KEY) to your Gemini key")

    # 1) Cleaning
    llm = ADKModelWrapper(api_key=gemini_key, model_name="gemini-2.0-flash")
    cleaner = DataCleaningAgent(model=llm, log=False)
    cleaner.invoke_agent(df, user_instructions="Clean data for modeling")
    cleaning_error = cleaner.response.get("data_cleaner_error")
    cleaned_df = cleaner.get_data_cleaned()
    # ── ⚡️ QUICK IMPUTE PASS TO KILL ANY LEFTOVER NULLS ──
    import pandas as pd
    for col in cleaned_df.columns:
        if cleaned_df[col].isnull().any():
            if pd.api.types.is_numeric_dtype(cleaned_df[col]):
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
            else:
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0])
    if cleaned_df is None or cleaned_df.empty:
        cleaning_error = cleaning_error or "Error during cleaning."
        cleaned_df = df.copy()

    # 2) EDA narrative
    eda = EDAToolsAgent()
    eda.invoke_agent(
        user_instructions=f"EDA narrative focused on '{target}'",
        data_raw=cleaned_df.to_dict(),
        target=target,
    )
    narrative = eda.get_ai_message()

    # 3) Feature engineering
    fe = FeatureEngineeringAgent(model=llm, log=False)
    fe.invoke_agent(
        data_raw=cleaned_df,
        user_instructions=f"Engineer features for '{target}'",
        target_variable=target,
    )
    fe_error    = fe.response.get("feature_engineer_error")
    fe_steps    = fe.get_recommended_feature_engineering_steps(markdown=True)
    fe_code     = fe.get_feature_engineer_function(markdown=True)
    engineered_df = fe.get_data_engineered()
    if engineered_df is None or engineered_df.empty:
        fe_error = fe_error or "No new features created."
        engineered_df = cleaned_df.copy()

    # 4) Sweetviz report
    sv_res  = generate_sweetviz_report(cleaned_df.to_dict())
    sv_path = _unwrap_sweetviz_path(sv_res)
    with open(sv_path, "r", encoding="utf-8") as f:
        sv_html = f.read()

    # 5) Missing-value viz
    mv_fig = visualize_missing(cleaned_df.to_dict())
    mv_img = _fig_to_base64(mv_fig)

    # 6) Correlation funnel
    cf_fig = generate_correlation_funnel(cleaned_df.to_dict(), target)
    cf_img = _fig_to_base64(cf_fig)

    # Assemble HTML
    html = f"""<!DOCTYPE html>
<html><head><meta charset='utf-8'><title>Full Report</title>
<style>
  body {{ font-family: sans-serif; margin:2rem; }}
  h1 {{ border-bottom:2px solid #444; margin-top:2rem; }}
  .error {{ color: red; }}
  .table-striped tr:nth-child(even) {{background:#f7f7f7}}
</style>
</head><body>

<h1>1. Data Cleaning</h1>
{"<p class='error'>" + cleaning_error + "</p>" if cleaning_error else ""}
{cleaner.get_recommended_cleaning_steps(markdown=True)}
<h2>Cleaned Data Sample</h2>
{cleaned_df.head().to_html(classes='table table-striped', index=False)}

<h1>2. EDA Narrative</h1>
<div style="white-space:pre-wrap">{narrative}</div>

<h1>3. Feature Engineering</h1>
{"<p class='error'>" + fe_error + "</p>" if fe_error else ""}
{fe_steps}
<h2>Generated Function</h2>
{fe_code}
<h2>Engineered Data Sample</h2>
{engineered_df.head().to_html(classes='table table-striped', index=False)}

<h1>4. Sweetviz Detailed Report</h1>
<div style="border:1px solid #ccc; height:700px; overflow:auto;">
  {sv_html}
</div>

<h1>5. Missing-Value Visualization</h1>
{mv_img}

<h1>6. Correlation Funnel</h1>
{cf_img}

</body></html>"""
    return html
