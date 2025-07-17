# Visuals/Final_Report.py

import sys, os

# 1) Make project root importable
this_file  = os.path.abspath(__file__)
visuals_dir= os.path.dirname(this_file)
root_dir   = os.path.dirname(visuals_dir)
sys.path.insert(0, root_dir)

import yaml
import pandas as pd

# 2) Load your single Gemini key
creds_path = os.path.join(root_dir, "D:\\Agentic_AI_Compiled\\Agentic_AI_DS_Team\\Visuals\\credentials.yml")
creds      = yaml.safe_load(open(creds_path))
gemini_key = creds.get("gemini_api_key") or creds.get("google_adk") or creds.get("google_genai")
if not gemini_key:
    raise KeyError("Please set 'gemini_api_key' in credentials.yml")

os.environ["GOOGLE_API_KEY"]     = gemini_key
os.environ["GOOGLE_ADK_API_KEY"] = gemini_key

# 3) Import the final agent
from agents_of_ds.agents.Final_Agent import generate_full_report

# 4) Load your dataset
data_path = os.path.join(root_dir, "data", "relevant.csv")
df = pd.read_csv(data_path)

# 5) Run the pipeline
print("🔄 Generating full report…")
report_html = generate_full_report(df, target="Salary")

# 6a) Save HTML
out_path = os.path.join(root_dir, "full_report.html")
with open(out_path, "w", encoding="utf-8") as f:
    f.write(report_html)
print(f"✅ Full report written to {out_path}")

# 6b) (Optional) display inline in Jupyter
try:
    from IPython.display import HTML, display
    display(HTML(report_html))
except ImportError:
    pass

