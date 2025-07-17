# Run_Agent.py

import os
import yaml
import pandas as pd

from Hypo_Agent import HypothesisAgent, ADKModel

def main():
    # ────────────────────────────────────────────────────────────────────────────
    # 1) Load API key from credentials.yml (either as a mapping or bare string)
    # ────────────────────────────────────────────────────────────────────────────
    raw = yaml.safe_load(open("credentials.yml"))
    if isinstance(raw, dict):
        # Expecting something like { "google_adk": "YOUR_KEY" }
        api_key = raw.get("google_adk") or raw.get("api_key") or next(iter(raw.values()))
        if not isinstance(api_key, str):
            raise ValueError("credentials.yml must contain a string API key.")
    elif isinstance(raw, str):
        api_key = raw
    else:
        raise ValueError("credentials.yml must be either a dict with your key or just the key string.")

    os.environ["GOOGLE_ADK_API_KEY"] = api_key

    # ────────────────────────────────────────────────────────────────────────────
    # 2) Instantiate the Gemini‐powered LLM and the LangGraph agent
    #    (HypothesisAgent and ADKModel come from Hypo_Agent.py) :contentReference[oaicite:1]{index=1}
    # ────────────────────────────────────────────────────────────────────────────
    model = ADKModel(api_key=api_key)
    agent = HypothesisAgent(model=model)

    # ────────────────────────────────────────────────────────────────────────────
    # 3) Load the dataset (expects a sample_data.csv in the same folder) 
    # ────────────────────────────────────────────────────────────────────────────
    try:
        df = pd.read_csv("sample_data.csv")
    except FileNotFoundError:
        raise FileNotFoundError("Could not find 'sample_data.csv' in the working directory.")

    # ────────────────────────────────────────────────────────────────────────────
    # 4) Invoke the agent. At each decision node, Hypo_Agent will call 
    #    human_input_text(...) (from tools.py) to prompt the user. :contentReference[oaicite:3]{index=3}
    # ────────────────────────────────────────────────────────────────────────────
    response = agent.invoke(df)

    # ────────────────────────────────────────────────────────────────────────────
    # 5) Print out the key fields from the agent’s final state
    #    These keys correspond to the state variables set by each node:
    #      - column            (the column you chose)
    #      - empirical_results (histogram + sample_stats)
    #      - fit_results       (list of all fitted families + KS scores)
    #      - chosen_family     (single‐element list of the accepted family)
    #      - chosen_params     (single‐element list of that family’s params)
    #      - to_test           (list of parameter names you elected to test)
    #      - pop_params        (dict of population values or None)
    #      - sample_params     (dict of sample estimates)
    #      - test_decision     (dict with "statistic" and "requirements")
    #      - test_output       (dict with "T_obs", "p_value", "df", "dist", "statistic")
    #      - interpretation    (plain‐English conclusion)
    # ────────────────────────────────────────────────────────────────────────────
    print("\n\n====== AGENT OUTPUT ======")
    print("Column chosen:           ", response.get("column"))
    print("Empirical summary:       ", response.get("empirical_results"))
    print("Fitted distributions:    ", response.get("fit_results"))
    print("Chosen family:           ", response.get("chosen_family"), response.get("chosen_params"))
    print("Parameters to test:      ", response.get("to_test"))
    print("Population parameters:   ", response.get("pop_params"))
    print("Sample estimates:        ", response.get("sample_params"))
    print("Test decision:           ", response.get("test_decision"))
    print("Test statistic output:   ", response.get("test_output"))
    print("Interpretation:          ", response.get("interpretation"))
    print("============================\n")


if __name__ == "__main__":
    main()
