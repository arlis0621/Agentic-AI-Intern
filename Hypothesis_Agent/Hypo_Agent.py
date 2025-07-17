# Hypo_Agent.py

import os
from typing import Any, Dict, Optional, Sequence

import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# LangChain imports (for BaseMessage, AIMessage, LLM)
# ─────────────────────────────────────────────────────────────────────────────
from langchain.schema import BaseMessage, AIMessage
from langchain.llms.base import LLM

# ─────────────────────────────────────────────────────────────────────────────
# LangGraph imports (mid-2025, still under "langgraph")
# ─────────────────────────────────────────────────────────────────────────────
from langgraph.prebuilt import create_react_agent, ToolNode
from langgraph.prebuilt.chat_agent_executor import AgentState
from langgraph.graph import START, END, StateGraph
from langgraph.types import Checkpointer

# ─────────────────────────────────────────────────────────────────────────────
# Google Generative AI (Gemini) imports (mid-2025 "google-generativeai" package)
# ─────────────────────────────────────────────────────────────────────────────
import google.generativeai as genai

# ─────────────────────────────────────────────────────────────────────────────
# Your custom tools in tools.py (including human_input_text)
# ─────────────────────────────────────────────────────────────────────────────
from tools import (
    list_columns,
    compute_empirical_distribution,
    fit_standard_distribution,
    select_closest_distribution,
    query_parameters_to_test,
    collect_population_values,
    estimate_sample_parameters,
    decide_test_statistic,
    compute_test_statistic,
    interpret_results,
    human_input_text,  # <-- new import
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Wrap Gemini via google.generativeai into a LangChain‐compatible LLM
# ─────────────────────────────────────────────────────────────────────────────
class ADKModel(LLM):
    """
    A LangChain LLM wrapper around Google’s Generative AI (Gemini 2.0)
    using the `google.generativeai` package.
    """

    def __init__(self, api_key: str, model_name: Optional[str] = None):
        genai.api_key = api_key
        self._model_name = model_name if model_name is not None else "gemini-2.0-flash"

    def _call(self, prompt: str, stop: Optional[Sequence[str]] = None) -> str:
        response = genai.ChatCompletion.create(
            model=self._model_name,
            prompt=prompt,
            stop_sequences=stop or []
        )
        return response.choices[0].message.content

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {"model_name": self._model_name}

    @property
    def _llm_type(self) -> str:
        return "gemini-2.0-flash"


# ─────────────────────────────────────────────────────────────────────────────
# 2. Agent wrapper
# ─────────────────────────────────────────────────────────────────────────────
class HypothesisAgent:
    def __init__(
        self,
        model: LLM,
        create_react_agent_kwargs: Optional[Dict] = None,
        invoke_react_agent_kwargs: Optional[Dict] = None,
        checkpointer: Optional[Checkpointer] = None,
    ):
        self._params = {
            "model": model,
            "create_react_agent_kwargs": create_react_agent_kwargs or {},
            "invoke_react_agent_kwargs": invoke_react_agent_kwargs or {},
            "checkpointer": checkpointer,
        }
        self._compiled_graph = self._build_graph()
        self.response: Dict[str, Any] = {}

    def _build_graph(self):
        class GraphState(AgentState):
            internal_messages: Sequence[BaseMessage]
            data_raw: dict
            column: str
            samples: list
            empirical_results: dict
            fit_results: list
            chosen_family: list
            chosen_params: list
            to_test: list
            pop_params_input: dict
            pop_params: dict
            sample_params: dict
            test_decision: dict
            test_output: dict
            interpretation: str

        # ── Node: select_column ──────────────────────────────────────────────────
        def select_column(state: Dict[str, Any]) -> Dict[str, Any]:
            cols = list_columns(state["data_raw"])
            prompt = f"Step 1: Available columns are {cols}. Please choose one: "
            choice = human_input_text(prompt)
            return {"column": choice}

        # ── Node: get_samples ───────────────────────────────────────────────────
        def get_samples(state: Dict[str, Any]) -> Dict[str, Any]:
            df = pd.DataFrame(state["data_raw"])
            col = state["column"]
            if isinstance(col, list):
                col = col[0]
            samples = df[col].dropna().tolist()
            return {"samples": samples}

        # ── Node: empirical_dist ─────────────────────────────────────────────────
        def empirical_dist(state: Dict[str, Any]) -> Dict[str, Any]:
            out = compute_empirical_distribution(state["samples"])
            return {"empirical_results": out}

        # ── Node: fit_distributions ───────────────────────────────────────────────
        def fit_distributions(state: Dict[str, Any]) -> Dict[str, Any]:
            results = fit_standard_distribution(state["samples"])
            return {"fit_results": results}

        # ── Node: choose_distribution ─────────────────────────────────────────────
        def choose_distribution(state: Dict[str, Any]) -> Dict[str, Any]:
            fit_results = state["fit_results"]
            top3 = fit_results[:3]
            lines = []
            for i, r in enumerate(top3, start=1):
                fam = r["family"].capitalize()
                params = r["fitted_params"]
                ks_val = r["gof_stat"]
                lines.append(f"{i}) {fam}  params={params}  KS={ks_val:.4f}")
            prompt = (
                "Step 2: Here are the top‐3 fits:\n  "
                + "\n  ".join(lines)
                + "\nType the exact family name to accept (e.g. 'normal'): "
            )
            choice = human_input_text(prompt).strip().lower()
            all_families = [r["family"] for r in fit_results]
            if choice not in all_families:
                choice = all_families[0]
            idx = all_families.index(choice)
            return {
                "chosen_family": [choice],
                "chosen_params": [fit_results[idx]["fitted_params"]],
            }

        # ── Node: ask_params_to_test ──────────────────────────────────────────────
        def ask_params_to_test(state: Dict[str, Any]) -> Dict[str, Any]:
            dist = state["chosen_family"][0]
            candidates = query_parameters_to_test(dist)
            prompt = (
                f"Step 3: For {dist.capitalize()}, available params = {candidates}.\n"
                "Type the parameter(s) (comma‐separated) you want to test: "
            )
            ans = human_input_text(prompt)
            chosen = [p.strip() for p in ans.split(",") if p.strip() in candidates]
            return {"to_test": chosen}

        # ── Node: collect_pop_vals ─────────────────────────────────────────────────
        def collect_pop_vals(state: Dict[str, Any]) -> Dict[str, Any]:
            to_test = state["to_test"]
            pop_dict: Dict[str, Optional[float]] = {}
            for param in to_test:
                prompt = f"Step 4: Please enter the population value for '{param}', or type 'unknown': "
                val = human_input_text(prompt).strip().lower()
                if val in ("unknown", ""):
                    pop_dict[param] = None
                else:
                    try:
                        pop_dict[param] = float(val)
                    except ValueError:
                        pop_dict[param] = None
            return {"pop_params": pop_dict}

        # ── Node: estimate_samples ─────────────────────────────────────────────────
        def estimate_samples(state: Dict[str, Any]) -> Dict[str, Any]:
            samp = estimate_sample_parameters(state["samples"], state["to_test"])
            return {"sample_params": samp}

        # ── Node: decide_stat ───────────────────────────────────────────────────────
        def decide_stat(state: Dict[str, Any]) -> Dict[str, Any]:
            decision = decide_test_statistic(
                state["chosen_family"][0], state["pop_params"]
            )
            return {"test_decision": decision}

        # ── Node: compute_test ──────────────────────────────────────────────────────
        def compute_test(state: Dict[str, Any]) -> Dict[str, Any]:
            stat_name = state["test_decision"]["statistic"]
            if stat_name is None:
                return {
                    "test_output": {
                        "statistic": None,
                        "T_obs": None,
                        "p_value": None,
                        "df": None,
                        "dist": None,
                    }
                }
            out = compute_test_statistic(
                state["samples"],
                state["pop_params"],
                stat_name,
            )
            out["statistic"] = stat_name
            return {"test_output": out}

        # ── Node: interpret ─────────────────────────────────────────────────────────
        def interpret(state: Dict[str, Any]) -> Dict[str, Any]:
            test_out = state["test_output"]
            if test_out.get("statistic") is None:
                return {
                    "interpretation": (
                        "No valid test statistic could be constructed "
                        "given the provided/missing population parameters."
                    )
                }
            text = interpret_results(test_out)
            return {"interpretation": text}

        # ────────────────────────────────────────────────────────────────────────────
        # Build and compile the graph
        # ────────────────────────────────────────────────────────────────────────────
        graph = StateGraph(GraphState)

        graph.add_node("select_column", select_column)
        graph.add_node("get_samples", get_samples)
        graph.add_node("empirical_dist", empirical_dist)
        graph.add_node("fit_distributions", fit_distributions)
        graph.add_node("choose_distribution", choose_distribution)
        graph.add_node("ask_params_to_test", ask_params_to_test)
        graph.add_node("collect_pop_vals", collect_pop_vals)
        graph.add_node("estimate_samples", estimate_samples)
        graph.add_node("decide_stat", decide_stat)
        graph.add_node("compute_test", compute_test)
        graph.add_node("interpret", interpret)

        steps = [
            "select_column",
            "get_samples",
            "empirical_dist",
            "fit_distributions",
            "choose_distribution",
            "ask_params_to_test",
            "collect_pop_vals",
            "estimate_samples",
            "decide_stat",
            "compute_test",
            "interpret",
        ]

        graph.add_edge(START, steps[0])
        for prev_node, next_node in zip(steps, steps[1:]):
            graph.add_edge(prev_node, next_node)
        graph.add_edge(steps[-1], END)

        return graph.compile(
            name="hypothesis_testing_agent",
            checkpointer=self._params["checkpointer"],
        )

    def invoke(self, data_raw: pd.DataFrame) -> Dict[str, Any]:
        payload = {"data_raw": data_raw.to_dict(orient="list")}
        self.response = self._compiled_graph.invoke(
            payload, self._params["invoke_react_agent_kwargs"]
        )
        return self.response

    async def ainvoke(self, data_raw: pd.DataFrame) -> Dict[str, Any]:
        payload = {"data_raw": data_raw.to_dict(orient="list")}
        self.response = await self._compiled_graph.ainvoke(
            payload, self._params["invoke_react_agent_kwargs"]
        )
        return self.response
