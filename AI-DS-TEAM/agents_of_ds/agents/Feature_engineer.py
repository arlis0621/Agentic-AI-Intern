"""
Feature Engineering Agent using LangGraph + Google GenAI (Gemini 2.0).
"""
import os
import re
import ast
import textwrap
import operator
from typing import TypedDict, Annotated, Any, Sequence, Optional

import pandas as pd

from langchain.prompts import PromptTemplate
from langchain.llms.base import LLM
from langchain_core.messages import BaseMessage
from langgraph.types import Checkpointer
from langgraph.checkpoint.memory import MemorySaver

from agents_of_ds.tempis.templates_of_agent import (
    node_func_execute_agent_code_on_data,
    node_func_report_agent_outputs,
    create_coding_agent_graph,
    BaseAgent,
)
from agents_of_ds.parsers.parsers import PythonOutputParser
from utils.regex import add_comments_to_top, format_agent_name, format_recommended_steps
from agents_of_ds.tools.dataframe import get_dataframe_summary
from google import genai
from pydantic import PrivateAttr

AGENT_NAME = "feature_engineering_agent"
LOG_PATH   = os.path.join(os.getcwd(), "logs/")


class GraphState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    user_instructions: str
    data_raw: dict
    recommended_steps: str
    all_datasets_summary: str
    target_variable: Optional[str]
    feature_engineer_function: str
    feature_engineer_error: str
    data_engineered: dict
    max_retries: int
    retry_count: int


class FeatureEngineeringAgent(BaseAgent):
    def __init__(
        self,
        model: LLM,
        n_samples: int = 30,
        log: bool = False,
        log_path: Optional[str] = None,
        file_name: str = "feature_engineer.py",
        function_name: str = "feature_engineer",
        overwrite: bool = True,
        human_in_the_loop: bool = False,
        bypass_recommended_steps: bool = False,
        bypass_explain_code: bool = False,
        checkpointer: Checkpointer = None,
    ):
        self._params = {
            "model": model,
            "n_samples": n_samples,
            "log": log,
            "log_path": log_path,
            "file_name": file_name,
            "function_name": function_name,
            "overwrite": overwrite,
            "human_in_the_loop": human_in_the_loop,
            "bypass_recommended_steps": bypass_recommended_steps,
            "bypass_explain_code": bypass_explain_code,
            "checkpointer": checkpointer,
        }
        self._compiled_graph = self._make_compiled_graph()
        self.response = None

    def _make_compiled_graph(self):
        if self._params["human_in_the_loop"] and not self._params["checkpointer"]:
            self._params["checkpointer"] = MemorySaver()
        return make_feature_engineering_agent(**self._params)

    def invoke_agent(
        self,
        data_raw: pd.DataFrame,
        user_instructions: str = None,
        target_variable: str = None,
        max_retries: int = 3,
        retry_count: int = 0,
        **kwargs,
    ):
        self.response = self._compiled_graph.invoke(
            {
                "user_instructions": user_instructions or "",
                "data_raw": data_raw.to_dict(),
                "target_variable": target_variable,
                "max_retries": max_retries,
                "retry_count": retry_count,
            },
            **kwargs,
        )

    def get_recommended_feature_engineering_steps(self, markdown: bool = False):
        steps = self.response.get("recommended_steps", "")
        from IPython.display import Markdown as _MD
        return _MD(steps) if markdown else steps

    def get_feature_engineer_function(self, markdown: bool = False):
        code = self.response.get("feature_engineer_function", "")
        from IPython.display import Markdown as _MD
        return _MD(f"```python\n{code}\n```") if markdown else code

    def get_data_engineered(self) -> Optional[pd.DataFrame]:
        data = self.response.get("data_engineered")
        return pd.DataFrame(data) if data is not None else None


def make_feature_engineering_agent(
    model: LLM,
    n_samples: int = 30,
    log: bool = False,
    log_path: Optional[str] = None,
    file_name: str = "feature_engineer.py",
    function_name: str = "feature_engineer",
    overwrite: bool = True,
    human_in_the_loop: bool = False,
    bypass_recommended_steps: bool = False,
    bypass_explain_code: bool = False,
    checkpointer: Checkpointer = None,
):
    llm = model

    def recommend_steps(state: GraphState):
        print(format_agent_name(AGENT_NAME), "* RECOMMEND FEATURE ENGINEERING STEPS")
        df = pd.DataFrame.from_dict(state["data_raw"])
        summary_str = "\n\n".join(get_dataframe_summary([df], n_sample=n_samples))
        prompt = PromptTemplate(
            template="""
You are a Feature Engineering expert. Recommend numbered steps to generate new features for predicting `{target_variable}`.
User instructions: {user_instructions}
Dataset summary: {summary}
""",
            input_variables=["user_instructions", "target_variable", "summary"],
        )
        resp = (prompt | llm).invoke(
            {
                "user_instructions": state["user_instructions"],
                "target_variable": state.get("target_variable", ""),
                "summary": summary_str,
            }
        )
        return {
            "recommended_steps": format_recommended_steps(resp, heading="# FE Steps:"),
            "all_datasets_summary": summary_str,
        }

    def create_code(state: GraphState):
        print(format_agent_name(AGENT_NAME), "* CREATE FEATURE ENGINEERING CODE")
    # 1) Prompt for *only* the body (no wrapper, no imports)
        prompt = PromptTemplate(template="""Generate only the *body* of a function `{function_name}(data_raw)` to implement these steps:
                                {recommended_steps}
                                (no wrapper, no imports)""",
                                input_variables=["recommended_steps", "function_name"],)
        raw = (prompt | llm | PythonOutputParser()).invoke({"recommended_steps": state["recommended_steps"],"function_name": function_name,})
        body_text = raw if isinstance(raw, str) else raw.get("text", "")
        dedented = textwrap.dedent(body_text)

    # ─── FIX #2: Strip out any user-imports ───────────────────────────────────
    # We manage imports ourselves at the top of the wrapper.
        body_lines = []
        for ln in dedented.splitlines():
            stripped = ln.strip()
            if not stripped:
            # skip blank lines entirely (optional)
                continue
        # drop any import/from lines coming from the LLM
            if stripped.startswith(("import ", "from ")):
                continue
            body_lines.append(ln)

    # ─── FIX #3: Build a clean wrapper with controlled imports ────────────────
        wrapper = [f"def {function_name}(data_raw):",
                "    import pandas as pd",
                "    import numpy as np",
                "    from scipy.stats.mstats import winsorize",
                "    from scipy.stats import iqr",
                "    from sklearn.preprocessing import OneHotEncoder, StandardScaler",
                "    df = data_raw.copy()",]
    # Now re-indent EVERY line of the body exactly one level (4 spaces)
        for ln in body_lines:
            wrapper.append("    " + ln.rstrip())
        wrapper.append("    return df")

        code = "\n".join(wrapper)

    # ─── FIX #4: Eliminate any stray “, inplace=True” and rewrite fillna───
        code = code.replace(", inplace=True", "")
        code = re.sub(r"(df\[['\"]?(\w+)['\"]?\]\.fillna\((.*?)\))",r"df['\2'] = \1",code,)
        code = add_comments_to_top(code, agent_name=AGENT_NAME)
        return {"feature_engineer_function": code,"feature_engineer_error": None}

    def execute_code(state: GraphState):
        return node_func_execute_agent_code_on_data(
            state=state,
            data_key="data_raw",
            result_key="data_engineered",
            error_key="feature_engineer_error",
            code_snippet_key="feature_engineer_function",
            agent_function_name=function_name,
            pre_processing=lambda d: pd.DataFrame.from_dict(d),
            post_processing=lambda df: df.to_dict(),
            error_message_prefix="FE error: ",
        )

    def fix_code(state: GraphState):
        return {"retry_count": state.get("retry_count", 0) + 1}

    def report(state: GraphState):
        return node_func_report_agent_outputs(
            state=state,
            keys_to_include=[
                "recommended_steps",
                "feature_engineer_function",
                "feature_engineer_error",
            ],
            result_key="messages",
            role=AGENT_NAME,
            custom_title="Feature Engineering Results",
        )

    node_funcs = {
        "recommend":    recommend_steps,
        "create_code":  create_code,
        "execute_code": execute_code,
        "fix_code":     fix_code,
        "report":       report,
    }

    return create_coding_agent_graph(
        GraphState=GraphState,
        node_functions=node_funcs,
        recommended_steps_node_name="recommend",
        create_code_node_name="create_code",
        execute_code_node_name="execute_code",
        fix_code_node_name="fix_code",
        explain_code_node_name="report",
        error_key="feature_engineer_error",
        human_in_the_loop=human_in_the_loop,
        human_review_node_name=None,
        checkpointer=checkpointer,
        bypass_recommended_steps=bypass_recommended_steps,
        bypass_explain_code=bypass_explain_code,
        agent_name=AGENT_NAME,
    )
