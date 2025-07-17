"""
Data Cleaning Agent using LangGraph + Google GenAI (Gemini 2.0).
Always produces a valid `data_cleaner` function and returns a DataFrame.
"""
import os
import ast
import textwrap
import operator
from typing import TypedDict, Annotated, Any, Sequence, Optional
import re
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
from utils.regex import (
    add_comments_to_top,
    format_agent_name,
    format_recommended_steps,
)
from agents_of_ds.tools.dataframe import get_dataframe_summary
from google import genai
from pydantic import PrivateAttr

AGENT_NAME = "data_cleaning_agent"
LOG_PATH   = os.path.join(os.getcwd(), "logs/")


class ADKModelWrapper(LLM):
    model_name: str = "gemini-2.0-flash"
    _client: Any = PrivateAttr()

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, api_key: str, model_name: Optional[str] = None):
        super().__init__()
        self.model_name = model_name or self.model_name
        self._client = genai.Client(api_key=api_key)

    def _call(self, prompt: Any, **kwargs) -> str:
        txt = prompt.to_string() if hasattr(prompt, "to_string") else str(prompt)
        resp = self._client.models.generate_content(
            model=self.model_name,
            contents=txt,
        )
        return resp.text

    @property
    def _identifying_params(self) -> dict:
        return {"model_name": self.model_name}

    @property
    def _llm_type(self) -> str:
        return self.model_name


class GraphState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    user_instructions: str
    recommended_steps: str
    dataset_summary: str
    data_raw: dict
    data_cleaned: dict
    data_cleaner_function: str
    data_cleaner_error: str
    max_retries: int
    retry_count: int


class DataCleaningAgent(BaseAgent):
    def __init__(
        self,
        model: LLM,
        n_samples: int = 30,
        log: bool = False,
        log_path: Optional[str] = None,
        file_name: str = "data_cleaner.py",
        function_name: str = "data_cleaner",
        overwrite: bool = True,
        human_in_the_loop: bool = False,
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
            "bypass_explain_code": bypass_explain_code,
            "checkpointer": checkpointer,
        }
        self._compiled_graph = self._make_compiled_graph()
        self.response = None

    def _make_compiled_graph(self):
        if self._params["human_in_the_loop"] and not self._params["checkpointer"]:
            self._params["checkpointer"] = MemorySaver()
        return make_data_cleaning_agent(**self._params)

    def invoke_agent(
        self,
        data_raw: pd.DataFrame,
        user_instructions: str = None,
        max_retries: int = 3,
        retry_count: int = 0,
        **kwargs,
    ):
        self.response = self._compiled_graph.invoke(
            {
                "user_instructions": user_instructions or "",
                "dataset_summary": None,
                "data_raw": data_raw.to_dict(),
                "max_retries": max_retries,
                "retry_count": retry_count,
            },
            **kwargs,
        )

    def get_recommended_cleaning_steps(self, markdown: bool = False):
        text = self.response["messages"][-1].content
        from IPython.display import Markdown as _MD
        return _MD(text) if markdown else text

    def get_data_cleaned(self) -> Optional[pd.DataFrame]:
        cleaned = self.response.get("data_cleaned")
        return pd.DataFrame(cleaned) if cleaned is not None else None

    def get_data_cleaner_function(self, markdown: bool = False):
        code = self.response.get("data_cleaner_function", "")
        from IPython.display import Markdown as _MD
        return _MD(f"```python\n{code}\n```") if markdown else code


def make_data_cleaning_agent(
    model: LLM,
    n_samples=30,
    log=False,
    log_path=None,
    file_name="data_cleaner.py",
    function_name="data_cleaner",
    overwrite=True,
    human_in_the_loop=False,
    bypass_explain_code=False,
    checkpointer: Checkpointer = None,
):
    llm = model

    def recommend_steps(state: GraphState):
        print(format_agent_name(AGENT_NAME), "* RECOMMEND CLEANING STEPS")
        df = pd.DataFrame.from_dict(state["data_raw"])
        summary = "\n\n".join(get_dataframe_summary([df], n_sample=n_samples))
        prompt = PromptTemplate(
            template="""
You are a Data Cleaning Expert. Recommend numbered cleaning steps (no code):
User instructions: {user_instructions}
Dataset summary: {dataset_summary}
""",
            input_variables=["user_instructions", "dataset_summary"],
        )
        resp = (prompt | llm).invoke(
            {
                "user_instructions": state["user_instructions"],
                "dataset_summary": summary,
            }
        )
        return {
            "recommended_steps": format_recommended_steps(resp, heading="# Cleaning Steps:"),
            "dataset_summary": summary,
        }

    def create_code(state: GraphState):
        print(format_agent_name(AGENT_NAME), "* CREATE CLEANER CODE")

        # 1) Ask LLM for body only
        prompt = PromptTemplate(
            template="""Generate only the *body* of a function `{function_name}(data_raw)` to
            implement these steps (no wrapper, no imports):
            {recommended_steps}""",
            input_variables=["recommended_steps", "function_name"],
        )
        raw = (prompt | llm | PythonOutputParser()).invoke({
            "recommended_steps": state["recommended_steps"],
            "function_name": function_name,
        })
        body_text = raw if isinstance(raw, str) else raw.get("text", "")

        # 2) Dedent the LLM’s answer
        dedented = textwrap.dedent(body_text)

        # 3) Parse into AST
        try:
            parsed = ast.parse(dedented)
        except SyntaxError as e:
            # fallback: wrap every line as a pass
            parsed = ast.parse("")
        stmts = parsed.body  # list of AST statements

        # 4) Build imports & copy AST nodes
        imports = [
            ast.Import(names=[ast.alias(name="pandas", asname="pd")]),
            ast.Import(names=[ast.alias(name="numpy", asname="np")]),
            ast.ImportFrom(module="scipy.stats.mstats",names=[ast.alias(name="winsorize", asname=None)],level=0),
            ast.ImportFrom(module="scipy.stats",names=[ast.alias(name="iqr", asname=None)],level=0),
        ]
        copy_node = ast.Assign(
            targets=[ast.Name(id="df", ctx=ast.Store())],
            value=ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="data_raw", ctx=ast.Load()),
                    attr="copy",
                    ctx=ast.Load(),
                ),
                args=[],
                keywords=[],
            ),
        )
        return_node = ast.Return(value=ast.Name(id="df", ctx=ast.Load()))

        # 5) Build the function def
        func_def = ast.FunctionDef(
            name=function_name,
            args=ast.arguments(
                posonlyargs=[],
                args=[ast.arg(arg="data_raw")],
                vararg=None,
                kwonlyargs=[],
                kw_defaults=[],
                defaults=[],
            ),
            body=[*imports, copy_node, *stmts, return_node],
            decorator_list=[],
        )
        module = ast.Module(body=[func_def], type_ignores=[])
        ast.fix_missing_locations(module)

        # 6) Unparse back to code (Python 3.9+)
        code = ast.unparse(module)

        # 7) Clean up fillna/inplace (same as before)
        code = code.replace(", inplace=True", "")
        code = re.sub(
            r"(df\[['\"]?(\w+)['\"]?\]\.fillna\((.*?)\))",
            r"df['\2'] = \1",
            code,
        )

        # 8) Prepend header comments
        code = add_comments_to_top(code, agent_name=AGENT_NAME)
        return {"data_cleaner_function": code, "data_cleaner_error": None}

    def execute_code(state: GraphState):
        return node_func_execute_agent_code_on_data(
            state=state,
            data_key="data_raw",
            result_key="data_cleaned",
            error_key="data_cleaner_error",
            code_snippet_key="data_cleaner_function",
            agent_function_name=function_name,
            pre_processing=lambda d: pd.DataFrame.from_dict(d),
            post_processing=lambda df: df.to_dict(),
            error_message_prefix="Error during cleaning: ",
        )

    def fix_code(state: GraphState):
        return {"retry_count": state.get("retry_count", 0) + 1}

    def report_outputs(state: GraphState):
        return node_func_report_agent_outputs(
            state=state,
            keys_to_include=[
                "recommended_steps",
                "data_cleaner_function",
                "data_cleaner_error",
            ],
            result_key="messages",
            role=AGENT_NAME,
            custom_title="Data Cleaning Results",
        )

    node_funcs = {
        "recommend_steps": recommend_steps,
        "create_code":    create_code,
        "execute_code":   execute_code,
        "fix_code":       fix_code,
        "report":         report_outputs,
    }

    return create_coding_agent_graph(
        GraphState=GraphState,
        node_functions=node_funcs,
        recommended_steps_node_name="recommend_steps",
        create_code_node_name="create_code",
        execute_code_node_name="execute_code",
        fix_code_node_name="fix_code",
        explain_code_node_name="report",
        error_key="data_cleaner_error",
        human_in_the_loop=human_in_the_loop,
        human_review_node_name=None,
        checkpointer=checkpointer,
        bypass_recommended_steps=False,
        bypass_explain_code=bypass_explain_code,
        agent_name=AGENT_NAME,
    )
