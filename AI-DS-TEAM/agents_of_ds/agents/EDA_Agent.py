import sys
import os
import yaml
import pandas as pd
import math

from typing import Any, Dict, Optional, Sequence

# add project root to path
here = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(here))
sys.path.insert(0, project_root)
from langchain_core.messages import BaseMessage
from langgraph.graph import START, END, StateGraph
from langgraph.types import Checkpointer
from agents_of_ds.tempis.templates_of_agent import BaseAgent
from utils.regex import format_agent_name

from agents_of_ds.tools.eda import (
    explain_data,
    describe_dataset,
    visualize_missing,
    generate_correlation_funnel,
    generate_sweetviz_report,
    generate_dtale_report,
)

from google.adk import Agent as ADKAgent, Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.function_tool import FunctionTool
from google.genai import types as genai_types

AGENT_NAME = "exploratory_data_analyst_agent"
APP_NAME   = "eda_app"
USER_ID    = "user_001"
SESSION_ID = "session_001"

def load_api_key() -> str:
    cred_path = "D:\\Agentic_AI_Compiled\\Agentic_AI_DS_Team\\Visuals\\credentials.yml"
    creds = yaml.safe_load(open(cred_path))
    gem = creds.get("gemini_api_key")
    if not gem:
        raise KeyError("`gemini_api_key` missing in credentials.yml")
    return gem

def sanitize_for_json(obj):
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
    return obj

def create_eda_tools(df: pd.DataFrame, target: str) -> Sequence[FunctionTool]:
    mapping = [
        ("ExplainData", explain_data, (df.to_dict(),)),
        ("DescribeDataset", describe_dataset, (df.to_dict(),)),
        ("VisualizeMissing", visualize_missing, (df.to_dict(),)),
        ("CorrelationFunnel", generate_correlation_funnel, (df.to_dict(), target)),
        ("SweetvizReport", generate_sweetviz_report, (df.to_dict(),)),
        ("DtaleReport", generate_dtale_report, (df.to_dict(),)),
    ]
    tools = []
    for name, fn, args in mapping:
        def make_wrapper(func, func_args):
            def wrapper():
                res = func(*func_args)
                if isinstance(res, tuple) and len(res) == 2 and isinstance(res[1], dict):
                    text, artifact = res
                    return {"text": text, "artifact": sanitize_for_json(artifact)}
                if isinstance(res, dict):
                    return sanitize_for_json(res)
                return {"text": res}
            return wrapper
        wrapped = make_wrapper(fn, args)
        wrapped.__name__ = name
        wrapped.__doc__ = fn.__doc__
        tools.append(FunctionTool(func=wrapped))
    return tools

environment_runner: Optional[Runner] = None

def init_runner(df: pd.DataFrame, target: str) -> Runner:
    global environment_runner
    if environment_runner is None:
        os.environ["gemini_api_key"] = load_api_key()
        tools = create_eda_tools(df, target)
        adk_agent = ADKAgent(
            name=AGENT_NAME,
            model="gemini-2.0-flash",
            description="EDA via Gemini",
            tools=tools,
        )
        svc = InMemorySessionService()
        svc.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)
        environment_runner = Runner(agent=adk_agent, app_name=APP_NAME, session_service=svc)
    return environment_runner

class EDAToolsAgent(BaseAgent):
    def __init__(self):
        self.response: Dict[str, Any] = {}
    



    def invoke_agent(self, user_instructions: str, data_raw: Dict[str, Any], target: str):
        df = pd.DataFrame.from_dict(data_raw)
        runner = init_runner(df, target)
        content = genai_types.Content(role="user", parts=[genai_types.Part(text=user_instructions)])
        events = runner.run(user_id=USER_ID, session_id=SESSION_ID, new_message=content)

        parts = []
        artifact = None
        tool_calls = []
        def flatten_and_str(x):
            if isinstance(x, list):
                return "\n".join(flatten_and_str(i) for i in x)
            return str(x)

        for ev in events:
            for call in ev.get_function_calls():
                tool_calls.append(call.name)
            for resp in ev.get_function_responses():
                d = getattr(resp, "response", {}) or {}
                txt = d.get("text")
                if txt:
                    parts.append(flatten_and_str(txt))
                if "artifact" in d:
                    artifact = d["artifact"]
            if ev.is_final_response():
                parts.append(flatten_and_str(ev.content.parts[0].text))


        final_text = "\n\n".join(parts).strip()
        self.response = {
            "messages": [final_text],
            "internal_messages": events,
            "eda_artifacts": artifact,
            "tool_calls": tool_calls,
        }

    def get_ai_message(self, markdown: bool = False) -> str:
        msg = self.response.get("messages", [""])[0]
        return msg if not markdown else f"```text\n{msg}\n```"

    def get_artifacts(self, as_dataframe: bool = False):
        art = self.response.get("eda_artifacts")
        if as_dataframe and isinstance(art, dict):
            return pd.DataFrame(art)
        return art

    def get_tool_calls(self):
        return self.response.get("tool_calls", [])

def make_eda_tools_agent(checkpointer: Optional[Checkpointer] = None):
    class State(BaseMessage):
        user_instructions: str
        data_raw: Dict[str, Any]
        target: str

    def node(state: Any):
        agent = EDAToolsAgent()
        agent.invoke_agent(
            user_instructions=state["user_instructions"],
            data_raw=state["data_raw"],
            target=state["target"],
        )
        return {
            "messages": [agent.get_ai_message()],
            "internal_messages": agent.response["internal_messages"],
            "eda_artifacts": agent.get_artifacts(),
            "tool_calls": agent.get_tool_calls(),
        }

    wf = StateGraph(State)
    wf.add_node("explore", node)
    wf.add_edge(START, "explore")
    wf.add_edge("explore", END)
    return wf.compile(checkpointer=checkpointer, name=AGENT_NAME)
