


from langchain_core.messages import AIMessage

from langgraph.graph import StateGraph,END
from langgraph.types import interrupt,Command
from langgraph.graph.state import CompiledStateGraph

#define Type
from typing import Type

from langchain_core.runnables import RunnableConfig

from langgraph.pregel.types import StreamMode

import pandas as pd

import sqlalchemy as sql

import json 

from typing import Any, Dict, Annotated, Optional, Callable,Union,List
from agents_of_ds.parsers.parsers import PythonOutputParser
from utils.regex import (
    relocate_imports_inside_function,
    add_comments_to_top,
    remove_consecutive_duplicates

)

from IPython.display import Image,display
import pandas as pd

class BaseAgent(CompiledStateGraph):
    def __init__(self, **params):
        self._params = params
        self._compiled_graph = self._make_compiled_graph()
        self.response = None
        self.name = self._compiled_graph.name
        self.checkpointer = self._compiled_graph.checkpointer
        self.store = self._compiled_graph.store
        self.output_channels = self._compiled_graph.output_channels
        self.nodes = self._compiled_graph.nodes
        self.stream_mode = self._compiled_graph.stream_mode
        self.builder = self._compiled_graph.builder
        self.channels = self._compiled_graph.channels
        self.input_channels = self._compiled_graph.input_channels
        self.input_schema = self._compiled_graph.input_schema
        self.output_schema = self._compiled_graph.output_schema
        self.debug = self._compiled_graph.debug
        self.interrupt_after_nodes = self._compiled_graph.interrupt_after_nodes
        self.interrupt_before_nodes = self._compiled_graph.interrupt_before_nodes
        self.config = self._compiled_graph.config
    
    def make_compiled_graph(self):
        raise NotImplementedError("Subclasses should implement this method.")
    
    
    def update_params(self,**kwargs):
        self._params.update(kwargs)
        self._compiled_graph = self._make_compiled_graph()
        
        
    def __getattr__(self,name, str):
        return getattr(self._compiled_graph,name)
    
    def invoke(
        self,
        input : Union[dict[str,Any],Any],
        config: Optional[RunnableConfig] = None,
        **kwargs
    ):
        self.response= self._compiled_graph.invoke(input=input,config=config,**kwargs)
        if self.response.get("messages"):
            self.response["messages"]=remove_consecutive_duplicates(self.response["messages"])
        return self.response
    
    
    async def ainvoke(
        self,
        input : Union[dict[str,Any],Any],
        config: Optional[RunnableConfig] = None,
        **kwargs
    ):
        self.response= await self._compiled_graph.ainvoke(input=input,config=config,**kwargs)
        if self.response.get("messages"):
            self.response["messages"]=remove_consecutive_duplicates(self.response["messages"])
        return self.response
    
    
    def stream(
        self,
        input : dict[str,Any] | Any,
        config: RunnableConfig | None = None,
        stream_mode : StreamMode | list[StreamMode] | None =None,
        **kwargs
        
    ):
        self.response=  self._compiled_graph.stream(input=input,config=config,stream_mode=stream_mode,**kwargs)
        if self.response.get("messages"):
            self.response["messages"]=remove_consecutive_duplicates(self.response["messages"])
        return self.response
    
    
    async def stream(
        self,
        input : dict[str,Any] | Any,
        config: RunnableConfig | None = None,
        stream_mode : StreamMode | list[StreamMode] | None =None,
        **kwargs
    ):
        self.response=await self._compiled_graph.stream(input=input,config=config,stream_mode=stream_mode,**kwargs)
        if self.response.get("messages"):
            self.response["messages"]=remove_consecutive_duplicates(self.response["messages"])
        return self.response
    
    
    def get_state_keys(self):
        return list(self.get_output_jsonschema()['properties'].keys())
    
    
    def get_state_properties(self):
        return self.get_output_jsonschema()['properties']
    
    
    def get_state(self,config,*,subgraphs=False):
        
        return self._compiled_graph.get_state(config,subgraphs=subgraphs)
    
    
    def get_state_history(self,config,*,filter=None,before=None,limit =None):
        return self._compiled_graph.get_state_history(config,filter=filter,before=before,limit=limit)
    
    
    def update_state(self, config, values, as_node = None):
        return self._compiled_graph.update_state(config, values, as_node)
    
    def get_response(self):
        if self.response.get("messages"):
            self.response["messages"] = remove_consecutive_duplicates(self.response["messages"])  
        
        return self.response

    def show(self, xray: int = 0):
        display(Image(self.get_graph(xray=xray).draw_mermaid_png()))
        



def create_coding_agent_graph(
    GraphState: Type,
    node_functions: Dict[str, Callable],
    recommended_steps_node_name: str,
    create_code_node_name: str,
    execute_code_node_name: str,
    fix_code_node_name: str,
    explain_code_node_name: str,
    error_key: str,
    max_retries_key: str = "max_retries",
    retry_count_key: str = "retry_count",
    human_in_the_loop: bool = False,
    human_review_node_name: str = "human_review",
    checkpointer: Optional[Callable] = None,
    bypass_recommended_steps: bool = False,
    bypass_explain_code: bool = False,
    agent_name: str = "coding_agent"
):

    workflow = StateGraph(GraphState)
    
    # * NODES
    
    # Always add create, execute, and fix nodes
    workflow.add_node(create_code_node_name, node_functions[create_code_node_name])
    workflow.add_node(execute_code_node_name, node_functions[execute_code_node_name])
    if  fix_code_node_name is not None:
        workflow.add_node(fix_code_node_name, node_functions[fix_code_node_name])
    
    if not bypass_recommended_steps:
        workflow.add_node(recommended_steps_node_name, node_functions[recommended_steps_node_name])
    
    if human_in_the_loop:
        workflow.add_node(human_review_node_name, node_functions[human_review_node_name])
    
    if not bypass_explain_code:
        workflow.add_node(explain_code_node_name, node_functions[explain_code_node_name])
    
    entry_point = create_code_node_name if bypass_recommended_steps else recommended_steps_node_name
    
    workflow.set_entry_point(entry_point)
    
    if not bypass_recommended_steps:
        workflow.add_edge(recommended_steps_node_name, create_code_node_name)
    
    workflow.add_edge(create_code_node_name, execute_code_node_name)
    if fix_code_node_name is not None:
        workflow.add_edge(fix_code_node_name, execute_code_node_name)
    
    # Define a helper to check if we have an error & can still retry
    def error_and_can_retry(state):
        return (
            state.get(error_key) is not None
            and state.get(retry_count_key) is not None
            and state.get(max_retries_key) is not None
            and state[retry_count_key] < state[max_retries_key]
        )
        
    # If human in the loop, add a branch for human review
    if human_in_the_loop:
        workflow.add_conditional_edges(
            execute_code_node_name,
            lambda s: "fix_code" if error_and_can_retry(s) else "human_review",
            {
                "human_review": human_review_node_name,
                "fix_code": fix_code_node_name,
            },
        )
    else:
        # If no human review, the next node is fix_code if error, else explain_code.
        if not bypass_explain_code:
            workflow.add_conditional_edges(
                execute_code_node_name,
                lambda s: "fix_code" if error_and_can_retry(s) else "explain_code",
                {
                    "fix_code": fix_code_node_name,
                    "explain_code": explain_code_node_name,
                },
            )
        else:
            workflow.add_conditional_edges(
                execute_code_node_name,
                lambda s: "fix_code" if error_and_can_retry(s) else "END",
                {
                    "fix_code": fix_code_node_name,
                    "END": END,
                },
            )
            
    if not bypass_explain_code:
        workflow.add_edge(explain_code_node_name, END)
    
    # Finally, compile
    app = workflow.compile(
        checkpointer=checkpointer,
        name=agent_name,
    )
    
    return app



def node_func_human_review(
    state: Any, 
    prompt_text: str, 
    yes_goto: str, 
    no_goto: str,
    user_instructions_key: str = "user_instructions",
    recommended_steps_key: str = "recommended_steps",
    code_snippet_key: str = "code_snippet",
    code_type: str = "python"
) -> Command[str]:
    print("    * HUMAN REVIEW")
    
    code_markdown=f"```{code_type}\n" + state.get(code_snippet_key)+"\n```"

    # Display instructions and get user response
    user_input = interrupt(value=prompt_text.format(steps=state.get(recommended_steps_key, '') + "\n\n" + code_markdown))

    # Decide next steps based on user input
    if user_input.strip().lower() == "yes":
        goto = yes_goto
        update = {}
    else:
        goto = no_goto
        modifications = "User Has Requested Modifications To Previous Code: \n" + user_input
        if state.get(user_instructions_key) is None:
            update = {user_instructions_key: modifications + "\n\nPrevious Code:\n" + code_markdown}
        else:
            update = {user_instructions_key: state.get(user_instructions_key) + modifications + "\n\nPrevious Code:\n" + code_markdown}

    return Command(goto=goto, update=update)


def node_func_execute_agent_code_on_data(
    state: Any, 
    data_key: str, 
    code_snippet_key: str, 
    result_key: str,
    error_key: str,
    agent_function_name: str,
    pre_processing: Optional[Callable[[Any], Any]] = None, 
    post_processing: Optional[Callable[[Any], Any]] = None,
    error_message_prefix: str = "An error occurred during agent execution: "
) -> Dict[str, Any]:
    
    print("    * EXECUTING AGENT CODE")
    
    # Retrieve raw data and code snippet from the state
    data = state.get(data_key)
    agent_code = state.get(code_snippet_key)
    
    # Preprocessing: If no pre-processing function is given, attempt a default handling
    if pre_processing is None:
        if isinstance(data, dict):
            df = pd.DataFrame.from_dict(data)
        elif isinstance(data, list):
            df = [pd.DataFrame.from_dict(item) for item in data]
        else:
            raise ValueError("Data is not a dictionary or list and no pre_processing function was provided.")
    else:
        df = pre_processing(data)
    
    # Execute the code snippet to define the agent function
    local_vars = {}
    global_vars = {}
    exec(agent_code, global_vars, local_vars)
    
    # Retrieve the agent function from the executed code
    agent_function = local_vars.get(agent_function_name, None)
    if agent_function is None or not callable(agent_function):
        raise ValueError(f"Agent function '{agent_function_name}' not found or not callable in the provided code.")
    
    # Execute the agent function
    agent_error = None
    result = None
    try:
        result = agent_function(df)
        
        # Test an error
        # if state.get("retry_count") == 0:
        #     10/0
        
        # Apply post-processing if provided
        if post_processing is not None:
            result = post_processing(result)
        else:
            if isinstance(result, pd.DataFrame):
                result = result.to_dict()   
        
    except Exception as e:
        print(e)
        agent_error = f"{error_message_prefix}{str(e)}"
    
    # Return results
    output = {result_key: result, error_key: agent_error}
    return output

def node_func_execute_agent_from_sql_connection(
    state: Any, 
    connection: Any, 
    code_snippet_key: str, 
    result_key: str,
    error_key: str,
    agent_function_name: str,
    post_processing: Optional[Callable[[Any], Any]] = None,
    error_message_prefix: str = "An error occurred during agent execution: "
) -> Dict[str, Any]:
    
    
    print("    * EXECUTING AGENT CODE ON SQL CONNECTION")
    
    # Retrieve SQLAlchemy connection and code snippet from the state
    is_engine = isinstance(connection, sql.engine.base.Engine)
    connection = connection.connect() if is_engine else connection
    agent_code = state.get(code_snippet_key)
    
    # Ensure the connection object is provided
    if connection is None:
        raise ValueError(f"Connection object not found.")
    
    # Execute the code snippet to define the agent function
    local_vars = {}
    global_vars = {}
    exec(agent_code, global_vars, local_vars)
    
    # Retrieve the agent function from the executed code
    agent_function = local_vars.get(agent_function_name, None)
    if agent_function is None or not callable(agent_function):
        raise ValueError(f"Agent function '{agent_function_name}' not found or not callable in the provided code.")
    
    # Execute the agent function
    agent_error = None
    result = None
    try:
        result = agent_function(connection)
        
        # Apply post-processing if provided
        if post_processing is not None:
            result = post_processing(result)
    except Exception as e:
        print(e)
        agent_error = f"{error_message_prefix}{str(e)}"
    
    # Return results
    output = {result_key: result, error_key: agent_error}
    return output


def node_func_fix_agent_code(
    state: Any, 
    code_snippet_key: str, 
    error_key: str, 
    llm: Any, 
    prompt_template: str,
    agent_name: str,
    retry_count_key: str = "retry_count",
    log: bool = False,
    file_path: str = "logs/agent_function.py",
    function_name: str = "agent_function"
) -> dict:
    print("    * FIX AGENT CODE")
    print("      retry_count:" + str(state.get(retry_count_key)))
    
    # Retrieve the code snippet and the error from the state
    code_snippet = state.get(code_snippet_key)
    error_message = state.get(error_key)

    # Format the prompt with the code snippet and the error
    prompt = prompt_template.format(
        code_snippet=code_snippet,
        error=error_message,
        function_name=function_name,
    )
    
    # Execute the prompt with the LLM
    response = (llm | PythonOutputParser()).invoke(prompt)
    
    response = relocate_imports_inside_function(response)
    response = add_comments_to_top(response, agent_name=agent_name)
    
    # Log the response if requested
    if log:
        with open(file_path, 'w') as file:
            file.write(response)
            print(f"      File saved to: {file_path}")
    
    # Return updated results
    return {
        code_snippet_key: response,
        error_key: None,
        retry_count_key: state.get(retry_count_key) + 1
    }

def node_func_explain_agent_code(
    state: Any, 
    code_snippet_key: str,
    result_key: str,
    error_key: str,
    llm: Any, 
    role: str,
    explanation_prompt_template: str,
    success_prefix: str = "# Agent Explanation:\n\n",
    error_message: str = "The agent encountered an error during execution and cannot be explained."
) -> Dict[str, Any]:
    
    print("   * EXPLAIN AGENT CODE")
    agent_error=state.get(error_key)
    if agent_error is None:
        code_snippet=state.get(code_snippet_key )
        prompt=explanation_prompt_template.format(code_snippet=code_snippet)
        response=llm.invoke(prompt)
        message=AIMessage(content=f"{success_prefix}{response.content}",role=role)
        return {"messages": [message]}
    else:
        message=AIMessage(content=error_message)
        return {result_key: [message]}
    
def node_func_report_agent_outputs(
    state: Dict[str,Any],
    keys_to_include: List[str],
    result_key:str,
    role : str,
    custom_title: str="Agent Output Summary")-> Dict[str, Any]:
    print("    * REPORT AGENT OUTPUTS")
    final_report={"report_title" : custom_title}
    for key in keys_to_include:
        final_report[key]=state.get(key, f"<{key}_not_found_in_state>")
        
    return {
        result_key: [
            AIMessage(
                content=json.dumps(final_report, indent=2), 
                role=role
            )
        ]
    }
    
    
    
    
    