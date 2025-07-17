

def get_tool_call_names(events):
    tool_calls = []
    for ev in events:
        # ADK Runner events expose a `.tool` attribute when a tool was used
        if hasattr(ev, 'tool') and ev.tool is not None:
            tool_calls.append(ev.tool.name)
    return tool_calls


            