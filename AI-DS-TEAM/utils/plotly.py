

import json
import plotly.io as pio 
def plotly_from_dict(plotly_graph_dict:dict):
    if plotly_from_dict is None:
        return None
    return pio.from_json(json.dumps(plotly_graph_dict))

