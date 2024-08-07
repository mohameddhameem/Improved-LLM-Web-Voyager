import re
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableLambda
from langgraph.graph import END, StateGraph
from agent_types import AgentState
from agent import agent
from tools import click, type_text, scroll, wait, go_back, to_google
from agent_types import AgentState
from agent import AgentAction  # Add this import

def update_scratchpad(state: AgentState):
    """After a tool is invoked, we want to update
    the scratchpad so the agent is aware of its previous steps"""
    old = state.get("scratchpad")
    if old:
        txt = old[0].content
        last_line = txt.rsplit("\n", 1)[-1]
        step = int(re.match(r"\d+", last_line).group()) + 1
    else:
        txt = "Previous action observations:\n"
        step = 1
    
    prediction = state['prediction']
    txt += f"\n{step}. Action: {prediction.action}, Args: {prediction.args}, Observation: {state['observation']}"

    return {**state, "scratchpad": [SystemMessage(content=txt)]}

MAX_STEPS = 10  # Adjust this value as needed

def select_tool(state: AgentState):
    prediction = state["prediction"]
    if prediction.step_count >= MAX_STEPS:
        return END
    
    action = prediction.action
    if action == "ANSWER":
        return END
    if action == "retry":
        return "agent"
    return action

def create_graph():
    graph_builder = StateGraph(AgentState)

    graph_builder.add_node("agent", agent)
    graph_builder.set_entry_point("agent")

    graph_builder.add_node("update_scratchpad", update_scratchpad)
    graph_builder.add_edge("update_scratchpad", "agent")

    tools = {
        "Click": click,
        "Type": type_text,
        "Scroll": scroll,
        "Wait": wait,
        "GoBack": go_back,
        "Google": to_google,
    }

    for node_name, tool in tools.items():
        graph_builder.add_node(
            node_name,
            RunnableLambda(tool) | (lambda observation: {"observation": observation}),
        )
        graph_builder.add_edge(node_name, "update_scratchpad")

    graph_builder.add_conditional_edges("agent", select_tool)

    return graph_builder.compile()