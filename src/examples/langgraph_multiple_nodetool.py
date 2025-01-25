from datetime import datetime
import operator
from typing import Annotated, Sequence, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool


@tool
def get_weather(location: str):
    """Get today's weather"""
    if location == "tokyo":
        return "晴れ、2度〜10度"
    else:
        return "雨、15度"


@tool
def get_todays_date():
    """Get today's date in isoformat"""
    return datetime.today().isoformat()


@tool
def multiply(a, b):
    """Multiply two numbers"""
    return a * b


def create_example(llm=ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=True)):
    # 1. Set up tools
    tools1 = [get_todays_date]
    tool_node1 = ToolNode(tools=tools1, messages_key="custom_messages")

    tools2 = [get_weather]
    tool_node2 = ToolNode(tools=tools2, messages_key="custom_messages")

    # 3. Bind the functions to the model
    llm1 = llm.bind_tools(tools1)
    llm2 = llm.bind_tools(tools2)

    # 4. Define AgentState
    class AgentState(TypedDict):
        custom_messages: Annotated[Sequence[BaseMessage], operator.add]

    # 5. Define nodes
    # node: function or runnable (e.g. agent, tool)

    # Define the function that determines whether to continue or not
    def should_continue(state):
        messages = state["custom_messages"]
        last_message = messages[-1]
        # If there is no function call, then we finish
        if last_message.tool_calls:
            return "continue"
        return "end"

    # Define the function that calls the model
    def call_model1(state):
        messages = state["custom_messages"]
        response = llm1.invoke(messages)
        # We return a list, because this will get added to the existing list
        return {"custom_messages": [response]}

    def call_model2(state):
        messages = state["custom_messages"]
        response = llm2.invoke(messages)
        # We return a list, because this will get added to the existing list
        return {"custom_messages": [response]}

    # 6. Define the graph

    # Define a new graph
    workflow = StateGraph(AgentState)

    # Define the two nodes we will cycle between
    workflow.add_node("agent1", call_model1)
    workflow.add_node("action1", tool_node1)
    workflow.add_node("agent2", call_model2)
    workflow.add_node("action2", tool_node2)

    # Set the entrypoint as `agent`
    # This means that this node is the first one called
    workflow.set_entry_point("agent1")

    # We now add a conditional edge
    workflow.add_conditional_edges(
        "agent1",
        should_continue,
        {
            "continue": "action1",
            "end": "agent2",
        },
    )

    workflow.add_conditional_edges(
        "agent2",
        should_continue,
        {
            "continue": "action2",
            "end": END,
        },
    )

    workflow.add_edge("action1", "agent1")
    workflow.add_edge("action2", "agent2")

    app = workflow.compile()
    return app


if __name__ == "__main__":
    app = create_example()
    print(app.get_graph().draw_mermaid())
    # 7. Use
    inputs = {"custom_messages": [HumanMessage(content="本日の日付の年、月、日を掛け算した値と天気から占ってください。")]}
    # you can specify run_name, metadata for langsmith
    for e in app.stream(inputs, config={"run_name": "ExampleAgent", "metadata": {"tenant": "t1"}, "project": "t1"}):
        print(e)
