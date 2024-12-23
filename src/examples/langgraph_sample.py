import operator
from typing import Annotated, Sequence, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from src.libs.tools import TOOL_GOOGLE


def create_example(llm=ChatOpenAI(temperature=0, streaming=True)):
    # 1. Set up tools
    tools = [
        TOOL_GOOGLE,
    ]
    tool_node = ToolNode(tools=tools)

    # 3. Bind the tools to the model
    llm = llm.bind_tools(tools)

    # 4. Define AgentState
    class AgentState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], operator.add]

    # 5. Define nodes
    # node: function or runnable (e.g. agent, tool)

    # Define the function that determines whether to continue or not
    def should_continue(state):
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "continue"
        else:
            return "end"

    # Define the function that calls the model
    def call_model(state):
        messages = state["messages"]
        response = llm.invoke(messages)
        # We return a list, because this will get added to the existing list
        return {"messages": [response]}

    # 6. Define the graph

    # Define a new graph
    workflow = StateGraph(AgentState)

    # Define the two nodes we will cycle between
    workflow.add_node("agent", call_model)
    workflow.add_node("action", tool_node)

    # Set the entrypoint as `agent`
    # This means that this node is the first one called
    workflow.set_entry_point("agent")

    # We now add a conditional edge
    workflow.add_conditional_edges(
        # First, we define the start node. We use `agent`.
        # This means these are the edges taken after the `agent` node is called.
        "agent",
        # Next, we pass in the function that will determine which node is called next.
        should_continue,
        # Finally we pass in a mapping.
        # The keys are strings, and the values are other nodes.
        # END is a special node marking that the graph should finish.
        # What will happen is we will call `should_continue`, and then the output of that
        # will be matched against the keys in this mapping.
        # Based on which one it matches, that node will then be called.
        {
            # If `tools`, then we call the tool node.
            "continue": "action",
            # Otherwise we finish.
            "end": END,
        },
    )

    # We now add a normal edge from `tools` to `agent`.
    # This means that after `tools` is called, `agent` node is called next.
    workflow.add_edge("action", "agent")

    # Finally, we compile it!
    # This compiles it into a LangChain Runnable,
    # meaning you can use it as you would any other runnable
    app = workflow.compile()
    return app


if __name__ == "__main__":
    app = create_example()
    # 7. Use
    inputs = {"messages": [HumanMessage(content="what is the weather in sf")]}
    print(app.invoke(inputs))
