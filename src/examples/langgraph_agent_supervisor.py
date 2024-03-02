# https://github.com/langchain-ai/langgraph/blob/main/examples/multi_agent/agent_supervisor.ipynb


# For this example, you will make an agent to do web research with a search engine, and one agent to create plots. Define the tools they'll use below:

import functools
import operator
from pydantic import BaseModel
from typing import Annotated, Sequence, TypedDict

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_experimental.tools import PythonREPLTool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser

tavily_tool = TavilySearchResults(max_results=5)
# This executes code locally, which can be unsafe
python_repl_tool = PythonREPLTool()


class AgentMember(BaseModel):
    name: str
    tools: list
    system_prompt: str


AGENT_MEMBERS = [
    AgentMember(
        name="Researcher",
        tools=[tavily_tool],
        system_prompt="You are a web researcher.",
    ),
    AgentMember(
        name="Coder",
        tools=[python_repl_tool],
        system_prompt="You may generate safe python code to analyze data and generate charts using matplotlib.",
    ),
]


def create_agent(llm: ChatOpenAI, tools: list, system_prompt: str):
    """Define a helper function below, which make it easier to add new agent worker nodes."""
    # Each worker node will be given a name and some tools.
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system_prompt,
            ),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    agent = create_openai_tools_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools)
    return executor


def agent_node(state, agent, name):
    """a function that we will use to be the nodes in the graph
    - it takes care of converting the agent response to a human message.
    This is important because that is how we will add it the global state of the graph"""
    result = agent.invoke(state)
    return {"messages": [HumanMessage(content=result["output"], name=name)]}


def construct_supervisor(llm):
    system_prompt = (
        "You are a supervisor tasked with managing a conversation between the"
        " following workers:  {members}. Given the following user request,"
        " respond with the worker to act next. Each worker will perform a"
        " task and respond with their results and status. When finished,"
        " respond with FINISH."
    )
    # Our team supervisor is an LLM node. It just picks the next agent to process
    # and decides when the work is completed
    options = ["FINISH"] + [m.name for m in AGENT_MEMBERS]
    # Using openai function calling can make output parsing easier for us
    function_def = {
        "name": "route",
        "description": "Select the next role.",
        "parameters": {
            "title": "routeSchema",
            "type": "object",
            "properties": {
                "next": {
                    "title": "Next",
                    "anyOf": [
                        {"enum": options},
                    ],
                }
            },
            "required": ["next"],
        },
    }
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            (
                "system",
                "Given the conversation above, who should act next?" " Or should we FINISH? Select one of: {options}",
            ),
        ]
    ).partial(options=str(options), members=", ".join([m.name for m in AGENT_MEMBERS]))

    supervisor_chain = prompt | llm.bind_functions(functions=[function_def], function_call="route") | JsonOutputFunctionsParser()
    return supervisor_chain


# The agent state is the input to each node in the graph
class AgentState(TypedDict):
    # The annotation tells the graph that new messages will always
    # be added to the current states
    messages: Annotated[Sequence[BaseMessage], operator.add]
    # The 'next' field indicates where to route to next
    next: str


def construct_graph(llm, supervisor_chain):
    workflow = StateGraph(AgentState)

    # supervisor
    workflow.add_node("supervisor", supervisor_chain)

    # agent members
    for member in AGENT_MEMBERS:
        agent = create_agent(llm, member.tools, member.system_prompt)
        node = functools.partial(agent_node, agent=agent, name=member.name)
        workflow.add_node(member.name, node)
        # We want our workers to ALWAYS "report back" to the supervisor when done
        workflow.add_edge(member.name, "supervisor")

    # The supervisor populates the "next" field in the graph state
    # which routes to a node or finishes
    conditional_map = {k.name: k.name for k in AGENT_MEMBERS}
    conditional_map["FINISH"] = END
    workflow.add_conditional_edges("supervisor", lambda x: x["next"], conditional_map)
    # Finally, add entrypoint
    workflow.set_entry_point("supervisor")

    graph = workflow.compile()

    return graph


if __name__ == "__main__":
    llm = ChatOpenAI(model="gpt-4-1106-preview")
    supervisor_chain = construct_supervisor(llm)
    graph = construct_graph(llm, supervisor_chain)

    for s in graph.stream({"messages": [HumanMessage(content="Code hello world and print it to the terminal")]}):
        if "__end__" not in s:
            print(s)
            print("----")
