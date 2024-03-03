# https://github.com/langchain-ai/langgraph/blob/main/examples/multi_agent/agent_supervisor.ipynb


# For this example, you will make an agent to do web research with a search engine, and one agent to create plots. Define the tools they'll use below:

import functools
import operator
from typing import Annotated, Sequence, TypedDict

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_experimental.tools import PythonREPLTool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel


class MemberAgentConfig(BaseModel):
    name: str
    tools: list
    system_prompt: str


def create_member_agents(llm, agent_configs: list[MemberAgentConfig]):
    """Use this function to generate a dictionary of member agents.
    If you have AgentExecutor objects, you can use them directly.
    """
    agents = {}
    for config in agent_configs:
        agent = create_agent(llm, config.tools, config.system_prompt)
        agents[config.name] = agent
    return agents


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


DEFAULT_SUPERVISOR_SYSTEM_PROMPT = """You are a supervisor tasked with managing a conversation between the
 following workers:  {members}. Given the following user request,
 respond with the worker to act next. Each worker will perform a
 task and respond with their results and status. When finished,
 respond with FINISH."""


def construct_supervisor(llm, agent_member_names: list[str], system_prompt: str = DEFAULT_SUPERVISOR_SYSTEM_PROMPT):
    # Our team supervisor is an LLM node. It just picks the next agent to process
    # and decides when the work is completed
    options = ["FINISH"] + agent_member_names
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
    ).partial(options=str(options), members=", ".join(agent_member_names))

    supervisor_chain = prompt | llm.bind_functions(functions=[function_def], function_call="route") | JsonOutputFunctionsParser()
    return supervisor_chain


# The agent state is the input to each node in the graph
class AgentState(TypedDict):
    # The annotation tells the graph that new messages will always
    # be added to the current states
    messages: Annotated[Sequence[BaseMessage], operator.add]
    # The 'next' field indicates where to route to next
    next: str


def construct_graph(supervisor_chain, agents: dict[str, AgentExecutor]):
    workflow = StateGraph(AgentState)

    # supervisor
    workflow.add_node("supervisor", supervisor_chain)

    # agent members
    nodes = {}
    for name, agent in agents.items():
        node = functools.partial(agent_node, agent=agent, name=name)
        nodes[name] = node

    # add edges (need to add edges after adding nodes)
    for name in agents.keys():
        workflow.add_node(name, nodes[name])
        # We want our workers to ALWAYS "report back" to the supervisor when done
        workflow.add_edge(name, "supervisor")

    # The supervisor populates the "next" field in the graph state
    # which routes to a node or finishes
    conditional_map = {name: name for name in agents.keys()}
    conditional_map["FINISH"] = END
    workflow.add_conditional_edges("supervisor", lambda x: x["next"], conditional_map)
    # Finally, add entrypoint
    workflow.set_entry_point("supervisor")

    graph = workflow.compile()

    return graph


if __name__ == "__main__":
    llm = ChatOpenAI(model="gpt-4")

    AGENT_MEMBERS = [
        MemberAgentConfig(
            name="Researcher",
            tools=[TavilySearchResults(max_results=5)],
            system_prompt="You are a web researcher.",
        ),
        MemberAgentConfig(
            name="Coder",
            tools=[PythonREPLTool()],
            system_prompt="You may generate safe python code to analyze data and generate charts using matplotlib.",
        ),
    ]

    supervisor_chain = construct_supervisor(llm, [m.name for m in AGENT_MEMBERS])
    agents = create_member_agents(llm, AGENT_MEMBERS)
    graph = construct_graph(supervisor_chain, agents)

    questions = [
        # "Code hello world and print it to the terminal",
        "バイデン大統領の年齢は?",
        # "Pythonでデータを分析し、matplotlibでプロットを作成してください。",
    ]
    for q in questions:
        for s in graph.stream({"messages": [HumanMessage(content=q)]}):
            if "__end__" not in s:
                print(s)
                print("----")
        print("finished!")
