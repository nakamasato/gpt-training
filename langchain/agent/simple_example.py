from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI

from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor
from langchain.agents.tool_calling_agent.base import create_tool_calling_agent
from langchain_core.tools import tool

search = TavilySearchResults()


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


tools = [search, multiply]

prompt = ChatPromptTemplate(
    messages=[
        ("system", "You are a helpful assistant"),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{placeholder}"),
    ]
)

llm = ChatOpenAI(model="gpt-4", temperature=0)
agent = create_tool_calling_agent(llm, tools, prompt)


agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
agent_executor.invoke({"input": "What's the weather in SF?"})
agent_executor.invoke({"input": "What's 2 times 3?"})
