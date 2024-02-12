from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_openai import ChatOpenAI

from src.libs.tools import TOOL_GOOGLE


def create_google_agent_executor(
    llm=ChatOpenAI(model="gpt-3.5-turbo"),
    prompt=hub.pull("hwchase17/react"),
    memory=None,
) -> AgentExecutor:
    """Return ReAct Agent Executor with Google Search as the only tool."""
    tools_google = [
        TOOL_GOOGLE,
    ]

    print(prompt)

    agent_google = create_react_agent(
        llm=llm,
        tools=tools_google,
        prompt=prompt,
    )

    agent_executor_google = AgentExecutor(
        agent=agent_google,
        tools=tools_google,
        memory=memory,
        verbose=True,
        handle_parsing_errors=False,
    )

    return agent_executor_google
