from langchain_openai import ChatOpenAI

from langchain_core.prompts import PromptTemplate
from langchain.agents import AgentExecutor, create_react_agent
from src.libs.tools import TOOL_GOOGLE


def create_google_agent_executor(
    llm=ChatOpenAI(model="gpt-3.5-turbo"),
    prompt=PromptTemplate.from_template(
        template="""Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""
    ),
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
