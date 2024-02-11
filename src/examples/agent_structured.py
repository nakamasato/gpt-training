import langchain
from dotenv import load_dotenv
from langchain import hub
from langchain.agents import (
    AgentExecutor,
    create_react_agent,
    create_structured_chat_agent,
)
from langchain.memory import ConversationBufferMemory
from langchain.tools import StructuredTool
from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI

load_dotenv()

langchain.debug = False


def multiplier(a, b):
    return a * b


google = GoogleSearchAPIWrapper()


def top5_results(query):
    return google.results(query, 5)


tools_google = [
    Tool(
        name="google-search",
        description="Search Google for recent results.",
        func=top5_results,
    ),
]

prompt = hub.pull("hwchase17/react")

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
print(prompt)

agent_google = create_react_agent(
    llm=llm,
    tools=tools_google,
    prompt=prompt,
)
agent_executor_google = AgentExecutor(
    agent=agent_google,
    tools=tools_google,
    # memory=memory,
    verbose=True,
    handle_parsing_errors=False,
)


def search_google_with_agent(query):
    return agent_executor_google.invoke({"input": query})["output"]


def main():
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
    )

    prompt = hub.pull("hwchase17/structured-chat-agent")

    print(prompt)

    tools = [
        StructuredTool.from_function(
            func=multiplier,
            name="Multiplier",
            description=(
                "useful for when you need to multiply two numbers together. "
                "The input to this tool is two numbers you want to multiply together. "
                "For example, (1, 2) would be the input if you wanted to multiply 1 by 2."
            ),
        ),
        Tool(
            name="google",
            description="Search Google for recent results.",
            func=search_google_with_agent,
        ),
    ]

    agent = create_structured_chat_agent(llm, tools, prompt)

    print(
        agent.invoke(
            input={
                "input": "現在の日本の総理大臣は誰ですか？",
                "intermediate_steps": [],
                "chat_history": [],
            }
        )
    )  # return AgentAction or AgentFinish
    print(agent_executor_google.invoke({"input": "能登半島の地震の犠牲者は何人ですか"}))

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        handle_parsing_errors=False,
    )

    print(agent_executor.invoke({"input": "3に4を掛けると？"}))
    print(agent_executor.invoke({"input": "5 x 4は？"}))

    print(agent_executor.invoke({"input": "日本の総理大臣は誰ですか？"}))
    print(agent_executor.invoke({"input": "能登半島の地震の犠牲者は何人ですか"}))

    print(memory)


if __name__ == "__main__":
    main()
