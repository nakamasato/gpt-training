from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI

import langchain
from langchain import hub
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain.memory import ConversationBufferMemory
from langchain.tools import StructuredTool
from src.libs.agents import create_google_agent_executor
from src.libs.tools import multiplier

langchain.debug = False


def execute(llm, questions):
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
    )

    prompt = hub.pull("hwchase17/structured-chat-agent")
    print(prompt)

    agent_executor_google = create_google_agent_executor(llm)

    def search_google_with_agent(query):
        """Use agent instead of function calling to let LLM to write the final sentence to answer the original question.
        If this tool should be used with return_direct=True.
        But the following example must use return_direct=False.
        """
        return agent_executor_google.invoke({"input": query})["output"]

    tools = [
        StructuredTool.from_function(
            func=multiplier,
            name="Multiplier",
            description=(
                "useful for when you need to multiply two numbers together. "
                "The input to this tool is two numbers you want to multiply together. "
                "For example, (1, 2) would be the input if you wanted to multiply 1 by 2."
            ),
            return_direct=False,  # must use False because the prompt exept action & action_input in the output parser
        ),
        Tool(
            name="google",
            description="Search Google for recent results.",
            func=search_google_with_agent,
            return_direct=False,  # must use False because the prompt exept action & action_input in the output parser
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
    # print(agent_executor_google.invoke({"input": "能登半島の地震の犠牲者は何人ですか"}))

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        handle_parsing_errors=False,
    )

    for q in questions:
        print(agent_executor.invoke({"input": q}))

    print(memory)


if __name__ == "__main__":
    llm = ChatOpenAI()
    questions = [
        "3に4を掛けると？",
        "5 x 4は？",
        "日本の総理大臣は誰ですか？",
        "能登半島の地震の犠牲者は何人ですか",
    ]
    execute(llm, questions)
