import langchain
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain.memory import ConversationBufferMemory
from langchain_openai import OpenAI

from src.libs.tools import TOOL_GOOGLE, TOOL_PARSE_MULTIPLIER

langchain.debug = False

tools = [TOOL_PARSE_MULTIPLIER, TOOL_GOOGLE]


def main():
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
    )

    llm = OpenAI(temperature=0)

    prompt = hub.pull("hwchase17/react-chat")
    print(prompt)

    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt,
    )
    print(
        agent.invoke(
            {
                "input": "Who is Japan's prime minister?",
                "intermediate_steps": [],
                "chat_history": [],
            }
        )
    )  # return AgentAction or AgentFinish

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        handle_parsing_errors=False,
    )

    print(agent_executor.invoke({"input": "3 multiplied by 4 is?"}))
    print(agent_executor.invoke({"input": "5 x 4 is ?"}))

    print(agent_executor.invoke({"input": "Who is Japan's prime minister?"}))
    print(agent_executor.invoke({"input": "能登半島の地震の犠牲者は何人ですか"}))

    print(memory)


if __name__ == "__main__":
    main()
