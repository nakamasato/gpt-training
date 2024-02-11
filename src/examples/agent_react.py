import langchain
from langchain import hub
from langchain.agents import AgentExecutor, Tool, create_react_agent
from langchain.memory import ConversationBufferMemory
from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain_openai import OpenAI

langchain.debug = False


def multiplier(a, b):
    return a * b


def parsing_multiplier(string):
    a, b = string.split(",")
    return multiplier(int(a), int(b))


google = GoogleSearchAPIWrapper()


def top5_results(query):
    return google.results(query, 5)


tools = [
    Tool(
        name="Multiplier",
        func=parsing_multiplier,
        description=(
            "useful for when you need to multiply two numbers together. "
            "The input to this tool should be a comma separated list of numbers of length two, representing the two numbers you want to multiply together. "
            "For example, `1,2` would be the input if you wanted to multiply 1 by 2."
        ),
    ),
    Tool(
        name="google-search",
        description="Search Google for recent results.",
        func=top5_results,
    ),
]


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
