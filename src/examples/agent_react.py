import langchain

# from langchain import hub
from langchain.agents import AgentExecutor, Tool, create_react_agent
from langchain.memory import ConversationBufferMemory
from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain_core.prompts import PromptTemplate
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

# almost same as prompt = hub.pull("hwchase17/react-chat")
# but as chat_history gets longer, llm tends to return the answer directly without the format.
# so the format instruction is placed at the bottom.
CUSTOM_PROMPT = """Assistant is a large language model trained by OpenAI.

Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.

Overall, Assistant is a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.

TOOLS:
------

Assistant has access to the following tools:

{tools}

You can also reference the previous conversation history:
{chat_history}

----------------

To use a tool, please use the following format:

```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
```

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

```
Thought: Do I need to use a tool? No
Final Answer: [your response here]
```

Begin!

New input: {input}
{agent_scratchpad}"""  # noqa: E501


def main():
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
    )

    llm = OpenAI(temperature=0)

    # prompt = hub.pull("hwchase17/react-chat")
    prompt = PromptTemplate.from_template(template=CUSTOM_PROMPT)
    print(prompt)

    agent_multiplier = create_react_agent(
        llm=llm,
        tools=[tools[0]],
        prompt=prompt,
    )

    agent_google = create_react_agent(
        llm=llm,
        tools=[tools[1]],
        prompt=prompt,
    )

    agent_tools = [
        Tool(
            name="Multiplier",
            func=agent_multiplier.invoke,
            description=(
                "useful for when you need to multiply two numbers together. "
                "The input to this tool should be a comma separated list of numbers of length two, representing the two numbers you want to multiply together. "
                "For example, `1,2` would be the input if you wanted to multiply 1 by 2."
            ),
        ),
        Tool(
            name="google-search",
            description="Search Google for recent results.",
            func=agent_google.invoke,
        ),
    ]

    agent = create_react_agent(
        llm=llm,
        tools=agent_tools,
        prompt=prompt,
    )
    print(
        agent.invoke(
            {"input": "現在の日本の総理大臣は誰ですか？", "intermediate_steps": [], "chat_history": []}
        )
    )  # return AgentAction or AgentFinish

    agent_executor = AgentExecutor(
        agent=agent,
        tools=agent_tools,
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
