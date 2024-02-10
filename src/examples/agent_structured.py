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
from langchain_core.prompts import PromptTemplate
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
        description="""Search Google for recent results.
When you call this tool, please carefully consider the best keywords to search with so that you don't need to call this function multiple times.""",
        func=top5_results,
    ),
]

# almost same as hub.pull("hwchase17/react")
CUSTOM_PROMPT = """Answer the following questions as best you can. You have access to Google Search:

{tools}

Use the following format:

To use google-search tool, please use the following format:

```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
```

When you have the answer to the original quertion, or if you do not need to search Google, you MUST use the format:

```
Thought: Do I need to use a tool? No
Final Answer: [your response here]
```

Begin!

Question: {input}
{agent_scratchpad}"""

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
prompt_google = PromptTemplate.from_template(template=CUSTOM_PROMPT)
print(prompt_google)

agent_google = create_react_agent(
    llm=llm,
    tools=tools_google,
    prompt=prompt_google,
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
    # 1. normal agent with google tool
    print(agent_executor_google.invoke({"input": "能登半島の地震の犠牲者は何人ですか"}))
    print("--------- [Complete] normal agent with google as a tool -----------")

    # 2. agent with tools (google agent + mulitplier)
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
            description="""Search Google for recent results.""",
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

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=False,
        handle_parsing_errors=False,
    )

    print(agent_executor.invoke({"input": "3に4を掛けると？"}))
    print(agent_executor.invoke({"input": "5 x 4は？"}))

    print(agent_executor.invoke({"input": "日本の総理大臣は誰ですか？"}))
    print(agent_executor.invoke({"input": "能登半島の地震の犠牲者は何人ですか"}))

    print(memory)


if __name__ == "__main__":
    main()
