from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI

import langchain
from langchain.agents import AgentExecutor, create_react_agent
from langchain.memory import ConversationBufferMemory
from src.libs.tools import TOOL_GOOGLE, TOOL_PARSE_MULTIPLIER

langchain.debug = False


tools = [TOOL_PARSE_MULTIPLIER, TOOL_GOOGLE]

# almost same as prompt = hub.pull("hwchase17/react-chat")
# https://github.com/hwchase17/langchain-hub/blob/master/agents/zero-shot-react-conversation/agent.json
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


def main(llm):
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
    )

    prompt = PromptTemplate.from_template(template=CUSTOM_PROMPT)
    print(prompt)

    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt,
    )

    print(
        agent.invoke(
            {"input": "現在の日本の総理大臣は誰ですか？", "intermediate_steps": [], "chat_history": []}
        )
    )  # return AgentAction or AgentFinish

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
    llm = OpenAI(temperature=0)
    main(llm=llm)
