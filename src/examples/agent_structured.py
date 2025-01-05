from langchain_core.tools import StructuredTool, Tool
from langchain_openai import ChatOpenAI

import langchain
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain.memory import ConversationBufferMemory
from src.libs.agents import create_google_agent_executor
from src.libs.tools import multiplier

langchain.debug = False


def execute(llm, questions):
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
    )

    prompt = ChatPromptTemplate(
        messages=[
            (
                "system",
                """Respond to the human as helpfully and accurately as possible. You have access to the following tools:

{tools}

Use a json blob to specify a tool by providing an action key (tool name) and an action_input key (tool input).

Valid "action" values: "Final Answer" or {tool_names}

Provide only ONE action per $JSON_BLOB, as shown:

```
{{
  "action": $TOOL_NAME,
  "action_input": $INPUT
}}
```

Follow this format:

Question: input question to answer
Thought: consider previous and subsequent steps
Action:
```
$JSON_BLOB
```
Observation: action result
... (repeat Thought/Action/Observation N times)
Thought: I know what to respond
Action:
```
{{
  "action": "Final Answer",
  "action_input": "Final response to human"
}}

Begin! Reminder to ALWAYS respond with a valid json blob of a single action. Use tools if necessary. Respond directly if appropriate. Format is Action:```$JSON_BLOB```then Observation""",
            ),
            ("placeholder", "{chat_history}"),
            (
                "human",
                """{input}

{agent_scratchpad}
 (reminder to respond in a JSON blob no matter what)""",
            ),
        ],
    )
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
