import re
from typing import Union

from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.exceptions import OutputParserException
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.tools import StructuredTool, Tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import langchain
from langchain.agents import (  # create_react_agent,
    AgentExecutor,
    create_structured_chat_agent,
)
from langchain.agents.agent import AgentOutputParser
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.mrkl.prompt import FORMAT_INSTRUCTIONS
from langchain.memory import ConversationBufferMemory
from src.libs.tools import TOOL_GOOGLE, multiplier

FINAL_ANSWER_ACTION = "Final Answer:"
MISSING_ACTION_INPUT_AFTER_ACTION_ERROR_MESSAGE = "Invalid Format: Missing 'Action Input:' after 'Action:'"
FINAL_ANSWER_AND_PARSABLE_ACTION_ERROR_MESSAGE = "Parsing LLM output produced both a final answer and a parse-able action:"


langchain.debug = False


# almost same as hub.pull("hwchase17/react")
PROMPT_GOOGLE = """Answer the following questions as best you can using Google Search:

To search Google, please use the following format:

```
Thought: Do I need to search Google? Yes
Google Search Query: the input to the action
Observation: the result of the action
```

When you have the answer to the original quertion, or if you do not need to search Google, you MUST use the format:

```
Thought: Do I need to search Google? No
Final Answer: [your response here]
```

Begin!

Question: {input}
{agent_scratchpad}"""

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
prompt_google = PromptTemplate.from_template(template=PROMPT_GOOGLE)
print(prompt_google)

# agent_google = create_react_agent(
#     llm=llm,
#     tools=tools_google,
#     prompt=prompt_google,
# )


class GoogleSearchOutputParser(AgentOutputParser):
    """Parses GoogleSearch LLM calls based on ReAct-style LLM calls that have a single tool input.

    This is customize version of ReActSingleInputOutputParser.
    from langchain.agents.output_parsers import ReActSingleInputOutputParser

    Expects output to be in one of two formats.

    If the output signals that an action should be taken,
    should be in the below format. This will result in an AgentAction
    being returned.

    ```
    Thought: agent thought here
    Google Search Query: what is the temperature in SF?
    ```

    If the output signals that a final answer should be given,
    should be in the below format. This will result in an AgentFinish
    being returned.

    ```
    Thought: agent thought here
    Final Answer: The temperature is 100 degrees
    ```

    """

    def get_format_instructions(self) -> str:
        return FORMAT_INSTRUCTIONS

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        includes_answer = FINAL_ANSWER_ACTION in text
        regex = r"Google\s*\d*\s*Search\s*\d*\s*Query:[\s]*(.*)"
        action_match = re.search(regex, text, re.DOTALL)
        if action_match:
            if includes_answer:
                raise OutputParserException(f"{FINAL_ANSWER_AND_PARSABLE_ACTION_ERROR_MESSAGE}: {text}")
            action_input = action_match.group(1)
            tool_input = action_input.strip(" ")
            tool_input = tool_input.strip('"')

            return AgentAction("google-search", tool_input, text)

        elif includes_answer:
            return AgentFinish({"output": text.split(FINAL_ANSWER_ACTION)[-1].strip()}, text)

        if not re.search(r"[\s]*Action\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)", text, re.DOTALL):
            raise OutputParserException(
                f"Could not parse LLM output: `{text}`",  # noqa: W604
                observation=MISSING_ACTION_INPUT_AFTER_ACTION_ERROR_MESSAGE,
                llm_output=text,
                send_to_llm=True,
            )
        else:
            raise OutputParserException(
                f"Could not parse LLM output: `{text}`"  # noqa: W604
            )

    @property
    def _type(self) -> str:
        return "google-search"


# replace create_react_agent start
llm_with_stop = llm.bind(stop=["\nObservation"])
agent_google = (
    RunnablePassthrough.assign(
        agent_scratchpad=lambda x: format_log_to_str(x["intermediate_steps"]),
    )
    | prompt_google
    | llm_with_stop
    | GoogleSearchOutputParser()
)
# create_react_agent end

agent_executor_google = AgentExecutor(
    agent=agent_google,
    tools=[TOOL_GOOGLE],
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

    tools = [
        StructuredTool.from_function(
            func=multiplier,
            name="Multiplier",
            description=(
                "useful for when you need to multiply two numbers together. "
                "The input to this tool is two numbers you want to multiply together. "
                "For example, (1, 2) would be the input if you wanted to multiply 1 by 2."
            ),
            return_direct=False,
        ),
        Tool(
            name="google",
            description="""Search Google for recent results.""",
            func=search_google_with_agent,
            return_direct=True,
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
        return_intermediate_steps=False,  # default False
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
