# Agent

- [Concepts](https://python.langchain.com/docs/modules/agents/concepts)

## 1. Getting Started

1. Create a tool that splits a given string with a comma and muliply them.

    ```py
    def multiplier(a, b):
        return a * b


    def parsing_multiplier(string):
        a, b = string.split(",")
        return multiplier(int(a), int(b))
    ```

1. Write the usage in the description.

    ```py
    Tool(
        name = "Multiplier",
        func=parsing_multiplier,
        description="useful for when you need to multiply two numbers together. The input to this tool should be a comma separated list of numbers of length two, representing the two numbers you want to multiply together. For example, `1,2` would be the input if you wanted to multiply 1 by 2."
    )
    ```

1. Prepare Prompt

    ```py
    prompt = hub.pull("hwchase17/react-chat")
    ```

1. Create Agent

    ```py
    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt,
    )
    ```

1. Create AgentExecutor

    ```py
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
    )
    ```

1. Invoke

    ```py
    agent_executor.invoke({"input": "3 multiplied by 4 is?"})
    ```

## 2. List

|llm|create agent func|prompt|output parser|link|
|---|---|---|---|---|
|`OpenAI()`|[create_react_agent](https://api.python.langchain.com/en/latest/_modules/langchain/agents/react/agent.html)|[hwchase17/react-chat](https://smith.langchain.com/hub/hwchase17/react-chat)|`ReActSingleInputOutputParser`|[link](https://python.langchain.com/docs/modules/agents/agent_types/react)|
|`ChatOpenAI(model="gpt-3.5-turbo-1106")`|[create_structured_chat_agent](https://api.python.langchain.com/en/latest/_modules/langchain/agents/structured_chat/base.html#create_structured_chat_agent)|[hwchase17/structured-chat-agent](https://smith.langchain.com/hub/hwchase17/structured-chat-chat)|`JSONAgentOutputParser`|[link](https://python.langchain.com/docs/modules/agents/agent_types/structured_chat)|
|`ChatOpenAI(model="gpt-3.5-turbo-1106")`|[create_openai_tools_agent](https://api.python.langchain.com/en/latest/_modules/langchain/agents/openai_tools/base.html#create_openai_tools_agent)|[hwchase17/openai-tools-agent](https://smith.langchain.com/hub/hwchase17/openai-tools-agent)|`OpenAIToolsAgentOutputParser`|[link](https://python.langchain.com/docs/modules/agents/agent_types/openai_tools)|
|`gpt-3.5-turbo-1106`, `gpt-4-0613`|[create_openai_functions_agent](https://api.python.langchain.com/en/latest/_modules/langchain/agents/openai_functions_agent/base.html#create_openai_functions_agent)|[hwchase17/openai-functions-agent](https://smith.langchain.com/hub/hwchase17/openai-functions-agent)|`OpenAIFunctionsAgentOutputParser`|[link](https://python.langchain.com/docs/modules/agents/agent_types/openai_functions_agent)|

For more details about prompt, please read [prompt](prompt.md).

Which agent type to use:

1. **Structured Agent**: you can use tools with mutliple inputs
1. **OpenAI Functions** vs. **OpenAI Tools**:
1. **React**:

## 3. Examples

### 3.1. React Normal `create_react_agent`

https://python.langchain.com/docs/modules/agents/agent_types/react

```py
from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain_openai import OpenAI

import langchain

from langchain import hub
from langchain.agents import AgentExecutor, Tool, create_react_agent
from langchain.memory import ConversationBufferMemory

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
        agent.invoke({"input": "Who is Japan's prime minister?", "intermediate_steps": [], "chat_history": []})
    )  # return AgentAction or AgentFinish

    agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True, handle_parsing_errors=False)

    print(agent_executor.invoke({"input": "3 multiplied by 4 is?"}))
    print(agent_executor.invoke({"input": "5 x 4 is ?"}))

    print(agent_executor.invoke({"input": "Who is Japan's prime minister?"}))
    print(agent_executor.invoke({"input": "能登半島の地震の犠牲者は何人ですか"}))

    print(memory)
```

```
poetry run python src/examples/agent_react_normal.py
```

```
> Entering new AgentExecutor chain...
Thought: Do I need to use a tool? Yes
Action: Multiplier
Action Input: 3,412Do I need to use a tool? No
Final Answer: 3 multiplied by 4 is 12.

> Finished chain.
{'input': '3 multiplied by 4 is?', 'chat_history': [HumanMessage(content='3 multiplied by 4 is?'), AIMessage(content='3 multiplied by 4 is 12.')], 'output': '3 multiplied by 4 is 12.'}
```

```py
agent_google = create_react_agent(
    llm=llm,
    tools=tools_google,
    prompt=prompt_google,
)
```

### 3.2. Structured `create_structured_chat_agent`

- https://python.langchain.com/docs/modules/agents/agent_types/structured_chat
- https://python.langchain.com/docs/modules/agents/tools/custom_tools#structuredtool-dataclass
- [agent_structured.py](https://github.com/nakamasato/gpt-training/blob/main/src/examples/agent_structured.py)


```py
def multiplier(a, b):
    return a * b

...

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
        ...
    ]
```

```py
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

...

tools = [
    ...
    Tool(
        name="google",
        description="Search Google for recent results.",
        func=search_google_with_agent,
    ),
]

agent = create_structured_chat_agent(llm, tools, prompt)
```

## 4. Implementation

### 4.1. [create_react_agent](https://github.com/langchain-ai/langchain/blob/ddaf9de169e629ab3c56a76b2228d7f67054ef04/libs/langchain/langchain/agents/react/agent.py#L16)

The function `create_react_agent` just combines the `llm`, `tools`, `prompt`, `output_parser` and `tools_renderer` with LCEL.

1. An agent is just a `Runnable`.
1. The key part is `llm_with_stop = llm.bind(stop=["\nObservation"])`.

```py
def create_react_agent(
    llm: BaseLanguageModel,
    tools: Sequence[BaseTool],
    prompt: BasePromptTemplate,
    output_parser: Optional[AgentOutputParser] = None,
    tools_renderer: ToolsRenderer = render_text_description,
) -> Runnable:
    missing_vars = {"tools", "tool_names", "agent_scratchpad"}.difference(
        prompt.input_variables
    )
    if missing_vars:
        raise ValueError(f"Prompt missing required variables: {missing_vars}")

    prompt = prompt.partial(
        tools=tools_renderer(list(tools)),
        tool_names=", ".join([t.name for t in tools]),
    )
    llm_with_stop = llm.bind(stop=["\nObservation"])
    output_parser = output_parser or ReActSingleInputOutputParser()
    agent = (
        RunnablePassthrough.assign(
            agent_scratchpad=lambda x: format_log_to_str(x["intermediate_steps"]),
        )
        | prompt
        | llm_with_stop
        | output_parser
    )
    return agent
```


## 5. Ref

1. [Structured chat](https://python.langchain.com/docs/modules/agents/agent_types/structured_chat)
1. [Defining Custom Tools](https://python.langchain.com/docs/modules/agents/tools/custom_tools#structuredtool-dataclass)
1. [agent_structured.py](https://github.com/nakamasato/gpt-training/blob/main/src/examples/agent_structured.py)
