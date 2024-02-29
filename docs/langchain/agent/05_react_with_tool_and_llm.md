# Custom ReAct with tool and llm

ToolとLLMに聞くのをうまく組み合わせるやり方が分からなかったので、Toolで直接llmに聞くというのを作った。

```py
    tools = [
        Tool(
            name="GetBirthplace",
            func=get_birthplace,
            description="Get birthplace of a person.",
        ),
        Tool(
            name="Llm",
            func=llm,
            description="Use this tool to ask general questions"
        ),
    ]
```

## Run

```
poetry run python src/langchain/react_with_tool_and_llm.py
```

```
> Entering new AgentExecutor chain...
Thought: I need to get birthplace of 佐藤.
Action: GetBirthplace[佐藤]
Observation: 青島
Thought:  I need to get birthplace of 田中.
Action: GetBirthplace[田中]
Observation: 大阪
Thought:  I need to get distance between 青島 and 大阪.
Action: Llm[I need to get distance between 青島 and 大阪.]
Observation:

The approximate distance between Qingdao and Osaka is 1,845 kilometers (1,145 miles).
Thought:  So the answer is 1845 km.
Action: Finish[1845]

> Finished chain.
```
