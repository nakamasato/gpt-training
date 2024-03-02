# [create_pandas_dataframe_agent](https://python.langchain.com/docs/integrations/toolkits/pandas)

https://api.python.langchain.com/en/latest/agents/langchain_experimental.agents.agent_toolkits.pandas.base.create_pandas_dataframe_agent.html

1. `df`: can ba a list of DataFrames or a single DataFrame
1. `tools`: `[PythonAstREPLTool(locals=df_locals)]` and you can add `extra_tools`
1. agent_type
    1. `openai-tools`
    1. `openai-functions`
    1. `zero-shot-react-description`: default
1. prompt
    1. zero-shot-react-description:`_get_prompt`
    1. openai-tools or openap-functions: `_get_functions_prompt`

```py
from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
from langchain_experimental.tools.python.tool import PythonAstREPLTool
import pandas as pd
df = pd.read_csv("data/titanic.csv")

agent = create_pandas_dataframe_agent(
    ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613"),
    df,
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS,
)
tools = [PythonAstREPLTool(locals=df)] # default
```

## Prompt

[mrkl/prompt.py](https://github.com/langchain-ai/langchain/blob/7bbff98dc721d5e588662f6f6d9fd916029be1c9/libs/langchain/langchain/agents/mrkl/prompt.py#L3)

```py
FORMAT_INSTRUCTIONS = """Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question"""
```

### ONE_SHOT_REACT_DESCRIPTION + Single DF

[_get_single_prompt](https://github.com/langchain-ai/langchain/blob/7bbff98dc721d5e588662f6f6d9fd916029be1c9/libs/experimental/langchain_experimental/agents/agent_toolkits/pandas/base.py#L70)

<details>

```py
PREFIX = """
You are working with a pandas dataframe in Python. The name of the dataframe is `df`.
You should use the tools below to answer the question posed of you:"""

SUFFIX_NO_DF = """
Begin!
Question: {input}
{agent_scratchpad}"""

SUFFIX_WITH_DF = """
This is the result of `print(df.head())`:
{df_head}

Begin!
Question: {input}
{agent_scratchpad}"""
```

```py
template = "\n\n".join([prefix, "{tools}", FORMAT_INSTRUCTIONS, suffix_to_use])
prompt = PromptTemplate.from_template(template)
```

```py
print("\n\n".join([PREFIX, "{tools}", FORMAT_INSTRUCTIONS, SUFFIX_WITH_DF]))
```

</details>

```
You are working with a pandas dataframe in Python. The name of the dataframe is `df`.
You should use the tools below to answer the question posed of you:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question


This is the result of `print(df.head())`:
{df_head}

Begin!
Question: {input}
{agent_scratchpad}
```

### ONE_SHOT_REACT_DESCRIPTION + Multi DF

<details>

```py
MULTI_DF_PREFIX = """
You are working with {num_dfs} pandas dataframes in Python named df1, df2, etc. You
should use the tools below to answer the question posed of you:"""

SUFFIX_WITH_MULTI_DF = """
This is the result of `print(df.head())` for each dataframe:
{dfs_head}

Begin!
Question: {input}
{agent_scratchpad}"""
```

```py
print("\n\n".join([MULTI_DF_PREFIX, "{tools}", FORMAT_INSTRUCTIONS, SUFFIX_WITH_MULTI_DF]))
```

</details>

```
You are working with {num_dfs} pandas dataframes in Python named df1, df2, etc. You
should use the tools below to answer the question posed of you:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question


This is the result of `print(df.head())` for each dataframe:
{dfs_head}

Begin!
Question: {input}
{agent_scratchpad}
```


### OPENAI_FUNCTIONS + Single DF

<details>

```py
PREFIX_FUNCTIONS = """
You are working with a pandas dataframe in Python. The name of the dataframe is `df`."""
FUNCTIONS_WITH_DF = """
This is the result of `print(df.head())`:
{df_head}"""
```

```py
from langchain_core.messages import SystemMessage
number_of_head_rows: int = 5
df_head = str(df.head(number_of_head_rows).to_markdown())
suffix = FUNCTIONS_WITH_DF.format(df_head=df_head)
PREFIX_FUNCTIONS + suffix
```

</details>

```
You are working with a pandas dataframe in Python. The name of the dataframe is `df`.
This is the result of `print(df.head())`:
|    |   PassengerId |   Survived |   Pclass | Name                                                | Sex    |   Age |   SibSp |   Parch | Ticket           |    Fare | Cabin   | Embarked   |
|---:|--------------:|-----------:|---------:|:----------------------------------------------------|:-------|------:|--------:|--------:|:-----------------|--------:|:--------|:-----------|
|  0 |             1 |          0 |        3 | Braund, Mr. Owen Harris                             | male   |    22 |       1 |       0 | A/5 21171        |  7.25   | nan     | S          |
|  1 |             2 |          1 |        1 | Cumings, Mrs. John Bradley (Florence Briggs Thayer) | female |    38 |       1 |       0 | PC 17599         | 71.2833 | C85     | C          |
|  2 |             3 |          1 |        3 | Heikkinen, Miss. Laina                              | female |    26 |       0 |       0 | STON/O2. 3101282 |  7.925  | nan     | S          |
|  3 |             4 |          1 |        1 | Futrelle, Mrs. Jacques Heath (Lily May Peel)        | female |    35 |       1 |       0 | 113803           | 53.1    | C123    | S          |
|  4 |             5 |          0 |        3 | Allen, Mr. William Henry                            | male   |    35 |       0 |       0 | 373450           |  8.05   | nan     | S          |
```

### OPENAI_FUNCTIONS + Multi DF

prefix:

```py
MULTI_DF_PREFIX_FUNCTIONS = """
You are working with {num_dfs} pandas dataframes in Python named df1, df2, etc."""

FUNCTIONS_WITH_MULTI_DF = """
This is the result of `print(df.head())` for each dataframe:
{dfs_head}"""
```

almost same as one for single df

# [create_csv_agent](https://api.python.langchain.com/en/latest/_modules/langchain_experimental/agents/agent_toolkits/csv/base.html#create_csv_agent)

Exactly same as `create_pandas_dataframe_agent`.

```py
return create_pandas_dataframe_agent(llm, df, **kwargs)
```

Run:

```
poetry run python src/examples/agent_toolkit_csv.py
```

```
> Entering new AgentExecutor chain...

Invoking: `python_repl_ast` with `{'query': 'df.shape[0]'}`


891There are 891 rows in the dataframe.

> Finished chain.
{'input': 'how many rows are there?', 'output': 'There are 891 rows in the dataframe.'}
```
