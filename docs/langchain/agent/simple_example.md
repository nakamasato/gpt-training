# Simple Example (LangChain 0.1)

## Prerequisites

- `langchain` 0.1
- `TAVILY_API_KEY`

## Steps

### Tool1

```py
from langchain_community.tools.tavily_search import TavilySearchResults

search = TavilySearchResults()

search.invoke("what is the weather in SF")
```

```
>>> search.invoke("what is the weather in SF")
[{'url': 'https://www.weatherapi.com/', 'content': "{'location': {'name': 'San Francisco', 'region': 'California', 'country': 'United States of America', 'lat': 37.78, 'lon': -122.42, 'tz_id': 'America/Los_Angeles', 'localtime_epoch': 1716417732, 'localtime': '2024-05-22 15:42'}, 'current': {'last_updated_epoch': 1716417000, 'last_updated': '2024-05-22 15:30', 'temp_c': 19.4, 'temp_f': 66.9, 'is_day': 1, 'condition': {'text': 'Sunny', 'icon': '//cdn.weatherapi.com/weather/64x64/day/113.png', 'code': 1000}, 'wind_mph': 15.0, 'wind_kph': 24.1, 'wind_degree': 260, 'wind_dir': 'W', 'pressure_mb': 1014.0, 'pressure_in': 29.95, 'precip_mm': 0.0, 'precip_in': 0.0, 'humidity': 61, 'cloud': 0, 'feelslike_c': 19.4, 'feelslike_f': 66.9, 'vis_km': 16.0, 'vis_miles': 9.0, 'uv': 5.0, 'gust_mph': 15.6, 'gust_kph': 25.1}}"}, {'url': 'https://www.yahoo.com/news/may-22-2024-san-francisco-121934270.html', 'content': 'KRON San Francisco. May 22, 2024 San Francisco Bay Area weather forecast. KRON San Francisco. May 22, 2024 at 8:19 AM. Read full article. KRON4 Meteorologist John Shrable has the latest weather ...'}, {'url': 'https://weatherspark.com/h/m/557/2024/5/Historical-Weather-in-May-2024-in-San-Francisco-California-United-States', 'content': 'San Francisco Temperature History May 2024. The daily range of reported temperatures (gray bars) and 24-hour highs (red ticks) and lows (blue ticks), placed over the daily average high (faint red line) and low (faint blue line) temperature, with 25th to 75th and 10th to 90th percentile bands.'}, {'url': 'https://www.wunderground.com/hourly/us/ca/san-francisco/94130/date/2024-05-22', 'content': 'San Francisco Weather Forecasts. Weather Underground provides local & long-range weather forecasts, weatherreports, maps & tropical weather conditions for the San Francisco area. ... Wednesday 05/ ...'}, {'url': 'https://www.weathertab.com/en/c/e/05/united-states/california/san-francisco/', 'content': 'Explore comprehensive May 2024 weather forecasts for San Francisco, including daily high and low temperatures, precipitation risks, and monthly temperature trends. Featuring detailed day-by-day forecasts, dynamic graphs of daily rain probabilities, and temperature trends to help you plan ahead. ... 22 64°F 50°F 18°C 10°C 11% 23 64°F 51°F ...'}]
```

### Tool2

https://python.langchain.com/v0.1/docs/modules/tools/custom_tools/

```py
from langchain.tools import tool
@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b
```

```
>>> multiply
StructuredTool(name='multiply', description='multiply(a: int, b: int) -> int - Multiply two numbers.', args_schema=<class 'pydantic.v1.main.multiplySchema'>, func=<function multiply at 0x10d809da0>)
```

```
>>> multiply.invoke({"a": 1, "b": 2})
2
```

### Set tools

```py
tools = [search, multiply]
```

### Chat

```py
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4", temperature=0)
```

### Prompts

```py
from langchain import hub

# Get the prompt to use - you can modify this!
prompt = hub.pull("hwchase17/openai-functions-agent")
prompt.messages
```

```
>>> prompt.messages
[SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=[], template='You are a helpful assistant')), MessagesPlaceholder(variable_name='chat_history', optional=True), HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['input'], template='{input}')), MessagesPlaceholder(variable_name='agent_scratchpad')]
```

### Agent

```py
from langchain.agents.tool_calling_agent.base import create_tool_calling_agent
agent = create_tool_calling_agent(llm, tools, prompt)
```

```py
agent.invoke({"input": "hi", "intermediate_steps": []})
```

```
>>> agent.invoke({"input": "hi", "intermediate_steps": []})
AgentFinish(return_values={'output': 'Hello! How can I assist you today?'}, log='Hello! How can I assist you today?')
```

### AgentExecutor

```py
from langchain.agents import AgentExecutor

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
```

Example1:

```py
agent_executor.invoke({"input": "What's the weather in SF?"})
```

Example2:

```py
agent_executor.invoke({"input": "What's 2 times 3?"})
```

## Details

### Agent.steps[0]

```py
>>> agent.steps[0]
RunnableAssign(mapper={
  agent_scratchpad: RunnableLambda(lambda x: format_to_tool_messages(x['intermediate_steps']))
})
```

[format_to_tool_messages](https://api.python.langchain.com/en/latest/agents/langchain.agents.format_scratchpad.tools.format_to_tool_messages.html): Convert (AgentAction, tool output) tuples into FunctionMessages.

```py
from langchain.agents.format_scratchpad.tools import (
    format_to_tool_messages,
)
```

```py
def format_to_tool_messages(
    intermediate_steps: Sequence[Tuple[AgentAction, str]],
) -> List[BaseMessage]:
    """Convert (AgentAction, tool output) tuples into FunctionMessages.

    Args:
        intermediate_steps: Steps the LLM has taken to date, along with observations

    Returns:
        list of messages to send to the LLM for the next prediction

    """
    messages = []
    for agent_action, observation in intermediate_steps:
        if isinstance(agent_action, ToolAgentAction):
            new_messages = list(agent_action.message_log) + [
                _create_tool_message(agent_action, observation)
            ]
            messages.extend([new for new in new_messages if new not in messages])
        else:
            messages.append(AIMessage(content=agent_action.log))
    return messages
```

### Agent.steps[1].prompt

```py
>>> agent.steps[1].invoke({"input": "hi,", "agent_scratchpad": []})
ChatPromptValue(messages=[SystemMessage(content='You are a helpful assistant'), HumanMessage(content='hi,')])
>>> agent.steps[1].input_types['agent_scratchpad']
typing.List[typing.Union[langchain_core.messages.ai.AIMessage, langchain_core.messages.human.HumanMessage, langchain_core.messages.chat.ChatMessage, langchain_core.messages.system.SystemMessage, langchain_core.messages.function.FunctionMessage, langchain_core.messages.tool.ToolMessage]]
>>> import langchain_core
>>> agent.steps[1].invoke({"input": "hi,", "agent_scratchpad": [langchain_core.messages.ai.AIMessage(content='a')]})
ChatPromptValue(messages=[SystemMessage(content='You are a helpful assistant'), HumanMessage(content='hi,'), AIMessage(content='a')])
```

`agent_scratchpad`: is set by step[0]
