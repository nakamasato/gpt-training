# [LangGraph](https://python.langchain.com/docs/langgraph)

## Overview

![](langgraph.drawio.svg)

**LangGraph** is a library for building stateful, multi-actor applications with LLMs, built on top of (and intended to be used with) LangChain.

!!! note
    The main use is for adding **cycles** to your LLM application.

1. **AgentState**: Defined by `TypedDict` with
1. **StateGraph**:
1. **Node**:
1. **Edge**:
    1. A conditional edge can be defined by `add_conditional_edge`
    1. An edge can be defined by `add_edge`

## Getting Started

1. Create a tool

1. Create `ToolExecutor`

    ```py
    from langgraph.prebuilt import ToolExecutor
    tool_executor = ToolExecutor(tools)
    ```
1. Set up a model
    ```py
    model = ChatOpenAI(temperature=0, streaming=True)

    from langchain.tools.render import format_tool_to_openai_function

    functions = [format_tool_to_openai_function(t) for t in tools]
    model = model.bind_functions(functions)
    ```
1. Define `StatefulGraph`

    ```py
    from typing import TypedDict, Annotated, Sequence
    import operator
    from langchain_core.messages import BaseMessage


    class AgentState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], operator.add]
    ```
1. Define node
    1. Agent
    1. Tool
1. Define graph
    1. Create workflow
    1. Add Edge
1. Use it.

