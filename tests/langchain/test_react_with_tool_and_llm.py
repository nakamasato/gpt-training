from langchain.agents import AgentExecutor, Tool
from langchain.llms.fake import FakeListLLM
from src.langchain.react_with_tool_and_llm import ReActTestAgent, get_birthplace


def test_react_with_tool_and_llm():
    """
    The test case is not based on the fact
    """
    responses = [
        "Thought: I need to get birthplace of 鈴木\nAction: GetBirthplace[鈴木]",
        "Thought: I need to get birthplace of 佐藤.\nAction: GetBirthplace[佐藤]",
        "Thought: I need to get distance between 東京 and 青島.\nAction: Llm[I need to get distance between 東京 and 青島.]",
        "1750km",
        "Thought: So the answer is 1750 km.\nAction: Finish[1750]",
    ]
    llm = FakeListLLM(responses=responses)

    tools = [
        Tool(
            name="GetBirthplace",
            func=get_birthplace,
            description="Get birthplace of a person.",
        ),
        Tool(
            name="Llm", func=llm, description="Use this tool to ask general questions"
        ),
    ]

    agent = ReActTestAgent.from_llm_and_tools(
        llm,
        tools,
    )
    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=True,
    )
    question = "How much distance between the birthplace of 鈴木 and 佐藤?"
    res = agent_executor.invoke({"input": question})

    assert res["output"] == "1750"
