from langchain.agents import AgentExecutor, initialize_agent
from src.langchain.react_docstore import tools
from langchain_community.llms import FakeListLLM


def test_react_docstore_tool():
    """
    The test case is not based on the fact
    """
    responses = [
        "Thought: I need to search 岸田総理 and find when he was thrown a bomb during a speech.\nAction: Search[岸田総理]",
        "Thought: The paragraph mentions 岸田総理 was thrown a bomb during a speech in 1993. So the answer is 1993.\nAction: Finish[1993]",
    ]
    llm = FakeListLLM(responses=responses)

    # initialize ReAct agent
    react = initialize_agent(tools, llm, agent="react-docstore", verbose=True)
    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=react.agent,
        tools=tools,
        #   max_iterations=2,
        verbose=True,
    )

    # perform question-answering
    question = "岸田総理が演説中に爆弾を投げ込まれたのはいつ?"
    result = agent_executor.invoke({"input": question})
    assert result["output"] == "1993"  # wrong
