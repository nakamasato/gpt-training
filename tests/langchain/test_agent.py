from langchain_core.tools import Tool

from langchain.agents import initialize_agent
from langchain.llms.fake import FakeListLLM


def multiplier(a, b):
    return a * b


def parsing_multiplier(string):
    a, b = string.split(",")
    return multiplier(int(a), int(b))


tools = [
    Tool(
        name="Multiplier",
        func=parsing_multiplier,
        description="""useful for when you need to multiply two numbers together.
The input to this tool should be a comma separated list of numbers of length two,
representing the two numbers you want to multiply together. For example, `1,2` would be the input if you wanted to multiply 1 by 2.""",
    )
]


def test_main():
    responses = [
        " I need to multiply 3 and 4 together\nAction: Multiplier\nAction Input: 3,4",
        "Thought: I now know the final answer\nFinal Answer: 3に4を掛けると12です。",
    ]
    llm = FakeListLLM(responses=responses)
    mrkl = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)

    result = mrkl.invoke({"input": "3に4を掛けると？"})
    assert result["output"] == "3に4を掛けると12です。"
