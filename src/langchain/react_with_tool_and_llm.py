from typing import Any, Sequence

from langchain_community.llms import OpenAI

from langchain.agents import AgentExecutor, Tool
from langchain.agents.agent import Agent, AgentOutputParser
from langchain.agents.react.output_parser import ReActOutputParser
from langchain.prompts.base import BasePromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain.tools.base import BaseTool

##########
# define tools
##########

birthplace_dic = {
    "鈴木": "東京",
    "佐藤": "青島",
    "山田": "ソウル",
    "田中": "大阪",
}


def get_birthplace(name):
    return birthplace_dic[name]


##########
# define agent
##########

EXAMPLES = [
    """Question: How much distance between the birthplace of 鈴木 and 佐藤 ?
Thought: I need to get birthplace of 鈴木.
Action: GetBirthplace[鈴木]
Observation: 東京
Thought: I need to get birthplace of 佐藤.
Action: GetBirthplace[佐藤]
Observation: 青島
Thought: I need to get distance between 東京 and 青島.
Action: Llm[I need to get distance between 東京 and 青島.]
Observation: 1750 km
Thought: So the answer is 1750 km.
Action: Finish[1750]""",
]

SUFFIX = """\nQuestion: {input}
{agent_scratchpad}"""

TEST_PROMPT = PromptTemplate.from_examples(
    EXAMPLES, SUFFIX, ["input", "agent_scratchpad"]
)


class ReActTestAgent(Agent):
    @classmethod
    def create_prompt(cls, tools: Sequence[BaseTool]) -> BasePromptTemplate:
        return TEST_PROMPT

    @classmethod
    def _validate_tools(cls, tools: Sequence[BaseTool]) -> None:
        if len(tools) != 2:
            raise ValueError("The number of tools is invalid.")
        tool_names = {tool.name for tool in tools}
        if tool_names != {"GetBirthplace", "Llm"}:
            raise ValueError("The name of tools is invalid.")

    @property
    def _agent_type(self) -> str:
        return "react-test"

    @property
    def finish_tool_name(self) -> str:
        return "Finish"

    @property
    def observation_prefix(self) -> str:
        return "Observation: "

    @property
    def llm_prefix(self) -> str:
        return "Thought: "

    @classmethod
    def _get_default_output_parser(cls, **kwargs: Any) -> AgentOutputParser:
        return ReActOutputParser()


def main():
    ##########
    # run agent
    ##########

    llm = OpenAI()

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

    question = "How much distance between the birthplace of 佐藤 and 田中 ?"
    agent_executor.invoke({"input": question})


if __name__ == "__main__":
    main()
