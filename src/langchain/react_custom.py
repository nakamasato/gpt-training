import re
from typing import Any, Sequence, Union

# from langchain.agents.react.output_parser import ReActOutputParser
from langchain_openai import OpenAI

from langchain.agents import AgentExecutor
from langchain.agents.agent import Agent, AgentOutputParser
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.exceptions import OutputParserException
from langchain_core.prompts import BasePromptTemplate, PromptTemplate
from langchain_core.tools import BaseTool, Tool

##########
# define tools
##########

company_dic = {
    "A": 2000,
    "B": 1500,
    "C": 20000,
    "D": 6700,
    "E": 1000,
    "F": 4100,
}


def get_invoice(company_name):
    return company_dic[company_name]


def diff(value_str):
    str_list = value_str.split(" ")
    assert len(str_list) == 2
    int_list = [int(s) for s in str_list]
    return str(abs(int_list[0] - int_list[1]))


def total(value_str):
    str_list = value_str.split(" ")
    int_list = [int(s) for s in str_list]
    return str(sum(int_list))


tools = [
    Tool(
        name="GetInvoice",
        func=get_invoice,
        description="Get invoice amount of trading company.",
    ),
    Tool(
        name="Diff",
        func=diff,
        description="Get diffrence.",
    ),
    Tool(
        name="Total",
        func=total,
        description="Get total.",
    ),
]

##########
# define agent
##########
WORD_QUESTION = "質問"  # Question
WORD_THOUGHT = "思考"  # Thought
WORD_ACTION = "行動"  # Action
WORD_OBSERVATION = "観察"  # Observation
WORD_FINISH = "完了"  # Finish


EXAMPLES = [
    f"""{WORD_QUESTION}: How much is the difference between the invoice of company A and company B ?
{WORD_THOUGHT}: I need to get invoice amount of company A.
{WORD_ACTION}: GetInvoice[A]
{WORD_OBSERVATION}: 2000
{WORD_THOUGHT}: I need to get invoice amount of company B.
{WORD_ACTION}: GetInvoice[B]
{WORD_OBSERVATION}: 1500
{WORD_THOUGHT}: I need to get difference of obtained amount between company A and company B.
{WORD_ACTION}: Diff[2000 1500]
{WORD_OBSERVATION}: 500
{WORD_THOUGHT}: So the answer is 500.
{WORD_ACTION}: {WORD_FINISH}[500]""",
    f"""{WORD_QUESTION}: How much is the total invoice amount of company B, C, and D ?
{WORD_THOUGHT}: I need to get invoice amount of company B.
{WORD_ACTION}: GetInvoice[B]
Observation 1: 1500
{WORD_THOUGHT}: I need to get invoice amount of company C.
{WORD_ACTION}: GetInvoice[C]
{WORD_OBSERVATION}: 20000
{WORD_THOUGHT}: I need to get invoice amount of company D.
{WORD_ACTION}: GetInvoice[D]
{WORD_OBSERVATION}: 6700
{WORD_THOUGHT}: I need to get total amount of obtained amount B, C, and D.
{WORD_ACTION}: Total[1500 20000 6700]
{WORD_OBSERVATION}: 28200
{WORD_THOUGHT}: So the answer is 28200.
{WORD_ACTION}: {WORD_FINISH}[28200]""",
    f"""{WORD_QUESTION}: How much is the difference between company C and the total invoice amount of company A, D ?
{WORD_THOUGHT}: I need to get invoice amount of company C.
{WORD_ACTION}: GetInvoice[C]
{WORD_OBSERVATION}: 20000
{WORD_THOUGHT}: I need to get invoice amount of company A.
{WORD_ACTION}: GetInvoice[A]
{WORD_OBSERVATION}: 2000
{WORD_THOUGHT}: I need to get invoice amount of company D.
{WORD_ACTION}: GetInvoice[D]
{WORD_OBSERVATION}: 6700
{WORD_THOUGHT}: I need to get total amount of obtained amount A and D.
{WORD_ACTION}: Total[2000 6700]
{WORD_OBSERVATION}: 8700
{WORD_THOUGHT}: I need to get difference of obtained amount C and the total of A, D.
{WORD_ACTION}: Diff[20000 8700]
{WORD_OBSERVATION}: 11300
{WORD_THOUGHT}: So the answer is 11300.
{WORD_ACTION}: {WORD_FINISH}[11300]""",
]

SUFFIX = """\n{word_question}: {input}
{agent_scratchpad}"""

prompt = PromptTemplate.from_examples(EXAMPLES, SUFFIX, ["word_question", "input", "agent_scratchpad"])
TEST_PROMPT = prompt.partial(word_question=WORD_QUESTION)


class ReActOutputParser(AgentOutputParser):
    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        action_prefix = f"{WORD_ACTION}: "
        if not text.strip().split("\n")[-1].startswith(action_prefix):
            raise OutputParserException(f"Could not parse LLM Output: {text}")
        action_block = text.strip().split("\n")[-1]

        action_str = action_block[len(action_prefix) :]

        re_matches = re.search(r"(.*?)\[(.*?)\]", action_str)
        if re_matches is None:
            raise OutputParserException(f"Could not parse action directive: {action_str}")
        action, action_input = re_matches.group(1), re_matches.group(2)

        # 最後が行動: Finishであれば処理を終わらせる
        if action == WORD_FINISH:
            return AgentFinish({"output": action_input}, text)
        else:
            return AgentAction(action, action_input, text)


class ReActTestAgent(Agent):
    @classmethod
    def create_prompt(cls, tools: Sequence[BaseTool]) -> BasePromptTemplate:
        return TEST_PROMPT

    @classmethod
    def _validate_tools(cls, tools: Sequence[BaseTool]) -> None:
        if len(tools) != 3:
            raise ValueError("The number of tools is invalid.")
        tool_names = {tool.name for tool in tools}
        if tool_names != {"GetInvoice", "Diff", "Total"}:
            raise ValueError("The name of tools is invalid.")

    @property
    def _agent_type(self) -> str:
        return "react-test"

    @property
    def finish_tool_name(self) -> str:
        return WORD_FINISH

    @property
    def observation_prefix(self) -> str:
        return f"{WORD_OBSERVATION}: "

    @property
    def llm_prefix(self) -> str:
        return f"{WORD_THOUGHT}: "

    @classmethod
    def _get_default_output_parser(cls, **kwargs: Any) -> AgentOutputParser:
        return ReActOutputParser()


def main():
    ##########
    # run agent
    ##########

    llm = OpenAI()
    agent = ReActTestAgent.from_llm_and_tools(
        llm,
        tools,
    )
    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=True,
    )

    question = "How much is the difference between the total of company C, F and the total of company A, E ?"
    res = agent_executor.invoke({"input": question})
    print(res)


if __name__ == "__main__":
    main()
