import re
from typing import Any, Sequence, Union

from langchain_openai import OpenAI

from langchain.agents import AgentExecutor
from langchain.agents.agent import Agent, AgentOutputParser
from src.langchain.react_custom import (
    EXAMPLES,
    SUFFIX,
    WORD_ACTION,
    WORD_FINISH,
    WORD_OBSERVATION,
    WORD_QUESTION,
    WORD_THOUGHT,
    tools,
)
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import CallbackManager
from langchain_core.exceptions import OutputParserException
from langchain_core.prompts import BasePromptTemplate, PromptTemplate
from langchain_core.tools import BaseTool

prompt = PromptTemplate.from_examples(EXAMPLES, SUFFIX, ["word_question", "input", "agent_scratchpad"])
prompt = prompt.partial(word_question=WORD_QUESTION)


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
        return prompt

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
        llm=llm,
        tools=tools,
    )
    callback_manager = CallbackManager.configure(
        inheritable_callbacks=None,
        local_callbacks=None,
        verbose=True,
        inheritable_tags=None,
        local_tags=None,
    )
    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        callback_manager=callback_manager,  # default: None (explicitly write None)
        verbose=True,  # use own PromptStdoutHandler
    )

    question = "How much is the difference between the total of company C, F and the total of company A, E ?"
    result = agent_executor(
        inputs=question,
        return_only_outputs=False,
        callbacks=None,  # ここにStdOutPutHandlerを渡すのでは verbose=Trueとはちょっと違う
    )
    print(result)


if __name__ == "__main__":
    main()
