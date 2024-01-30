import os

from langchain.agents import AgentExecutor, create_react_agent
from langchain.llms.fake import FakeListLLM
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate

from src.examples.agent_react import CUSTOM_PROMPT, tools

os.environ["GOOGLE_CSE_ID"] = "dummy"
os.environ["GOOGLE_API_KEY"] = "dummy"


def test_multiplier():
    # prepare fake llm
    responses = [
        " I need to multiply 3 and 4 together\nAction: Multiplier\nAction Input: 3,4",
        "Thought: I now know the final answer\nFinal Answer: 3に4を掛けると12です。",
    ]
    llm = FakeListLLM(responses=responses)

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
    )
    prompt = PromptTemplate.from_template(template=CUSTOM_PROMPT)
    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        handle_parsing_errors=False,
    )
    res = agent_executor.invoke({"input": "3に4を掛けると？"})
    assert res["output"] == "3に4を掛けると12です。"


# Error: Did not find google_api_key, please add an environment variable `GOOGLE_API_KEY` which contains it, or pass `google_api_key` as a named parameter. (type=value_error)
# def test_google_search():
#     # prepare fake llm
#     responses = [
#         "I need to search for current Japan's prime minister\nAction: google-search\nAction Input: current Japan's prime minister",
#         "Thought: I now know the final answer\nFinal Answer: 現在の日本の総理大臣は岸田文雄です。",
#     ]
#     llm = FakeListLLM(responses=responses)

#     memory = ConversationBufferMemory(
#         memory_key="chat_history",
#         return_messages=True,
#     )
#     prompt = PromptTemplate.from_template(template=CUSTOM_PROMPT)
#     agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
#     agent_executor = AgentExecutor(
#         agent=agent,
#         tools=tools,
#         memory=memory,
#         verbose=True,
#         handle_parsing_errors=False,
#     )
#     res = agent_executor.invoke({"input": "日本の現在の総理大臣は誰ですか？"})
#     assert res["output"] == "現在の日本の総理大臣は岸田文雄です。"
