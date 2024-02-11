from unittest.mock import patch

from langchain.agents import AgentExecutor, create_react_agent
from langchain.llms.fake import FakeListLLM
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate

from examples.agent_react_custom import CUSTOM_PROMPT, tools


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


@patch("examples.agent_react_custom.google")
def test_google_search(google_mock):
    google_mock.results.return_value = [
        {
            "title": "歴代内閣 | 首相官邸ホームページ",
            "link": "https://www.kantei.go.jp/jp/rekidainaikaku/",
            "snippet": "歴代内閣の情報をご覧になれます。",
        },
        {
            "title": "内閣総理大臣 - Wikipedia",
            "link": "https://ja.wikipedia.org/wiki/%E5%86%85%E9%96%A3%E7%B7%8F%E7%90%86%E5%A4%A7%E8%87%A3",
            "snippet": "内閣総理大臣（ないかくそうりだいじん、英: Prime Minister）は、日本の内閣の首長たる国務大臣。文民である国会議員が就任し、その地位及び権限は日本国憲法や内閣法\xa0...",
        },
        {
            "title": "首相官邸ホームページ",
            "link": "https://www.kantei.go.jp/",
            "snippet": "首相官邸のホームページです。内閣や総理大臣に関する情報をご覧になれます。",
        },
        {
            "title": "岸田総理大臣による第10回核兵器不拡散条約（NPT）運用検討会議 ...",
            "link": "https://www.mofa.go.jp/mofaj/dns/ac_d/page3_003388.html",
            "snippet": "Aug 2, 2022 ... 7月31日から8月1日にかけて、岸田文雄内閣総理大臣は米国を訪問し、日本の総理大臣として初めて、第10回核兵器不拡散条約（NPT）運用検討会議に出席\xa0...",
        },
        {
            "title": "内閣制度と歴代内閣",
            "link": "https://www.kantei.go.jp/jp/rekidai/ichiran.html",
            "snippet": "(外務大臣 内田康哉が内閣総理大臣臨時兼任）. 22, （第2次） 山本權兵衞, 大正12.9.2 －大正13.1.7, 128, 70歳, 嘉永5.10.15, 昭和8.12.8 (81歳), 鹿児島県, 549. 23(13)\xa0...",
        },
    ]
    # prepare fake llm
    responses = [
        "I need to search for current Japan's prime minister\nAction: google-search\nAction Input: current Japan's prime minister",
        "Thought: I now know the final answer\nFinal Answer: 現在の日本の総理大臣は岸田文雄です。",
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
    res = agent_executor.invoke({"input": "日本の現在の総理大臣は誰ですか？"})
    assert res["output"] == "現在の日本の総理大臣は岸田文雄です。"
