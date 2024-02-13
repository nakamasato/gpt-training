from unittest.mock import patch

from langchain.llms.fake import FakeListLLM
from src.examples.agent_structured import execute


@patch("src.libs.tools.google")
def test_google_search(google_mock):
    google_mock.results.return_value = [
        {
            "title": "内閣制度と歴代内閣",
            "link": "https://www.kantei.go.jp/jp/rekidai/ichiran.html",
            "snippet": "(外務大臣 内田康哉が内閣総理大臣臨時兼任）. 22, （第2次） 山本權兵衞, 大正12.9.2 －大正13.1.7, 128, 70歳, 嘉永5.10.15, 昭和8.12.8 (81歳), 鹿児島県, 549. 23(13)\xa0...",
        },
    ]
    # prepare fake llm
    responses = [
        '{"action": "google-search","action_input": "current Japan\'s prime minister"}',
        '{"action": "Final Answer", "action_input": "現在の日本の総理大臣は岸田文雄です。"}',
    ]
    llm = FakeListLLM(responses=responses)

    execute(llm, ["日本の総理大臣は誰ですか？"])
