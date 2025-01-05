from unittest.mock import MagicMock, call, patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.examples.langgraph_sample import create_example


@patch("src.libs.tools.google")
def test_langgraph(google_mock):
    google_mock.results.return_value = [
        {
            "title": "10-Day Weather Forecast for San Francisco, CA - The Weather ...",
            "link": "https://weather.com/weather/tenday/l/San+Francisco+CA+USCA0987:1:US",
            "snippet": "Be prepared with the most accurate 10-day forecast for San Francisco, CA with highs, lows, chance of precipitation from The Weather Channel and Weather.com.",
        },
    ]

    llm = MagicMock()
    llm.bind_tools.return_value.invoke.side_effect = [
        AIMessage(
            content="",
            additional_kwargs={
                "tool_calls": [{"index": 0, "id": "call_aJsmWxcJT2zEqvmqDeMddm4B", "function": {"arguments": '{"__arg1":"weather in San Francisco"}', "name": "google-search"}, "type": "function"}]
            },
            response_metadata={"finish_reason": "tool_calls", "model_name": "gpt-3.5-turbo-0125"},
            id="run-e5a2304c-36b7-43e7-9983-29a0b7f49857-0",
            tool_calls=[{"name": "google-search", "args": {"__arg1": "weather in San Francisco"}, "id": "call_aJsmWxcJT2zEqvmqDeMddm4B", "type": "tool_call"}],
        ),  # call_model
        AIMessage(
            content="The weather in San Francisco can vary, but you can check the 10-day weather forecast on The Weather Channel's website [here](https://weather.com/weather/tenday/l/San+Francisco+CA+USCA0987:1:US). Additionally, you can find the current weather conditions and forecast on AccuWeather's website [here](https://www.accuweather.com/en/us/san-francisco/94103/weather-forecast/347629).",  # noqa: E501
            tool_calls=[],
        ),  # should_continue
    ]

    app = create_example(llm)
    inputs = {"messages": [HumanMessage(content="what is the weather in sf")]}
    res = app.invoke(inputs)
    print(res)

    assert llm.bind_tools.return_value.invoke.call_args_list == [
        call(
            [
                HumanMessage(content="what is the weather in sf"),
            ]
        ),
        call(
            [
                HumanMessage(content="what is the weather in sf", additional_kwargs={}, response_metadata={}),
                AIMessage(
                    content="",
                    additional_kwargs={
                        "tool_calls": [
                            {"index": 0, "id": "call_aJsmWxcJT2zEqvmqDeMddm4B", "function": {"arguments": '{"__arg1":"weather in San Francisco"}', "name": "google-search"}, "type": "function"}
                        ]
                    },
                    response_metadata={"finish_reason": "tool_calls", "model_name": "gpt-3.5-turbo-0125"},
                    id="run-e5a2304c-36b7-43e7-9983-29a0b7f49857-0",
                    tool_calls=[{"name": "google-search", "args": {"__arg1": "weather in San Francisco"}, "id": "call_aJsmWxcJT2zEqvmqDeMddm4B", "type": "tool_call"}],
                ),
                ToolMessage(
                    content='[{"title": "10-Day Weather Forecast for San Francisco, CA - The Weather ...", "link": "https://weather.com/weather/tenday/l/San+Francisco+CA+USCA0987:1:US", "snippet": "Be prepared with the most accurate 10-day forecast for San Francisco, CA with highs, lows, chance of precipitation from The Weather Channel and Weather.com."}]',
                    name="google-search",
                    tool_call_id="call_aJsmWxcJT2zEqvmqDeMddm4B",
                ),
            ]
        ),
    ]
    assert llm.bind_tools.return_value.invoke.call_count == 2
