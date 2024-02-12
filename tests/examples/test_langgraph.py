from unittest.mock import MagicMock, call, patch

from langchain_core.messages import AIMessage, FunctionMessage, HumanMessage

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
    bound_llm = MagicMock()
    bound_llm.invoke.side_effect = [
        AIMessage(
            content="",
            additional_kwargs={
                "function_call": {
                    "arguments": '{\n  "__arg1": "weather in San Francisco"\n}',
                    "name": "google-search",
                }
            },
        ),
        AIMessage(
            content="The weather in San Francisco can vary, but you can check the 10-day weather forecast on The Weather Channel's website [here](https://weather.com/weather/tenday/l/San+Francisco+CA+USCA0987:1:US). Additionally, you can find the current weather conditions and forecast on AccuWeather's website [here](https://www.accuweather.com/en/us/san-francisco/94103/weather-forecast/347629)."  # noqa: E501
        ),
    ]
    llm.bind_functions.return_value = bound_llm

    app = create_example(llm)
    inputs = {"messages": [HumanMessage(content="what is the weather in sf")]}
    res = app.invoke(inputs)
    print(res)

    assert bound_llm.invoke.call_args_list == [
        call(
            [
                HumanMessage(content="what is the weather in sf"),
            ]
        ),
        call(
            [
                HumanMessage(content="what is the weather in sf"),
                AIMessage(
                    content="",
                    additional_kwargs={
                        "function_call": {
                            "arguments": '{\n  "__arg1": "weather in San Francisco"\n}',
                            "name": "google-search",
                        }
                    },
                ),
                FunctionMessage(
                    content="[{'title': '10-Day Weather Forecast for San Francisco, CA - The Weather ...', 'link': 'https://weather.com/weather/tenday/l/San+Francisco+CA+USCA0987:1:US', 'snippet': 'Be prepared with the most accurate 10-day forecast for San Francisco, CA with highs, lows, chance of precipitation from The Weather Channel and Weather.com.'}]",  # noqa: E501
                    name="google-search",
                ),
            ]
        ),
    ]
    assert bound_llm.invoke.call_count == 2
