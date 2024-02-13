import argparse

from langchain_openai import ChatOpenAI

from langchain.agents import AgentType, Tool, initialize_agent
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.tools import DuckDuckGoSearchRun
from langchain.utilities import SerpAPIWrapper


def main(llm, search_tool):
    """
    Example of self-ask-with-search agent
    - DuckDuckGo: ❌ (TODO: fix)
    - SerpAPI: ✅
    Intermediate answer: No good DuckDuckGo Search Result was found
    """

    tools = [
        Tool(
            name="Intermediate Answer",
            func=search_tool.run,
            description="useful for when you need to ask with search",
        )
    ]

    self_ask_with_search = initialize_agent(
        tools=tools, llm=llm, agent=AgentType.SELF_ASK_WITH_SEARCH, verbose=True
    )
    print(
        self_ask_with_search.run(
            "What is the hometown of the reigning men's U.S. Open champion?"
        )
    )
    # self_ask_with_search.run("Obama's first name?")
    # print(search.run("Who is the reigning men's U.S. Open champion?"))


if __name__ == "__main__":
    # search = DuckDuckGoSearchRun()
    # print(search.run("Obama's first name?"))
    parser = argparse.ArgumentParser(prog="self_ask_with_search")
    parser.add_argument("-m", "--model", type=str, default="gpt-3.5-turbo")
    parser.add_argument(
        "-s",
        "--search-tool",
        default=DuckDuckGoSearchRun.__name__,
        choices=[DuckDuckGoSearchRun.__name__, SerpAPIWrapper.__name__],
    )
    args = parser.parse_args()
    llm = ChatOpenAI(
        model=args.model,
        verbose=True,
        callbacks=[StreamingStdOutCallbackHandler()],
    )
    search_tool = globals()[args.search_tool]()
    main(llm=llm, search_tool=search_tool)
