from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms.fake import FakeListLLM
from langchain.tools import Tool
from src.langchain.self_ask_with_search import main


def search(query):
    """dummy search"""
    if "Who" in query:
        return "Novak Djokvoc"
    else:
        return "Belgrade, Serbia"


def test_self_ask_with_search():
    search_tool = Tool.from_function(
        search,
        name="Search",
        description="useful for when you need to ask with search",
    )
    responses = [
        "Yes.\nFollow up: Who is the reigning men's U.S. Open champion?",
        "Yes.\nFollow up: Where is Novak Djokovic from?",
        "So the final answer is: Belgrade, Serbia",
    ]
    llm = FakeListLLM(responses=responses, callbacks=[StreamingStdOutCallbackHandler()])
    main(llm, search_tool)
