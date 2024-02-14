from langchain.llms.fake import FakeListLLM
from src.langchain.callback import PromptStdoutHandler


def test_callback():
    responses = ["Action: Python REPL\nAction Input: print(2 + 2)", "Final Answer: 4"]
    llm = FakeListLLM(responses=responses, callbacks=[PromptStdoutHandler()])
    llm.invoke("Tell me a joke")
    # poetry run pytest -k test_callback -s
