import os

from langchain_openai import OpenAI

from langchain.llms.fake import FakeListLLM


def get_llm():
    if os.getenv("ENV") == "test":
        responses = [
            "Action: Python REPL\nAction Input: print(2 + 2)",
            "Final Answer: 4",
        ]
        return FakeListLLM(responses=responses)
    return OpenAI(temperature=0, streaming=True)
