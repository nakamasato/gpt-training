from unittest.mock import MagicMock
from langchain.schema import Document
from langchain_core.messages import AIMessage

from src.examples.langgraph_self_reflection import create_graph, GradeDocuments, GradeHallucinations, GradeAnswer


def test_graph():
    retriever = MagicMock()
    retriever.invoke.side_effect = [
        [
            Document(metadata={"title": "Page 1", "source": "http://example.com/page1"}, page_content=""),
            Document(metadata={"title": "Page 2", "source": "http://example.com/page2"}, page_content=""),
        ],  # retrieve
    ]

    chat_mock = MagicMock()
    chat_mock.side_effect = [
        AIMessage(
            content="""以下のページを見つけました:
1. <http://example.com/page1|Page 1>
2. <http://example.com/page2|Page 2>"""
        ),  # rag_chain generate
    ]

    chat_mock.with_structured_output.return_value.side_effect = [
        GradeDocuments(binary_score="yes"),  # retrieval_grader for the first document
        GradeDocuments(binary_score="no"),  # retrieval_grader for the second document
        GradeHallucinations(binary_score="yes"),  # hallucination_grader
        GradeAnswer(binary_score="yes"),  # answer_grader
    ]

    graph = create_graph(chat=chat_mock, retriever=retriever)

    res = graph.invoke({"question": "What's the origin of the name 'Sofia'?"})

    # Verify the expected behavior
    expected_output = """以下のページを見つけました:
1. <http://example.com/page1|Page 1>
2. <http://example.com/page2|Page 2>"""

    assert res == {
        "question": "What's the origin of the name 'Sofia'?",
        "generation": expected_output,
        "documents": [
            Document(metadata={"title": "Page 1", "source": "http://example.com/page1"}, page_content=""),
        ],  # the second document is filtered out by the retrieval_grader
    }
