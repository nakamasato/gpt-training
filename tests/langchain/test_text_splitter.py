from src.langchain.text_splitter import create_documents, split_markdown_text


def test_text_split():
    texts = [
        """
このくらい長いテキストを切ってみます。
さらにたくさんのテキストを入れます。
"""
    ]
    doc = create_documents(texts=texts)
    assert len(doc) == 2
    assert doc[0].page_content == "このくらい長いテキストを切ってみます。"
    assert doc[1].page_content == "さらにたくさんのテキストを入れます。"


def test_split_markdown_text():
    filename = "docs/langchain/agent/03_react_custom.md"
    docs = split_markdown_text(md_filename=filename)
    for i, doc in enumerate(docs):
        print(i, doc.page_content[:10])
    assert len(docs) == 10
