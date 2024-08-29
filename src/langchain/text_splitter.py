from typing import List

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownTextSplitter, RecursiveCharacterTextSplitter


def create_documents(texts):
    splitter = RecursiveCharacterTextSplitter(
        separators=None,
        chunk_size=20,
        chunk_overlap=0,
    )
    docs = splitter.create_documents(
        texts=texts,
    )
    return docs


def split_text(text) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        separators=None,
        chunk_size=20,
        chunk_overlap=0,
    )
    lst = splitter.split_text(text)
    return lst


def split_markdown_text(md_filename) -> List[Document]:
    splitter = MarkdownTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    loader = TextLoader(md_filename, encoding="utf-8")
    return loader.load_and_split(text_splitter=splitter)


if __name__ == "__main__":
    texts = [
        """
このくらい長いテキストを切ってみます。
さらにたくさんのテキストを入れます。
"""
    ]
    print(create_documents(texts=texts))
    print(split_text(text=texts[0]))

    filename = "src/langchain/agent/03_react_custom.md"
    print(split_markdown_text(md_filename=filename))
