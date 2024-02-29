# [Text Splitter](https://python.langchain.com/docs/modules/data_connection/document_transformers/text_splitters/character_text_splitter)

## Overview

Embeddingするまえにちょうどいい長さやChunkに分割するために使われる。

https://langchain-text-splitter.streamlit.app/ でいろんなケースを試してみるといい。

## Types

1. [TextSplitter](https://github.com/hwchase17/langchain/blob/ee70d4a0cd743fd9376ebc81d2948a85f83e621f/langchain/text_splitter.py#L73C7-L73C19): Abstract class
1. [CharacterTextSplitter](https://github.com/hwchase17/langchain/blob/ee70d4a0cd743fd9376ebc81d2948a85f83e621f/langchain/text_splitter.py#L261C7-L261C28)
1. [RecursiveCharacterTextSplitter](https://github.com/hwchase17/langchain/blob/ee70d4a0cd743fd9376ebc81d2948a85f83e621f/langchain/text_splitter.py#L601): `from_language`を使うことで、対応するseparatorsによってSplitする。 `split_documents`を使ってdocumentを再分割することもできる
1. [MarkdownHeaderTextSplitter](https://github.com/hwchase17/langchain/blob/ee70d4a0cd743fd9376ebc81d2948a85f83e621f/langchain/text_splitter.py#L292C7-L292C33): Markdownのヘッダーによって分割する。 metadataにそのヘッダー情報も入るのでいい。
1. [TokenTextSplitter](https://github.com/hwchase17/langchain/blob/ee70d4a0cd743fd9376ebc81d2948a85f83e621f/langchain/text_splitter.py#L460C7-L460C24)
1. [SentenceTransformersTokenTextSplitter](https://github.com/hwchase17/langchain/blob/ee70d4a0cd743fd9376ebc81d2948a85f83e621f/langchain/text_splitter.py#L508C7-L508C44)

## Sample

```py
from langchain.text_splitter import RecursiveCharacterTextSplitter


def text_split(texts):
    splitter = RecursiveCharacterTextSplitter(
        separators=None,
        chunk_size = 20,
        chunk_overlap = 0,
    )
    docs = splitter.create_documents(
        texts=texts,
    )
    return docs


if __name__ == "__main__":
    texts = [
        """
このくらい長いテキストを切ってみます。
さらにたくさんのテキストを入れます。
"""
    ]
    print(text_split(texts=texts))
```

Result:

```
[Document(page_content='このくらい長いテキストを切ってみます。', metadata={}), Document(page_content='さらにたくさんのテキストを入れます。', metadata={})]
```
