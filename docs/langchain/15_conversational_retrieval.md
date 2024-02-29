# Conversational Retrieval

## Overview

[Retrieval](https://python.langchain.com/docs/modules/data_connection/) uses following components:

1. Document loaders
1. Document transformers
1. Text embedding models
1. Vector stores
1. Retrievers


RetrievalChain は、`question_generator` (LLMChain) を使って与えられた`question`と`chat_history`から質問を生成して投げる。これによって過去のchat_historyの文脈を含んだ質問ができるようになる。一方で、質問のtoken上限に達してしまう可能性がある。

## Types

1. [BaseConversationalRetrievalChain](https://github.com/langchain-ai/langchain/blob/8c150ad7f6e9073f3508187ce79ea4715b74e105/libs/langchain/langchain/chains/conversational_retrieval/base.py#L59C7-L59C39): Chain for chatting with an index これを基本は継承する
1. [ConversationalRetrievalChain(BaseConversationalRetrievalChain)](https://github.com/langchain-ai/langchain/blob/8c150ad7f6e9073f3508187ce79ea4715b74e105/libs/langchain/langchain/chains/conversational_retrieval/base.py#L224): Chain for having a conversation based on retrieved documents.
    ```py
    This chain takes in chat history (a list of messages) and new questions,
    and then returns an answer to that question.
    The algorithm for this chain consists of three parts:
     1. Use the chat history and the new question to create a "standalone question".
    This is done so that this question can be passed into the retrieval step to fetch
    relevant documents. If only the new question was passed in, then relevant context
    may be lacking. If the whole conversation was passed into retrieval, there may
    be unnecessary information there that would distract from retrieval.
     2. This new question is passed to the retriever and relevant documents are
    returned.
     3. The retrieved documents are passed to an LLM along with either the new question
    (default behavior) or the original question and chat history to generate a final
    response.
    ```
1. [ChatVectorDBChain(BaseConversationalRetrievalChain)](https://github.com/langchain-ai/langchain/blob/8c150ad7f6e9073f3508187ce79ea4715b74e105/libs/langchain/langchain/chains/conversational_retrieval/base.py#L384): Chain for chatting with a vector database.


## Implementation

- [_call](https://github.com/langchain-ai/langchain/blob/8c150ad7f6e9073f3508187ce79ea4715b74e105/libs/langchain/langchain/chains/conversational_retrieval/base.py#L127-L168): これがメインで実行される関数

    ```py
    qa =
    ```

## Components

### 1. RetrievalQA

1. [RetrievalQA(BaseRetrievalQA)](https://github.com/langchain-ai/langchain/blob/44c8b159b9bfcb878e6898ed703399d5459a1168/libs/langchain/langchain/chains/retrieval_qa/base.py#L192)
    1. 初期化: [from_llm](https://github.com/langchain-ai/langchain/blob/44c8b159b9bfcb878e6898ed703399d5459a1168/libs/langchain/langchain/chains/retrieval_qa/base.py#L64)で、retrieverを渡して初期化する (RetrievalQAは、`retriever: BaseRetriever = Field(exclude=True)` を持っている。)
    1. Query: `qa(query)` で実行される。 [BaseRetrievalQA._call](https://github.com/langchain-ai/langchain/blob/44c8b159b9bfcb878e6898ed703399d5459a1168/libs/langchain/langchain/chains/retrieval_qa/base.py#L114)が呼ばれる
    1. [self._get_docs(question)](https://github.com/langchain-ai/langchain/blob/44c8b159b9bfcb878e6898ed703399d5459a1168/libs/langchain/langchain/chains/retrieval_qa/base.py#L209-L218)でDocumentを`retriever.get_relevant_documents`で取得する
        ```py
        def _get_docs(
            self,
            question: str,
            *,
            run_manager: CallbackManagerForChainRun,
        ) -> List[Document]:
            """Get docs."""
            return self.retriever.get_relevant_documents(
                question, callbacks=run_manager.get_child()
            )
        ```
    1. `_call`内で`_get_docs`の後にdocsを`combine_documents_chain`に渡して、DocsとQuestionで最終的な答えを出す。
        ```py
        answer = self.combine_documents_chain.run(
            input_documents=docs, question=question, callbacks=_run_manager.get_child()
        )
        ```
    1. [PromptTemplate](https://github.com/langchain-ai/langchain/blob/0ea837404a54217d4cbc6c861e675cdfeafd2048/libs/langchain/langchain/prompts/prompt.py#L16)
1. [ConversationalRetrievalChain](https://github.com/langchain-ai/langchain/blob/392cfbee24450e60a910fbf46a7e223eb4a0b541/libs/langchain/langchain/chains/conversational_retrieval/base.py#L224)
    1. step:
        1. standalone questionsをChat Historyから作成
        1. questionを使って関連DocumentをVectorStoreから取得する
        1. 取得したDocumentをLLMに(新しいquestionまたは元々のQuestionとChatHistoryと共に)渡して最終的な答えを作成する
    1. template: [CONDENSE_QUESTION_PROMPT](https://github.com/langchain-ai/langchain/blob/master/libs/langchain/langchain/chains/conversational_retrieval/prompts.py#L10)
        ```py
        _template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

        Chat History:
        {chat_history}
        Follow Up Input: {question}
        Standalone question:"""
        ```

### 2. Retriever

RetrievalQAはRetrieverを使ってDocumentを検索する。

1. VectorStoreは `as_retriever()`という関数で[VectorStoreRetriever](https://github.com/langchain-ai/langchain/blob/60d025b83be4d4f884c67819904383ccd89cff87/libs/langchain/langchain/schema/vectorstore.py#L615C7-L615C27)を取得できる
1. retrieverは、検索の種類が `similarity`, `similarity_score_threshold`, `mmr`の3種類がある (TODO: どういうときにどれを使うかの比較)

1. 検索時は、[RetrievalQA._get_docs](https://github.com/langchain-ai/langchain/blob/44c8b159b9bfcb878e6898ed703399d5459a1168/libs/langchain/langchain/chains/retrieval_qa/base.py#L209C1-L218C10)が`retriever.get_relevant_documents`を呼ぶ
1. [retriever._get_relevant_documents](https://github.com/langchain-ai/langchain/blob/60d025b83be4d4f884c67819904383ccd89cff87/libs/langchain/langchain/schema/vectorstore.py#L653C9-L671) が呼ばれる
    1. 内部では `self.vectorstore`の
        1. `similarity_search(query, **self.search_kwargs)`,
        1. `similarity_search_with_relevance_scores(query, **self.search_kwargs)`
        1. `max_marginal_relevance_search(query, **self.search_kwargs)`のいづれかが呼ばれる。

    ```py
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        if self.search_type == "similarity":
            docs = self.vectorstore.similarity_search(query, **self.search_kwargs)
        elif self.search_type == "similarity_score_threshold":
            docs_and_similarities = (
                self.vectorstore.similarity_search_with_relevance_scores(
                    query, **self.search_kwargs
                )
            )
            docs = [doc for doc, _ in docs_and_similarities]
        elif self.search_type == "mmr":
            docs = self.vectorstore.max_marginal_relevance_search(
                query, **self.search_kwargs
            )
        else:
            raise ValueError(f"search_type of {self.search_type} not allowed.")
        return docs
    ```

vectorstoreから取得する部分は、次のVectorstoreを参照。

### 2. Vectorstore

VectorstoreはLangChain内でラップされたDeepLakeを使っている。が、DeepLakeの機能をすべて使いたい場合には、中身を理解して適宜拡張する必要がある。
デフォルトでは、Scoreが返ってこないので、DeepLakeのVectorStoreを直接いじれるように変更する必要がある。

1. LangChain
    1. [DeepLake.similarity_search](https://github.com/langchain-ai/langchain/blob/60d025b83be4d4f884c67819904383ccd89cff87/libs/langchain/langchain/vectorstores/deeplake.py#L451C1-L516)
        ```py
        def similarity_search(
            self,
            query: str,
            k: int = 4,
            **kwargs: Any,
        ) -> List[Document]:
            return self._search(
                query=query,
                k=k,
                use_maximal_marginal_relevance=False,
                return_score=False,
                **kwargs,
            )
        ```

        [_search](https://github.com/langchain-ai/langchain/blob/60d025b83be4d4f884c67819904383ccd89cff87/libs/langchain/langchain/vectorstores/deeplake.py#L311)は、queryからembedding_functionを使ってembeddingを計算して self.vectorstore.searchを呼ぶ。
        ```py
        result = self.vectorstore.search(
            embedding=embedding,
            k=fetch_k if use_maximal_marginal_relevance else k,
            distance_metric=distance_metric,
            filter=filter,
            exec_option=exec_option,
            return_tensors=["embedding", "metadata", "text", "id"],
            deep_memory=deep_memory,
        )
        ```
    1. [VectorStore.similarity_search_with_relevance_scores](https://github.com/langchain-ai/langchain/blob/60d025b83be4d4f884c67819904383ccd89cff87/libs/langchain/langchain/schema/vectorstore.py#L284-L329)
        ```py
        def similarity_search_with_relevance_scores(
            self,
            query: str,
            k: int = 4,
            **kwargs: Any,
        ) -> List[Tuple[Document, float]]:
            docs_and_similarities = self._similarity_search_with_relevance_scores(
                query, k=k, **kwargs
            )
            ...
            return docs_and_similarities
        ```
        [_similarity_search_with_relevance_scores](https://github.com/langchain-ai/langchain/blob/60d025b83be4d4f884c67819904383ccd89cff87/libs/langchain/langchain/schema/vectorstore.py#L230)で実際の計算はするが、計算式はSubclassで`self._select_relevance_score_fn()`で変更することができる。

    1. [DeepLake.max_marginal_relevance_search](https://github.com/langchain-ai/langchain/blob/60d025b83be4d4f884c67819904383ccd89cff87/libs/langchain/langchain/vectorstores/deeplake.py#L716)
        ```py
        return self._search(
            query=query,
            k=k,
            fetch_k=fetch_k,
            use_maximal_marginal_relevance=True,
            lambda_mult=lambda_mult,
            exec_option=exec_option,
            embedding_function=embedding_function,  # type: ignore
            **kwargs,
        )
        ```

        [maximum_marginal_relevance](https://github.com/langchain-ai/langchain/blob/60d025b83be4d4f884c67819904383ccd89cff87/libs/langchain/langchain/vectorstores/utils.py#L23)で、fetch_k取得した分にたいして計算する。(TODO:詳細確認)

**Maximal Marginal Relevance**: *The idea behind using MMR is that it tries to reduce redundancy and increase diversity in the result and is used in text summarization.* ([Maximal Marginal Relevance to Re-rank results in Unsupervised KeyPhrase Extraction](https://medium.com/tech-that-works/maximal-marginal-relevance-to-rerank-results-in-unsupervised-keyphrase-extraction-22d95015c7c5))

1. DeepLake: [libs/langchain/langchain/vectorstores/deeplake.py](https://github.com/langchain-ai/langchain/blob/master/libs/langchain/langchain/vectorstores/deeplake.py)
    1. [Update dataset](https://docs.activeloop.ai/tutorials/deep-learning/updating-datasets)
    1. [Offcial Docs](https://docs.activeloop.ai)
    1. [Deep Lake Vector Store API](https://docs.activeloop.ai/tutorials/vector-store/vector-search-options/deep-lake-vector-store-api)

### 3. Datastore

1. **SQLite**: For permanent data (Chat History, document source data (Confluence, Directory, Text, etc.), imported document.)

### 4. Text Splitter

1. Check: https://langchain-text-splitter.streamlit.app/
1. [MarkdownHeaderTextSplitter](https://api.python.langchain.com/en/latest/text_splitter/langchain.text_splitter.MarkdownHeaderTextSplitter.html)
1. [RecursiveCharacterTextSplitter](https://api.python.langchain.com/en/latest/text_splitter/langchain.text_splitter.RecursiveCharacterTextSplitter.html)

### 5. Document Loader

1. [TextLoader](https://api.python.langchain.com/en/latest/document_loaders/langchain.document_loaders.text.TextLoader.html?highlight=textloader#langchain.document_loaders.text.TextLoader) for Directory import
1. [ConfluenceLoader](https://api.python.langchain.com/en/latest/_modules/langchain/document_loaders/confluence.html#ConfluenceLoader) for Confluence ([Source](https://github.com/langchain-ai/langchain/blob/8c150ad7f6e9073f3508187ce79ea4715b74e105/libs/langchain/langchain/document_loaders/confluence.py))
    1. https://developer.atlassian.com/server/confluence/confluence-server-rest-api/
    1. https://python.langchain.com/docs/integrations/document_loaders/confluence
1. [GoogleDriveLoader](https://python.langchain.com/docs/integrations/document_loaders/google_drive)
    1. [google.auth package](https://google-auth.readthedocs.io/en/master/reference/google.auth.html) added to langchain in [langchain#6035](https://github.com/langchain-ai/langchain/pull/6035)

## FAQ

1. `ConversationalRetrievalChain` vs. `ChatVectorDBChain`?
1. retrievalqaをagentのtoolとして使う時にsource_documentを表示する方法は？
    1. ToolのInputとOutputはStringである必要があるので以下のように sourceをまとめて返す関数でwrapする (`return_direct=True`)
        ```py
        def run_qa_chain(question):
            results = qa_chain({"question":question},return_only_outputs=True)
            return str(results)
        ```
    1. `_run` を拡張してSourceを含める (`return_direct=True`)
        ```py
        def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
            """Use the tool."""
            retrieval_chain = self.create_retrieval_chain()
            answer = retrieval_chain(query)['answer']
            sources = '\n'.join(retrieval_chain(query)['sources'].split(', '))

            return f'{answer}\nSources:\n{sources}'
        ```
## Ref

1. [RetrievalQA chain return source_documents when using it as a Tool for an Agent #5097](https://github.com/langchain-ai/langchain/discussions/5097)
1. [Vector store agent with sources #4187](https://github.com/langchain-ai/langchain/discussions/4187)
