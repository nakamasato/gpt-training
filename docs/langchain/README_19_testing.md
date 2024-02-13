# Testing


1. [FakeLLMList](https://github.com/langchain-ai/langchain/blob/ca7da8f7ef9bc7a613ff07279c4603cad5fd175a/libs/community/langchain_community/llms/fake.py#L14)
    ```py
    from langchain.llms.fake import FakeListLLM
    ```
1. [FakeChatModel](https://github.com/langchain-ai/langchain/blob/ca7da8f7ef9bc7a613ff07279c4603cad5fd175a/libs/langchain/tests/unit_tests/llms/fake_chat_model.py#L14)
    自分で定義して使う
    [tests/fake_chat_model.py](tests/fake_chat_model.py)を参照
1. [FakeEmbeddings](https://github.com/langchain-ai/langchain/blob/ca7da8f7ef9bc7a613ff07279c4603cad5fd175a/libs/community/langchain_community/embeddings/fake.py#L9)
    ```py
    from langchain.embeddings import FakeEmbeddings
    embeddings = FakeEmbeddings(size=1352)
    ```

    [doc](https://python.langchain.com/docs/integrations/text_embedding/fake)
