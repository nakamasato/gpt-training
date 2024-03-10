# DocumentsChain

Use given Documents as context for the LLM chain to answer a question.

## StuffDocumentsChain

StuffDocumentsChain just stuffs the documents into the prompt as context without modifying the contents of the documents.


[default prompt](https://github.com/langchain-ai/langchain/blob/4c47f39fcb539fdeff6dd6d9b1f483cd9a1af69b/libs/langchain/langchain/chains/question_answering/stuff_prompt.py#L10-L15)

```
Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Helpful Answer:
```

!!! notes

    Actual contents of `{context}` would be generated in the following way:

    1. The retrieved Documents are joined together with a separator after formatting with `format_document`.
    1. Return a Dict with the key `context` and the value as the joined Documents.

    ```py
    # Format each document according to the prompt
    doc_strings = [format_document(doc, self.document_prompt) for doc in docs]
    # Join the documents together to put them in the prompt.
    inputs = {
        k: v
        for k, v in kwargs.items()
        if k in self.llm_chain.prompt.input_variables
    }
    inputs[self.document_variable_name] = self.document_separator.join(doc_strings)
    ```

    Ref: [StuffDocumentsChain](https://api.python.langchain.com/en/latest/_modules/langchain/chains/combine_documents/stuff.html#create_stuff_documents_chain)


```py
document_prompt = PromptTemplate(
    input_variables=["page_content"],
    template="{page_content}",
)

combine_documents_chain = StuffDocumentsChain(
    llm_chain=llm_chain,
    document_variable_name="context",
    document_prompt=document_prompt,
    callbacks=None,
)
```

## RefineDocumentsChain

## MapReduceDocumentsChain

## Ref

1. https://nakamasato.medium.com/enhancing-langchains-retrievalqa-for-real-source-links-53713c7d802a
1. [Retrieval](retrieval.md)
