# LCEL (LangChain Expression Language)

## Why use LCEL?

> LCEL makes it easy to build complex chains from basic components, and supports out of the box functionality such as streaming, parallelism, and logging.

But it seems that not all cases are suitable for LCEL.

## Basics

The syntax of LCEL is implemented by `__or__` and `__call__` methods.


```
--8<-- "src/langchain/runnable_basics.py"
```

## Example

```py
chain1 = prompt1 | model | StrOutputParser()
chain2 = prompt2 | model | StrOutputParser()
```

1. `Generate an argument about: {input}` -> base_response
    ```py
    planner = (
        ChatPromptTemplate.from_template("Generate an argument about: {input}")
        | ChatOpenAI()
        | StrOutputParser()
        | {"base_response": RunnablePassthrough()}
    )
    ```
1. `List the pros or positive aspects of {base_response}`
    ```py
    arguments_for = (
        ChatPromptTemplate.from_template(
            "List the pros or positive aspects of {base_response}"
        )
        | ChatOpenAI()
        | StrOutputParser()
    )
    ```
1. `List the cons or negative aspects of {base_response}`
    ```py
    arguments_against = (
        ChatPromptTemplate.from_template(
            "List the cons or negative aspects of {base_response}"
        )
        | ChatOpenAI()
        | StrOutputParser()
    )
    ```
1. `Generate a final response given the critique`

    ```py
    final_responder = (
        ChatPromptTemplate.from_messages(
            [
                ("ai", "{original_response}"),
                ("human", "Pros:\n{results_1}\n\nCons:\n{results_2}"),
                ("system", "Generate a final response given the critique"),
            ]
        )
        | ChatOpenAI()
        | StrOutputParser()
    )
    ```
1. Combine

    ```py
    chain = (
        planner
        | {
            "results_1": arguments_for,
            "results_2": arguments_against,
            "original_response": itemgetter("base_response"),
        }
        | final_responder
    )
    ```

1. Run

    ```py
    chain.invoke({"input": "scrum"})
    ```

## Idea

人間の情報検索の無意識な思考を反映できるのではないか。

例.
1. どうやって設定するんだろう
1. Documentはどこだ？→人に聞くかも
1. Documentを読む -> ある程度できる
1. 問題に直面する -> 人に聞く

例.
1. 人に聞く
1. 回答をもらう
1. 自分の理解していないところをさらに聞く
1. 相手から回答が得られる
1. 問題があることに気づく
1. チケットにする

自分の作業　↔ 人とのやりとり
