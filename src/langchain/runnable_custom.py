from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from operator import itemgetter

df = pd.read_csv("data/titanic.csv")


def get_survived(df):
    return df[df["Survived"] == 1]


def simple_example():
    # retrieval = RunnableParallel({"df": df, "question": RunnablePassthrough()})
    # df -> user_id を渡してしぼる
    # input = RunnablePassthrough()
    model = ChatOpenAI()

    print(get_survived(df))
    prompt = ChatPromptTemplate.from_template("df {df} は何行ありますか")
    chain = {"df": itemgetter("df") | RunnableLambda(get_survived)} | prompt | model

    print(chain.invoke({"df": df}))


def example_runnable_parallel():
    # mapper
    runnable = RunnableParallel(
        passed=RunnablePassthrough(),  # そのまま渡す {"df": df}
        extra=RunnablePassthrough.assign(survived=lambda x: get_survived(x["df"]).shape[0]),  # {'extra': {"df": df, "survived": 342}}
        count=lambda x: {"survived": get_survived(x["df"]).shape[0], "total": x["df"].shape[0]},  # {'count': {'survived': 342, 'total': 891}}
    )
    print(runnable.invoke({"df": df}))


def example_runnable_lambda():
    TEXT_KEY = "text"

    def summarize(x):
        return x[TEXT_KEY].replace("text", "summary")

    texts = [{"id": 1, TEXT_KEY: "text1"}, {"id": 2, TEXT_KEY: "text2"}, {"id": 3, TEXT_KEY: "text3"}]

    summarizer = RunnableParallel(
        id=RunnableLambda(lambda x: x["id"]),  # id
        summary=RunnableLambda(summarize),  # summary
    )

    summaries = summarizer.batch(texts)
    print(summaries)

    prompt = ChatPromptTemplate.from_template("以下の文章のSummaryを3行で作成してください:\n{context}")
    llm = ChatOpenAI()
    final_summarizer = prompt | llm
    res = final_summarizer.invoke({"context": "\n".join([f"- {s['id']}: {s['summary']}" for s in summaries])})
    print(res)


def example_runnable_assign():
    # assign

    create_sql_query = RunnableLambda(lambda x: f"SELECT * FROM {x['table']} WHERE {x['column']} = {x['value']}")
    create_graph = RunnableLambda(lambda x: f"[{x['title']}] graph based on {x['sql']} and original data ({x['graph_table']} and {x['column']})")

    runnable = RunnablePassthrough.assign(sql=create_sql_query, graph_table=lambda x: x["table"]) | create_graph
    print(runnable.invoke({"table": "titanic", "column": "Survived", "value": 1, "title": "Graph Title"}))


def example_runnable_parallel():
    """Use batch result as input for the next invoke
          parallel ->
        /            \
        - parallel -> -> summarize
    """
    parallel = RunnableParallel(
        batch_result=RunnableLambda(lambda x: f"SELECT * FROM {x['table']} WHERE {x['column']} = {x['value']}"),
        batch_result2=RunnableLambda(lambda x: f"SELECT * FROM {x['table']} WHERE {x['column']} = {x['value']}"),
    )

    print(parallel.invoke({"table": "titanic", "column": "survived", "value": 1}))
    chain = parallel | RunnableLambda(lambda x: "\n".join([v for k, v in x.items()]))
    print(chain.invoke({"table": "titanic", "column": "survived", "value": 1}))
    print(chain.batch([{"table": "titanic", "column": "Survived", "value": 1}, {"table": "titanic", "column": "Survived", "value": 0}]))


# example_runnable_parallel()

chain = itemgetter("lst") | RunnableLambda(lambda x: {i: e for i, e in enumerate(x)})
print(chain.invoke({"lst": ["1", "2", "3"]}))
