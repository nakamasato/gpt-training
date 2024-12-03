from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda, RunnableBranch
import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from operator import itemgetter
from random import randint

df = pd.read_csv("data/titanic.csv")
print(df)


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


def example_runnable_branch():
    # branch
    extract_message_chain = RunnableLambda(lambda x: {"message": ["message"] * randint(0, 2)})  # randomize message
    message_summary_chain = RunnableLambda(lambda x: f"this is the summary of {len(x['message'])} messages")  # emulate chain with LLM

    # chain
    chain_with_branch = extract_message_chain | RunnableBranch(
        (lambda x: bool(x["message"]), message_summary_chain),
        lambda x: "no message found",  # fixed result when empty
    )
    print(chain_with_branch.batch([{}, {}, {}]))


example_runnable_branch()
