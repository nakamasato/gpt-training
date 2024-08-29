# import wikipedia


from langchain.agents import AgentExecutor, AgentType, initialize_agent
from langchain.agents.react.base import DocstoreExplorer
from langchain_community.docstore import Wikipedia
from langchain_core.tools import Tool
from langchain_openai import OpenAI

# wikipedia.set_lang("ja")
# build tools
docstore = DocstoreExplorer(Wikipedia())
tools = [
    Tool(
        name="Search",
        func=docstore.search,
        description="Search for a term in the docstore.",
    ),
    Tool(
        name="Lookup",
        func=docstore.lookup,
        description="Lookup a term in the docstore.",
    ),
]


def main(llm):
    # initialize ReAct agent
    react = initialize_agent(tools, llm, agent=AgentType.REACT_DOCSTORE, verbose=True)
    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=react.agent,
        tools=tools,
        #   max_iterations=2,
        verbose=True,
    )

    # perform question-answering
    question = "岸田総理が演説中に爆弾を投げ込まれたのはいつ?"
    agent_executor.invoke({"input": question})


if __name__ == "__main__":
    llm = OpenAI()
    main(llm=llm)
