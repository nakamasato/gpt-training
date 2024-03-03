import streamlit as st
from langchain.agents import AgentExecutor, create_react_agent
from langchain.memory import ConversationBufferMemory
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_community.callbacks.streamlit.mutable_expander import MutableExpander
from langchain_core.runnables.config import RunnableConfig
from langchain_core.messages import AIMessage, HumanMessage
from src.examples.agent_react_custom import CUSTOM_PROMPT, tools
from src.examples.langgraph_agent_supervisor import MemberAgentConfig, construct_graph, construct_supervisor, create_member_agents
from src.libs.llm import get_llm
from langchain_openai import ChatOpenAI
from src.libs.tools import TOOL_GOOGLE, TOOL_PARSE_MULTIPLIER
from langchain.prompts import PromptTemplate
from langchain_experimental.tools.python.tool import PythonAstREPLTool
import pandas as pd


def langgraph_example(llm):
    df = pd.read_csv("data/titanic.csv")
    agent_configs = [
        MemberAgentConfig(
            name="GoogleSearcher",
            tools=[TOOL_GOOGLE],
            system_prompt="You are web researcher. You can use Google to search for information.",
        ),
        MemberAgentConfig(
            name="Multiplier",
            tools=[TOOL_PARSE_MULTIPLIER],
            system_prompt="You are multiplier of two numbers.",
        ),
        MemberAgentConfig(
            name="titanic CSV analyst",
            tools=[PythonAstREPLTool(locals={"df": df})],
            system_prompt="You are a CSV Analyst. you have titanic.csv loaded in df. You can use df to analyze the data.",
        ),
    ]
    st.info(
        f"""こちらはtitancのデータについて質問できるサンプルアプリです。
以下のエージェントが利用可能です。

{'\n- '.join([f'{agent.name}: {agent.system_prompt}' for agent in agent_configs])}
"""
    )
    agents = create_member_agents(llm, agent_configs)

    supervisor_prompt = """You are a supervisor tasked with managing a conversation between the
    following workers:  {members}. Given the following user request,
    respond with the worker to act next. Each worker will perform a
    task and respond with their results and status. When getting the final answer or if you think the question cannot be answered with the members,
    respond with FINISH."""

    supervisor_chain = construct_supervisor(llm, list(agents.keys()), supervisor_prompt)
    graph = construct_graph(supervisor_chain, agents)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    messages = st.session_state.get("messages", [])
    with st.chat_message("assistant"):
        st.markdown("titanicのデータについて質問することができます。")
        st.dataframe(df.head())
    for message in messages:
        if isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)
        else:
            st.write(f"System message: {message.content}")

    prompt = st.chat_input("Titanicの生存率は?")
    if prompt:
        st.session_state.messages.append(HumanMessage(content=prompt))
        st.chat_message("user").write(prompt)

        with st.chat_message("assistant"):
            intermediate_container = st.container()
            for output in graph.stream({"messages": st.session_state.messages}):
                for key, value in output.items():
                    a = MutableExpander(intermediate_container, key, True)
                    a.markdown(f"{key}: {value}")
                    a.update(new_label=key, new_expanded=False)
                if output.get("__end__"):
                    answer = output["__end__"]["messages"][-1].content
                    st.markdown(answer)
                    st.session_state.messages.append(AIMessage(content=answer))


def agent_example(llm):
    st.info(
        """こちらは掛け算のツールが使えるAgentとの対話を行うサンプルアプリです。
https://python.langchain.com/docs/integrations/callbacks/streamlit を参考にしています。
"""
    )
    prompt = PromptTemplate.from_template(template=CUSTOM_PROMPT)

    agent = create_react_agent(
        llm=llm,
        tools=[TOOL_GOOGLE],
        prompt=prompt,
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
    )

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        handle_parsing_errors=False,
    )
    prompt = st.chat_input()
    if prompt:
        st.chat_message("user").write(prompt)
        with st.chat_message("assistant"):
            st_callback = StreamlitCallbackHandler(st.container())
            config = RunnableConfig(callbacks=[st_callback])
            response = agent_executor.invoke({"input": prompt}, config=config)
            st.write(response["output"])


def main():
    st.title("Langchain Sample App")
    st.sidebar.selectbox("Select an example", ["Agent Example", "Langgraph Example"], key="example")

    if st.session_state["example"] == "Agent Example":
        llm = get_llm()
        agent_example(llm)
    else:
        llm = ChatOpenAI(verbose=True, streaming=True)
        langgraph_example(llm)


if __name__ == "__main__":
    main()
