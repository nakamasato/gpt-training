import streamlit as st
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables.config import RunnableConfig

from src.examples.agent_react_custom import CUSTOM_PROMPT, tools
from langchain.agents import AgentExecutor, create_react_agent
from langchain.memory import ConversationBufferMemory
from src.libs.llm import get_llm


def main(llm):
    st.title("Langchain Sample App")
    st.info(
        """こちらは掛け算のツールが使えるAgentとの対話を行うサンプルアプリです。
https://python.langchain.com/docs/integrations/callbacks/streamlit を参考にしています。
"""
    )

    prompt = PromptTemplate.from_template(template=CUSTOM_PROMPT)

    agent = create_react_agent(
        llm=llm,
        tools=[tools[0]],
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


if __name__ == "__main__":
    llm = get_llm()
    main(llm)
