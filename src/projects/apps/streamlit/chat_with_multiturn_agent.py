import streamlit as st
from langchain_community.callbacks import StreamlitCallbackHandler

from langchain.memory import ConversationBufferMemory, StreamlitChatMessageHistory
from langchain.prompts import MessagesPlaceholder
from langchain.schema import SystemMessage
from src.examples.agent_multiturn import MEMORY_KEY, get_multiturn_agent

SESSION_KEY_MESSAGE = MEMORY_KEY
SESSION_KEY_AGENT = "agent"
MESSAGE_TYPE_ASSISTANT = "assistant"
MESSAGE_TYPE_USER = "user"


def post_process():
    st.sidebar.subheader("Messages")
    st.sidebar.json(st.session_state.get(SESSION_KEY_MESSAGE))


def init_page():
    st.set_page_config(page_title="My Great ChatGPT", page_icon="🤗")
    st.header("My Great ChatGPT 🤗")
    st.sidebar.title("Options")


def select_model():
    st.sidebar.radio("Choose a model:", ("gpt-3.5-turbo", "gpt-4"), key="GPT_MODEL")

    # サイドバーにスライダーを追加し、temperatureを0から2までの範囲で選択可能にする
    # 初期値は0.0、刻み幅は0.1とする
    st.sidebar.slider(
        "Temperature:",
        min_value=0.0,
        max_value=2.0,
        value=0.0,
        step=0.01,
        key="GPT_TEMPERATURE",
    )


def init_messages():
    clear_button = st.sidebar.button("Clear Conversation", key="clear")
    if clear_button or SESSION_KEY_MESSAGE not in st.session_state:
        st.session_state[SESSION_KEY_MESSAGE] = [
            SystemMessage(
                content="""ようこそ！このチャットは https://www.ogis-ri.co.jp/otc/hiroba/technical/similar-document-search/part29.html
をチャット形式で実装したものです。
```
AgentDipatcherが以下のAgentをツールとして持ち、どのAgentが応答するかを決定します。
1. horoscope agent
2. parts_order agent
3. default agent

また、ConversationBufferMemoryを使っているので、過去の会話を参照することが可能で、各AgentはSharedReadOnlyMemoryを使って会話履歴を読み取ります。
```
"""
            )
        ]
        st.session_state[SESSION_KEY_AGENT] = {}
        st.session_state.costs = []


def process():
    # https://python.langchain.com/docs/integrations/memory/streamlit_chat_message_history
    msgs = StreamlitChatMessageHistory(key=SESSION_KEY_MESSAGE)
    memory = ConversationBufferMemory(
        memory_key=SESSION_KEY_MESSAGE,
        chat_memory=msgs,
        return_messages=True,
    )
    chat_history = MessagesPlaceholder(variable_name=SESSION_KEY_MESSAGE)
    for msg in msgs.messages:
        st.chat_message(msg.type).write(msg.content)

    agent = get_multiturn_agent(memory=memory, chat_history=chat_history, verbose=True)

    # ユーザーの入力を監視
    user_input = st.chat_input("ここに質問や回答を入力してください")
    if user_input:
        st.chat_message(MESSAGE_TYPE_USER).markdown(user_input)

        with st.chat_message(MESSAGE_TYPE_ASSISTANT):
            st_callback = StreamlitCallbackHandler(st.container())
            response = agent.run(input=user_input, callbacks=[st_callback])
            st.markdown(response)


def main():
    init_page()
    select_model()

    # チャット履歴の初期化
    init_messages()
    process()

    post_process()


if __name__ == "__main__":
    main()
