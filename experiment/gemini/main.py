import os

import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
# モデルはGemini Proを使用
llm = ChatGoogleGenerativeAI(
    model="gemini-pro", google_api_key=os.getenv("GOOGLE_API_KEY_GEMINI")
)

st.title("Gemini Chat")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("What is up?")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        response = llm.invoke(prompt)
        st.markdown(response.content)

    st.session_state.messages.append({"role": "assistant", "content": response.content})
