import uuid
from typing import Any, Dict, List, Optional

import streamlit as st
from langchain_community.callbacks.streamlit.mutable_expander import MutableExpander
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, MessageGraph
from streamlit.delta_generator import DeltaGenerator

from src.libs.streamlitcallback import LLMResult, LLMThought, LLMThoughtLabeler, StreamlitCallbackHandler
from streamlit.runtime.scriptrunner.script_run_context import get_script_run_ctx


class ThreadSafeStreamlitCallbackHandler(BaseCallbackHandler):
    """Callback handler that writes to a Streamlit app."""

    def __init__(
        self,
        parent_container: DeltaGenerator,
        *,
        max_thought_containers: int = 4,
        expand_new_thoughts: bool = True,
        collapse_completed_thoughts: bool = True,
        thought_labeler: Optional[LLMThoughtLabeler] = None,
    ):
        """Create a StreamlitCallbackHandler instance.

        Parameters
        ----------
        parent_container
            The `st.container` that will contain all the Streamlit elements that the
            Handler creates.
        max_thought_containers
            The max number of completed LLM thought containers to show at once. When
            this threshold is reached, a new thought will cause the oldest thoughts to
            be collapsed into a "History" expander. Defaults to 4.
        expand_new_thoughts
            Each LLM "thought" gets its own `st.expander`. This param controls whether
            that expander is expanded by default. Defaults to True.
        collapse_completed_thoughts
            If True, LLM thought expanders will be collapsed when completed.
            Defaults to True.
        thought_labeler
            An optional custom LLMThoughtLabeler instance. If unspecified, the handler
            will use the default thought labeling logic. Defaults to None.
        """
        self._parent_container = parent_container
        self._history_parent = parent_container.container()
        self._history_container: Optional[MutableExpander] = None
        self._current_thought: Optional[LLMThought] = None
        self._completed_thoughts: List[LLMThought] = []
        self._max_thought_containers = max(max_thought_containers, 1)
        self._expand_new_thoughts = expand_new_thoughts
        self._collapse_completed_thoughts = collapse_completed_thoughts
        self._thought_labeler = thought_labeler or LLMThoughtLabeler()

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str]) -> None:
        """Run on LLM start."""
        ctx = get_script_run_ctx()
        print(f"on_llm_start {ctx}, {serialized}, {prompts}")

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Run on new LLM token. Only available when streaming is enabled."""
        ctx = get_script_run_ctx()
        print(f"on_llm_new_token {ctx}, {token}")

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: uuid.UUID,
        parent_run_id: Optional[uuid.UUID] = None,
        **kwargs: Any,
    ) -> Any:
        print(f"on_llm_end {response}, {run_id}, {parent_run_id}")
        # self._parent_container.write(response)


def main():
    _ = StreamlitCallbackHandler(st.container())

    model = ChatOpenAI(temperature=0)

    graph = MessageGraph()

    graph.add_node("oracle", model)
    graph.add_edge("oracle", END)

    graph.set_entry_point("oracle")

    runnable = graph.compile()

    user_msg = st.text_input("User message", value="日本の総理大臣がアメリカを訪れたのは何回ぐらいありますか？")

    if user_msg:
        with st.chat_message("user"):
            st.write(user_msg)

        with st.chat_message("agent"):
            st.write("Let me think about that...")
            handler = ThreadSafeStreamlitCallbackHandler(st.container())
            res = runnable.invoke(HumanMessage(user_msg), {"callbacks": [handler]})
            st.write(res)


if __name__ == "__main__":
    main()
