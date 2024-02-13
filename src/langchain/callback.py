from typing import Any, Dict, List, Optional
from uuid import UUID

from langchain_openai import ChatOpenAI

from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.stdout import StdOutCallbackHandler
from langchain.schema import HumanMessage


class MyCustomHandler(BaseCallbackHandler):
    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        print(f"{run_id=}\n{prompts=}\n{serialized=}")


# StdOutCallbackHandlerを明示的に設定してverbose=Trueの時と同じ結果になるか見たかったが
# 思い通りの結果になっていない
class PromptStdoutHandler(StdOutCallbackHandler):
    def __init__(self, color: Optional[str] = None) -> None:
        """Initialize callback handler."""
        self.color = color

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        for i, prompt in enumerate(prompts):
            print(f"\n\n\033[1m> {i}. Prompt:\n{prompt}\n-----\n\033[0m")  # noqa: #231


def main():
    # To enable streaming, we pass in `streaming=True` to the ChatModel constructor
    # Additionally, we pass in a list with our custom handler
    chat = ChatOpenAI(
        max_tokens=25, streaming=True, callbacks=[PromptStdoutHandler()]
    )  # TODO: 思うように動いていない

    print(chat([HumanMessage(content="Tell me a joke")]))


if __name__ == "__main__":
    main()
