## [Token Count](https://python.langchain.com/docs/modules/model_io/models/llms/how_to/token_usage_tracking)

```py
from langchain_community.callbacks import get_openai_callback

with get_openai_callback() as cb:
    result = llm("Tell me a joke")
    print(cb)
```

```
Tokens Used: 6659
        Prompt Tokens: 6321
        Completion Tokens: 338
Successful Requests: 4
Total Cost (USD): $0.13318
```

↑ これはstreamingには対応していない (https://github.com/langchain-ai/langchain/issues/4583) ので、Streamingで使う場合には、

```py
from typing import Any, Dict, List

import tiktoken
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.openai_info import (
    MODEL_COST_PER_1K_TOKENS,
    get_openai_token_cost_for_model,
)
from langchain.schema import LLMResult


class Cost:
    def __init__(
        self, total_cost=0, total_tokens=0, prompt_tokens=0, completion_tokens=0
    ):
        self.total_cost = total_cost
        self.total_tokens = total_tokens
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens

    def __str__(self):
        return (
            f"Tokens Used: {self.total_tokens}\n"
            f"\tPrompt Tokens: {self.prompt_tokens}\n"
            f"\tCompletion Tokens: {self.completion_tokens}\n"
            f"Total Cost (USD): ${self.total_cost}"
        )

    def add_prompt_tokens(self, prompt_tokens):
        self.prompt_tokens += prompt_tokens
        self.total_tokens += prompt_tokens

    def add_completion_tokens(self, completion_tokens):
        self.completion_tokens += completion_tokens
        self.total_tokens += completion_tokens


class CostCalcCallbackHandler(BaseCallbackHandler):
    def __init__(self, model_name, cost, *args, **kwargs):
        self.model_name = model_name
        self.encoding = tiktoken.encoding_for_model(model_name)
        self.cost = cost
        super().__init__(*args, **kwargs)

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        for prompt in prompts:
            self.cost.add_prompt_tokens(len(self.encoding.encode(prompt)))

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        for generation_list in response.generations:
            for generation in generation_list:
                self.cost.add_completion_tokens(
                    len(self.encoding.encode(generation.text))
                )
        if self.model_name in MODEL_COST_PER_1K_TOKENS:
            prompt_cost = get_openai_token_cost_for_model(
                self.model_name, self.cost.prompt_tokens
            )
            completion_cost = get_openai_token_cost_for_model(
                self.model_name, self.cost.completion_tokens, is_completion=True
            )
            self.cost.total_cost = prompt_cost + completion_cost
```

を定義して、以下のように使うことができる。

```py
cost = Cost()
qa = ConversationalRetrievalChain.from_llm(
    ChatOpenAI(
        model_name=MODEL_NAME,
        streaming=True,
        callbacks=[handler, CostCalcCallbackHandler(MODEL_NAME, cost)],
    ),
    retriever=retriever,
    return_source_documents=True,
)

# アシスタントのメッセージを表示
result = qa(
    {"question": user_msg, "chat_history": st.session_state.chat_history}
)

print(result)
st.session_state[SESSION_KEY_FOR_COST].append(
    {"usage": "normal", "amount": cost.total_cost}
)
```
