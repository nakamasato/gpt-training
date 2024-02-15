# Memory

[LangchainのMemory機能の覚え書き](https://qiita.com/ayoyo/items/07da43bbab6652d37421) <- 全部の種類のサマリーとしてはとてもまとまっていて素晴らしい

1. ConversationBufferMemory
    1. 会話の履歴を保持する
    1. 会話の履歴を全て保持するため、会話が長くなるとプロンプトが膨大になる。
1. ConversationBufferWindowMemory
    1. 直近k個の会話履歴のみ保持する
    1. 古い会話が削除されるため、プロンプトが膨大になる問題が解決する。
1. Entity Memory
    1. エンティティに関してのみメモリを保持する
1. Conversation Knowledge Graph Memory
    1. ナレッジグラフを作って会話についての情報を保持する
1. ConversationSummaryMemory
    1. 過去会話を要約して保持する
    1. 要約は英語で行われる
1. ConversationSummaryBufferMemory
    1. 過去会話の要約と、直近の会話（トークン数で指定）を保持する
1. ConversationTokenBufferMemory
1. ConversationBufferWindowMemoryのトークン版。直近のk会話ではなく、直近のnトークンを保持する。
1. VectorStore-Backed Memory
    1. 過去会話をベクターストアに保持し、関連する会話のみ探索して用いる
    1. 似ている会話が無い場合は、同じものを複数表示するなど挙動が怪しい点がある
1. `ReadOnlySharedMemory`: Read-onlyで追加はしない


## Usage

```py
from langchain.memory import ConversationBufferMemory
memory = ConversationBufferMemory(memory_key="history", return_messages=True)

agent_kwargs = {
    "extra_prompt_messages": [MessagesPlaceholder(variable_name="history")],
}
mrkl = initialize_agent(
    tools, llm, agent=AgentType.OPENAI_FUNCTIONS, agent_kwargs=agent_kwargs, memory=memory, verbose=True
)
```

これでAgentが呼ばれるたびに、`memory_key`に指定した名前のkeyに、会話の履歴が追加される。毎回LLMに渡すpromptには、`extra_prompt_messages`で指定した`MessagesPlaceholder`が追加される。

Promptが肥大化しやすいデメリットもあるので注意。


## Implementation

[Chain.\_\_call__](https://github.com/langchain-ai/langchain/blob/a2d30428237695f076060dec881bae0258123775/libs/langchain/langchain/chains/base.py#L252-L287)内で、

1. [prep_inputs](https://github.com/langchain-ai/langchain/blob/a2d30428237695f076060dec881bae0258123775/libs/langchain/langchain/chains/base.py#L416-L446)でmemoryからのmessageをpromptに注入している`self.memory.load_memory_variables(inputs)`

    ```py
    def prep_inputs(self, inputs: Union[Dict[str, Any], Any]) -> Dict[str, str]:
        """Validate and prepare chain inputs, including adding inputs from memory.

        Args:
            inputs: Dictionary of raw inputs, or single input if chain expects
                only one param. Should contain all inputs specified in
                `Chain.input_keys` except for inputs that will be set by the chain's
                memory.

        Returns:
            A dictionary of all inputs, including those added by the chain's memory.
        """
        if not isinstance(inputs, dict):
            _input_keys = set(self.input_keys)
            if self.memory is not None:
                # If there are multiple input keys, but some get set by memory so that
                # only one is not set, we can still figure out which key it is.
                _input_keys = _input_keys.difference(self.memory.memory_variables)
            if len(_input_keys) != 1:
                raise ValueError(
                    f"A single string input was passed in, but this chain expects "
                    f"multiple inputs ({_input_keys}). When a chain expects "
                    f"multiple inputs, please call it by passing in a dictionary, "
                    "eg `chain({'foo': 1, 'bar': 2})`"
                )
            inputs = {list(_input_keys)[0]: inputs}
        if self.memory is not None:
            external_context = self.memory.load_memory_variables(inputs)
            inputs = dict(inputs, **external_context)
        self._validate_inputs(inputs)
        return inputs
    ```

1. [prep_output](https://github.com/langchain-ai/langchain/blob/a2d30428237695f076060dec881bae0258123775/libs/langchain/langchain/chains/base.py#L390C1-L414)でメモリーに保存される

    ```py
    def prep_outputs(
        self,
        inputs: Dict[str, str],
        outputs: Dict[str, str],
        return_only_outputs: bool = False,
    ) -> Dict[str, str]:
        """Validate and prepare chain outputs, and save info about this run to memory.

        Args:
            inputs: Dictionary of chain inputs, including any inputs added by chain
                memory.
            outputs: Dictionary of initial chain outputs.
            return_only_outputs: Whether to only return the chain outputs. If False,
                inputs are also added to the final outputs.

        Returns:
            A dict of the final chain outputs.
        """
        self._validate_outputs(outputs)
        if self.memory is not None:
            self.memory.save_context(inputs, outputs)
        if return_only_outputs:
            return outputs
        else:
            return {**inputs, **outputs}
    ```

