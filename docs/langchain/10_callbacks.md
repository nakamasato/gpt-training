# [Callbacks](https://python.langchain.com/docs/modules/callbacks/how_to/custom_callbacks)

## Overview

LLMを呼ぶタイミングなどに処理を挟むことができる

1. `on_llm_start`
1. `on_llm_end`
1. `on_llm_new_token`
1. `on_llm_error`
1. `on_chain_start`
1. `on_chain_end`
1. `on_agent_action`
1. `on_tool_end`
1. `on_tool_error`
1. `on_agent_finish`

## Examples

1. [StdOutCallbackHandler](https://github.com/hwchase17/langchain/blob/6d15854cda60823d1c2a8efc395d9c074336bdca/langchain/callbacks/stdout.py#L9) <- `verbose=True` とするとセットされるCallbackHandler
1. [CostCalcCallbackHandler](09_token_count.md): CostStreaming用のCallback
1. [SimpleStreamlitCallbackHandler](../apps/README.md#streaming): StreamlitでチャットがStreamingで出るようにするためのCallback

## Run

```
poetry run pytest -k test_callback -s
```

```
platform darwin -- Python 3.9.9, pytest-7.4.0, pluggy-1.2.0
rootdir: /Users/m.naka/repos/nakamasato/gpt-poc
plugins: cov-4.1.0
collected 4 items / 3 deselected / 1 selected

tests/test_callbacks.py run_id=UUID('a49fbd93-2745-4b73-807b-7ae668fcc9fd')
prompts=['Tell me a joke']
serialized={'lc': 1, 'type': 'not_implemented', 'id': ['langchain', 'llms', 'fake', 'FakeListLLM']}
.
```

promptsをチェックしたりできる。
