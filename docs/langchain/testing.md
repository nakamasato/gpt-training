# Testing

## FakeListLLM

```py
from langchain.llms.fake import FakeListLLM

responses = [
    '{"action": "google-search","action_input": "current Japan\'s prime minister"}',
    '{"action": "Final Answer", "action_input": "現在の日本の総理大臣は岸田文雄です。"}',
]
llm = FakeListLLM(responses=responses)
```
