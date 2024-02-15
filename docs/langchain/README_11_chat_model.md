# [ChatModels](https://python.langchain.com/docs/modules/model_io/models/chat/)

Chat ModelsはText in, Text outではなく、MessageというInterfaceを使ってやり取りする。
Langchainでは、 `gpt-3.5-turbo`や`gpt-4`で`OpenAI()`を初期化する場合は内部的に`ChatOpenAI()`が初期化されていたが、直接`ChatOpenAI()`で[初期化するようにWarningが出ている](https://github.com/hwchase17/langchain/blob/560c4dfc98287da1bc0cfc1caebbe86d1e66a94d/langchain/llms/openai.py#L172-L178)

Messageの種類:
1. `AIMessage`: *A chat message representing information coming from the AI system.* 人が操作するものではなくAIから返ってくるメッセージ
1. `HumanMessage`: *A chat message representing information coming from a human interacting with the AI system.* 聞きたい内容をいれる e.g. `""`
1. `SystemMessage`: *A chat message representing information that should be instructions to the AI system.* Systemに何かを言い渡す。 e.g. `"You are a helpful assistant."`
1. `ChatMessage`: こいつは基本使わないかも。

Messageの使い方：**`SystemMessage`で、AIのスタンスを決め、`HumanMessage`と`AIMessage`の組み合わせを渡して、それまでのChatの履歴を模倣して、最終的に結果を出してもらう**

例:

```py
messages = [
    SystemMessage(content="あなたは親切なアシスタントです。"),
    HumanMessage(content="春の季語を絡めた冗談を教えてください。"),
    AIMessage(content="「春眠（しゅんみん）暁（ぎょう）を覚（さ）えず」という言葉がありますが、「春は眠くても、アシスタントは覚えてるよ！」と言って、ツッコミを入れるのはいかがでしょうか？笑"),
    HumanMessage(content="面白くない。もう一度。"),
]
```

Ref: [ChatMessages](https://docs.langchain.com/docs/components/schema/chat-messages)

使い方:

```py
from langchain_openai import ChatOpenAI

chat = ChatOpenAI()
```

```py
# 直接メッセージを入れる
chat([HumanMessage(content="Translate this sentence from English to French: I love programming.")])

# 複数メッセージ
messages = [
    SystemMessage(content="You are a helpful assistant that translates English to French."),
    HumanMessage(content="I love programming.")
]
chat(messages)

# generate
batch_messages = [
    [
        SystemMessage(content="You are a helpful assistant that translates English to French."),
        HumanMessage(content="I love programming.")
    ],
    [
        SystemMessage(content="You are a helpful assistant that translates English to French."),
        HumanMessage(content="I love artificial intelligence.")
    ],
]
result = chat.generate(batch_messages)
```
