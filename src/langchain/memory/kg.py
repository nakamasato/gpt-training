# ConversationKGMemory
# https://python.langchain.com/docs/modules/memory/types/kg
from langchain.memory import ConversationKGMemory
from langchain_openai import OpenAI


llm = OpenAI(temperature=0)
memory = ConversationKGMemory(llm=llm)
memory.save_context({"input": "say hi to sam"}, {"output": "who is sam"})
memory.save_context({"input": "sam is a friend"}, {"output": "okay"})

memory.load_memory_variables({"input": "who is sam"})
