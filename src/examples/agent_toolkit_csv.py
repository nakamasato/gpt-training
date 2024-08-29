# GoogleGenerativeAI
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_openai import ChatOpenAI

from langchain.agents.agent_types import AgentType

agent = create_csv_agent(
    ChatOpenAI(temperature=0, model="gpt-4o-mini", verbose=True),
    "data/titanic.csv",
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS,
)

print(agent.invoke({"input": "how many rows are there?"}))
