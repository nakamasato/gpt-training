from langchain.agents import AgentExecutor
from langchain.llms.fake import FakeListLLM
from src.langchain.react_custom import ReActTestAgent, tools


def test_main():
    responses = [
        "思考: I need to get invoice amount of company C.\n行動: GetInvoice[C]",
        "思考: I need to get invoice amount of company F.\n行動: GetInvoice[F]",
        "思考: I need to get invoice amount of company A.\n行動: GetInvoice[A]",
        "思考: I need to get invoice amount of company E.\n行動: GetInvoice[E]",
        "思考: I need to get total amount of obtained amount C and F.\n行動: Toal[20000 4100]",
        "思考: I need to get total amount of obtained amount A and E.\n行動: Toal[2000 1000]",
        "思考: I need to get difference of obtained amount C, F and the total of A, E.\n行動: Diff[24100 3000]",
        "思考: So the answer is 21100.\n行動: 完了[21100]",
    ]
    llm = FakeListLLM(responses=responses)
    agent = ReActTestAgent.from_llm_and_tools(
        llm,
        tools,
    )
    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=True,
    )

    question = "How much is the difference between the total of company C, F and the total of company A, E ?"
    result = agent_executor.run(question)
    assert result == "21100"
