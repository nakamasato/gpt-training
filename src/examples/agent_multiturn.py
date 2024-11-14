import argparse
import datetime
import json
from typing import Any, List, Set, Tuple, Union

import requests
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

import langchain
from langchain.agents import AgentExecutor, AgentType, BaseSingleActionAgent, initialize_agent
from langchain.chains.llm import LLMChain
from langchain.memory import ConversationBufferMemory, ReadOnlySharedMemory
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import BaseCallbackHandler, BaseCallbackManager
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.tools import Tool, tool

MEMORY_KEY = "chat_history"

HOROSCOPE_SYSTEM_PROMPT = """あなたは星占いの専門家です。
星占いをしてその結果を回答します。

ただし星占いには誕生日が必要です。
もし誕生日が分からない場合は、誕生日を予測や仮定をせずに「星占いをするので誕生日を教えてください。」と回答して下さい。

誕生日がわかる場合は、例えば"4月24日"であれば"04/24"の形式に変換した上で horoscope 関数を使って占いを行って下さい。
"""

MODEL = "gpt-4o-mini"


class HoroscopeInput(BaseModel):
    birthday: str = Field(description="'mm/dd'形式の誕生日です。例: 3月7日生まれの場合は '03/07' です。")


@tool("horoscope", return_direct=True, args_schema=HoroscopeInput)
def horoscope(birthday):
    """星占いで今日の運勢を占います。"""
    birthday = "02/28" if birthday == "02/29" else birthday
    yday = datetime.datetime.strptime(birthday, "%m/%d").timetuple().tm_yday
    sign_table = {
        20: "山羊座",
        50: "水瓶座",
        81: "魚座",
        111: "牡羊座",
        142: "牡牛座",
        174: "双子座",
        205: "蟹座",
        236: "獅子座",
        267: "乙女座",
        298: "天秤座",
        328: "蠍座",
        357: "射手座",
        999: "山羊座",
    }
    for k, v in sign_table.items():
        if yday < k:
            sign = v
            break

    t_delta = datetime.timedelta(hours=9)
    JST = datetime.timezone(t_delta, "JST")
    today = datetime.datetime.now(JST).strftime("%Y/%m/%d")
    url = f"http://api.jugemkey.jp/api/horoscope/free/{today}"
    response = requests.get(url)
    horoscope = json.loads(response.text)["horoscope"][today]
    horoscope = {h["sign"]: h for h in horoscope}
    horoscope[sign]
    content = f"""今日の{sign}の運勢は...
・{horoscope[sign]["content"]}
・ラッキーアイテム:{horoscope[sign]["item"]}
・ラッキーカラー:{horoscope[sign]["color"]}"""
    return content


def get_horoscope_agent(llm: BaseChatModel, memory, chat_history, verbose: bool = False) -> AgentExecutor:
    horoscope_tools = [horoscope]

    agent_kwargs = {
        "system_message": SystemMessage(content=HOROSCOPE_SYSTEM_PROMPT),
        "extra_prompt_messages": [chat_history],
    }
    horoscope_agent = initialize_agent(
        tools=horoscope_tools,
        llm=llm,
        agent=AgentType.OPENAI_FUNCTIONS,
        verbose=verbose,
        agent_kwargs=agent_kwargs,
        memory=memory,
    )
    return horoscope_agent


PARTS_ORDER_SYSTEM_PROMPT = """あなたはプラモデルの部品の個別注文を受け付ける担当者です。
部品を注文するには以下の注文情報が全て必要です。


注文情報
--------
* 注文される方のお名前 :
* 注文される方のお名前（カタカナ） :
* 部品送付先の郵便番号 :
* 部品送付先の住所 :
* 注文される方のお電話番号 :
* 注文される方のメールアドレス :
* 部品注文の対象となる商品の名称 :
* 部品注文の対象となる商品の番号 :
* 注文する部品とその個数 :


部品送付先の住所に関する注意
----------------------------
また、住所に関しては番地まで含めた正確な情報が必要です。
「東京都」や「大阪府豊中市」などあいまいな場合は番地まで必要であることを回答に含めて下さい。


あなたの取るべき行動
--------------------
* 注文情報に未知の項目がある場合は予測や仮定をせず、"***" に置き換えた上で、
  把握している注文情報を parts_order 関数に設定し confirmed = false で実行して下さい。
* あなたの「最終確認です。以下の内容で部品を注文しますが、よろしいですか?」の問いかけに対して、
  ユーザーから肯定的な返答が確認できた場合のみ parts_order 関数を confirmed = true で実行し部品の注文を行って下さい。
* ユーザーから部品の注文の手続きをやめる、キャンセルする意思を伝えられた場合のみ、
  parts_order 関数を canceled = true で実行し、あなたはそれまでの部品の注文に関する内容を全て忘れます。

parts_order 関数を実行する際の注文する部品とその個数の扱い
----------------------------------------------------------
また、parts_order 関数を実行する際、注文する部品とその個数は part_no_and_quantities に設定して下さい。

part_no_and_quantities は注文する部品とその個数の表現する dict の list です。
list の要素となる各 dict は key として 'part_no' と 'quantity' を持ちます。
'part_no' の value が部品名称の文字列、'quantity'の value が個数を意味する数字の文字列です。
以下は部品'J-26'を2個と部品'デカールC'を1枚注文する場合の part_no_and_quantities です。

\u0060\u0060\u0060
[{"part_no": "J-26", "quantity": "2"}, {"part_no": "デカールC", "quantity": "1"}]
\u0060\u0060\u0060

"""


class PartsOrderInput(BaseModel):
    name: str = Field(description="注文される方のお名前です。")
    kana: str = Field(description="注文される方のお名前（カタカナ）です。")
    post_code: str = Field(description="部品送付先の郵便番号です。")
    address: str = Field(description="部品送付先の住所です。")
    tel: str = Field(description="注文される方のお電話番号です。")
    email: str = Field(description="注文される方のメールアドレスです。")
    product_name: str = Field(description="部品注文の対象となる商品の名称です。例:'PG 1/24 ダンバイン'")
    product_no: str = Field(description="部品注文の対象となる商品の箱や説明書に記載されている6桁の数字の文字列です。")
    part_no_and_quantities: list[dict[str, str]] | None = Field(
        description=(
            "注文する部品とその個数の表現する dict の list です。\n"
            'dict は key "part_no"の value が部品名称の文字列、key "quantity"の value が個数を意味する整数です。\n'
            '例: 部品"J-26"を2個と部品"デカールC"を1枚注文する場合は、\n'
            "\n"
            '[{"part_no": "J-26", "quantity": 2}, {"part_no": "デカールC", "quantity": 1}]\n'
            "\n"
            "としてください。"
        )
    )
    confirmed: bool = Field(
        description=(
            "注文内容の最終確認状況です。最終確認が出来ている場合は True, そうでなければ False としてください。\n"
            "* confirmed が True の場合は部品の注文が行われます。 \n"
            "* confirmed が False の場合は注文内容の確認が行われます。"
        )
    )
    canceled: bool = Field(
        description=(
            "注文の手続きを継続する意思を示します。\n"
            "通常は False としますがユーザーに注文の手続きを継続しない意図がある場合は True としてください。\n"
            "* canceled が False の場合は部品の注文手続きを継続します。 \n"
            "* canceled が True の場合は注文手続きをキャンセルします。"
        )
    )


@tool("parts_order", return_direct=True, args_schema=PartsOrderInput)
def parts_order(
    name: str,
    kana: str,
    post_code: str,
    address: str,
    tel: str,
    email: str,
    product_name: str,
    product_no: str,
    part_no_and_quantities: list[dict[str, str]],
    confirmed: bool,
    canceled: bool,
) -> str:
    """プラモデルの部品を紛失、破損した場合に必要な部品を個別注文します。注文の内容確認にも使用します"""
    if canceled:
        return "わかりました。また部品の注文が必要になったらご相談ください。"

    def check_params(
        name,
        kana,
        post_code,
        address,
        tel,
        email,
        product_name,
        product_no,
        part_no_and_quantities,
    ):
        for arg in [
            name,
            kana,
            post_code,
            address,
            tel,
            email,
            product_name,
            product_no,
        ]:
            if arg is None or arg == "***" or arg == "":
                return False
        if not part_no_and_quantities:
            return False

        for p_and_q in part_no_and_quantities:
            part_no = p_and_q.get("part_no", "***")
            quantity = p_and_q.get("quantity", "***")
            if part_no == "***":
                return False
            if quantity == "***":
                return False
        return True

    has_required = check_params(
        name,
        kana,
        post_code,
        address,
        tel,
        email,
        product_name,
        product_no,
        part_no_and_quantities,
    )

    if part_no_and_quantities:
        part_no_and_quantities_str = "\n    ".join([f"{row.get('part_no','***')} x {row.get('quantity','***')}" for row in part_no_and_quantities])
    else:
        part_no_and_quantities_str = "    ***"

    # 注文情報のテンプレート
    order_template = f"""- お名前: {name}
- お名前(カナ): {kana}
- 郵便番号: {post_code}
- 住所: {address}
- 電話番号: {tel}
- メールアドレス: {email}
- 商品名: {product_name}
- 商品番号: {product_no}
- ご注文の部品: {part_no_and_quantities_str}"""

    # 追加情報要求のテンプレート
    request_information_template = f'ご注文には以下の情報が必要です。"***" の項目を教えてください。\n\n{order_template}'

    # 注文確認のテンプレート
    confirm_template = f"最終確認です。以下の内容で部品を注文しますが、よろしいですか?\n\n{order_template}"

    # 注文完了のテンプレート
    complete_template = f"以下の内容で部品を注文しました。\n\n{order_template}\n\n2営業日以内にご指定のメールアドレスに注文確認メールが届かない場合は、\n弊社HPからお問い合わせください。"

    if has_required and confirmed:
        # TODO invoke order here!
        return complete_template
    else:
        if has_required:
            return confirm_template
        else:
            return request_information_template


PARTS_ORDER_SSUFFIX_PROMPT = """

重要な注意事項
--------------
注文情報に未知の項目がある場合は予測や仮定をせず "***" に置き換えてください。

parts_order 関数はユーザーから部品の注文の手続きをやめる、キャンセルする意思を伝えられた場合のみ canceled = true で実行して、
それまでの部品の注文に関する内容を全て忘れてください。。

parts_order 関数は次に示す例外を除いて confirmed = false で実行してください。

あなたの「最終確認です。以下の内容で部品を注文しますが、よろしいですか?」の問いかけに対して、
ユーザーから肯定的な返答が確認できた場合のみ parts_order 関数を confirmed = true で実行して部品を注文してください。

最終確認に対するユーザーの肯定的な返答なしで parts-order 関数を confirmed = true で実行することは誤発注であり事故になるので、固く禁止します。
"""


def get_parts_order_agent(memory, chat_history, verbose: bool = False) -> AgentExecutor:
    parts_order_tools = [parts_order]

    gpt35_po = ChatOpenAI(
        temperature=0,
        model=MODEL,
        model_kwargs={"top_p": 0.1, "function_call": {"name": "parts_order"}},
    )

    print(gpt35_po._default_params)

    agent_kwargs = {
        "system_message": SystemMessage(content=PARTS_ORDER_SYSTEM_PROMPT),
        "extra_prompt_messages": [chat_history],
    }
    parts_order_agent = initialize_agent(
        parts_order_tools,
        gpt35_po,
        agent=AgentType.OPENAI_FUNCTIONS,
        verbose=verbose,
        agent_kwargs=agent_kwargs,
        memory=memory,
    )

    messages = []
    messages.extend(parts_order_agent.agent.prompt.messages[:3])
    messages.append(SystemMessage(content=PARTS_ORDER_SSUFFIX_PROMPT))
    messages.append(parts_order_agent.agent.prompt.messages[3])
    parts_order_agent.agent.prompt.messages = messages

    return parts_order_agent


DEFAULT_SYSTEM_PROMPT = """あなたはAIのアシスタントです。
ユーザーの質問に答えたり、議論したり、日常会話を楽しんだりします。
"""


def get_multiturn_agent(memory, chat_history, verbose: bool = False) -> AgentExecutor:
    """multiturn Agent"""
    gpt35 = ChatOpenAI(temperature=0, model=MODEL, model_kwargs={"top_p": 0.1})

    gpt35_po = ChatOpenAI(
        temperature=0,
        model=MODEL,
        model_kwargs={"top_p": 0.1, "function_call": {"name": "parts_order"}},
    )
    readonly_memory = ReadOnlySharedMemory(memory=memory)

    agent_kwargs = {
        "system_message": SystemMessage(content=HOROSCOPE_SYSTEM_PROMPT),
        "extra_prompt_messages": [chat_history],
    }
    horoscope_agent = initialize_agent(
        tools=[horoscope],
        llm=gpt35,
        agent=AgentType.OPENAI_FUNCTIONS,
        verbose=verbose,
        agent_kwargs=agent_kwargs,
        memory=readonly_memory,  # ★
    )

    agent_kwargs = {
        "system_message": SystemMessage(content=PARTS_ORDER_SYSTEM_PROMPT),
        "extra_prompt_messages": [chat_history],
    }
    parts_order_agent = initialize_agent(
        tools=[parts_order],
        llm=gpt35_po,
        agent=AgentType.OPENAI_FUNCTIONS,
        verbose=verbose,
        agent_kwargs=agent_kwargs,
        memory=readonly_memory,  # ★
    )

    messages = []
    messages.extend(parts_order_agent.agent.prompt.messages[:3])
    messages.append(SystemMessage(content=PARTS_ORDER_SSUFFIX_PROMPT))
    messages.append(parts_order_agent.agent.prompt.messages[3])
    parts_order_agent.agent.prompt.messages = messages

    chat_prompt_template = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=DEFAULT_SYSTEM_PROMPT),
            chat_history,
            HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=["input"], template="{input}")),
        ]
    )

    default_chain = LLMChain(llm=gpt35, prompt=chat_prompt_template, memory=readonly_memory, verbose=verbose)

    tools = [
        Tool.from_function(
            func=horoscope_agent.run,
            name="horoscope_agent",
            description="星占いの担当者です。星占いに関係する会話の対応はこの担当者に任せるべきです。",
            args_schema=HoroscopeAgentInput,
            return_direct=True,
        ),
        Tool.from_function(
            func=parts_order_agent.run,
            name="parts_order_agent",
            description="プラモデルの部品の個別注文の担当者です。プラモデルの部品注文やキャンセルに関係する会話の対応はこの担当者に任せるべきです。",
            args_schema=PartsOrderAgentInput,
            return_direct=True,
        ),
        Tool.from_function(
            func=default_chain.run,
            name="DEFAULT",
            description="一般的な会話の担当者です。一般的で特定の専門家に任せるべきでない会話の対応はこの担当者に任せるべきです。",
            args_schema=DefaultAgentInput,
            return_direct=True,
        ),
    ]

    dispatcher_agent = DispatcherAgent(chat_model=gpt35, readonly_memory=readonly_memory, tools=tools, verbose=verbose)

    agent = AgentExecutor.from_agent_and_tools(agent=dispatcher_agent, tools=tools, memory=memory, verbose=verbose)

    return agent


ROUTER_TEMPLATE = """あなたの仕事はユーザーとあなたとの会話内容を読み、
以下の選択候補からその説明を参考にしてユーザーの対応を任せるのに最も適した候補を選び、その名前を回答することです。
あなたが直接ユーザーへ回答してはいけません。あなたは対応を任せる候補を選ぶだけです。

<< 選択候補 >>
名前: 説明
{destinations}

<< 出力形式の指定 >>
選択した候補の名前のみを出力して下さい。
注意事項: 出力するのは必ず選択候補として示された候補の名前の一つでなければなりません。
ただし全ての選択候補が不適切であると判断した場合には "DEFAULT" とすることができます。

<< 回答例 >>
「あなたについて教えて下さい。」と言われても返事をしてはいけません。
選択候補に適切な候補がないケースですから"DEFAULT"と答えて下さい。

"""

ROUTER_PROMPT_SUFFIX = """<< 出力形式の指定 >>
最後にもう一度指示します。選択した候補の名前のみを出力して下さい。
注意事項: 出力は必ず選択候補として示された候補の名前の一つでなければなりません。
ただし全ての選択候補が不適切であると判断した場合には "DEFAULT" とすることができます。
"""


class DestinationOutputParser(BaseOutputParser[str]):
    destinations: Set[str]

    class Config:
        extra = "allow"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.destinations_and_default = list(self.destinations) + ["DEFAULT"]

    def parse(self, text: str) -> str:
        matched = [int(d in text) for d in self.destinations_and_default]
        if sum(matched) != 1:
            raise OutputParserException(f"DestinationOutputParser expected output value includes " f"one(and only one) of {self.destinations_and_default}. " f"Received {text}.")

        return self.destinations_and_default[matched.index(1)]

    @property
    def _type(self) -> str:
        return "destination_output_parser"


class DispatcherAgent(BaseSingleActionAgent):
    chat_model: BaseChatModel
    readonly_memory: ReadOnlySharedMemory
    tools: List[Tool]
    verbose: bool = False

    class Config:
        extra = "allow"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        destinations = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        router_template = ROUTER_TEMPLATE.format(destinations=destinations)
        router_prompt_template = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=router_template),
                MessagesPlaceholder(variable_name=MEMORY_KEY),
                HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=["input"], template="{input}")),
                SystemMessage(content=ROUTER_PROMPT_SUFFIX),
            ]
        )
        self.router_chain = LLMChain(
            llm=self.chat_model,
            prompt=router_prompt_template,
            memory=self.readonly_memory,
            verbose=self.verbose,
        )

        self.route_parser = DestinationOutputParser(destinations=set([tool.name for tool in self.tools]))

    @property
    def input_keys(self):
        return ["input"]

    def plan(
        self,
        intermediate_steps: List[Tuple[AgentAction, str]],
        callbacks: list[BaseCallbackHandler] | BaseCallbackManager | None = ...,
        **kwargs: Any,
    ) -> Union[AgentAction, AgentFinish]:
        router_output = self.router_chain.run(kwargs["input"])
        try:
            destination = self.route_parser.parse(router_output)
        except OutputParserException:
            destination = "DEFAULT"

        return AgentAction(tool=destination, tool_input=kwargs["input"], log="")

    async def aplan(
        self,
        intermediate_steps: List[Tuple[AgentAction, str]],
        callbacks: list[BaseCallbackHandler] | BaseCallbackManager | None = ...,
        **kwargs: Any,
    ) -> Union[AgentAction, AgentFinish]:
        router_output = await self.router_chain.arun(kwargs["input"])
        try:
            destination = self.route_parser.parse(router_output)
        except OutputParserException:
            destination = "DEFAULT"

        return AgentAction(tool=destination, tool_input=kwargs["input"], log="")


class HoroscopeAgentInput(BaseModel):
    user_utterance: str = Field(description="星占いの専門家に伝達するユーザーの直近の発話内容です。")


class PartsOrderAgentInput(BaseModel):
    user_utterance: str = Field(description="プラモデルの部品の個別注文の担当者に伝達するユーザーの直近の発話内容です。")


class DefaultAgentInput(BaseModel):
    user_utterance: str = Field(description="一般的な内容を担当する担当者に伝達するユーザーの直近の発話内容です。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--type", choices=["one", "two", "multiturn"], default="one")
    parser.add_argument("-v", "--verbose", action="store_true", default=False)
    args = parser.parse_args()

    langchain.debug = args.verbose
    llm = ChatOpenAI(
        temperature=0,
        model=MODEL,
        model_kwargs={"top_p": 0.1},
        verbose=args.verbose,
    )
    memory = ConversationBufferMemory(memory_key=MEMORY_KEY, return_messages=True)
    chat_history = MessagesPlaceholder(variable_name=MEMORY_KEY)
    if args.type == "one":
        agent = get_horoscope_agent(llm=llm, memory=memory, chat_history=chat_history, verbose=args.verbose)
    elif args.type == "two":
        agent = get_parts_order_agent(memory=memory, chat_history=chat_history, verbose=args.verbose)
    elif args.type == "multiturn":
        agent = get_multiturn_agent(memory=memory, chat_history=chat_history, verbose=args.verbose)

    # 会話ループ
    user = ""
    while user != "exit":
        user = input("入力してください:")
        print(user)
        ai = agent.run(input=user)
        print(ai)
