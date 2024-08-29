from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain_core.tools import Tool


def multiplier(a, b):
    return a * b


def parsing_multiplier(string):
    a, b = string.split(",")
    return multiplier(int(a), int(b))


google = GoogleSearchAPIWrapper()


def top5_results(query):
    return google.results(query, 5)


TOOL_PARSE_MULTIPLIER = Tool(
    name="Multiplier",
    func=parsing_multiplier,
    description=(
        "useful for when you need to multiply two numbers together. "
        "The input to this tool should be a comma separated list of numbers of length two, representing the two numbers you want to multiply together. "
        "For example, `1,2` would be the input if you wanted to multiply 1 by 2."
    ),
)
TOOL_GOOGLE = Tool(
    name="google-search",
    description="Search Google for recent results.",
    func=top5_results,
)
