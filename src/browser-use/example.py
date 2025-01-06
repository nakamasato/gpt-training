from langchain_openai import ChatOpenAI
from browser_use import Agent, BrowserConfig, Browser
import asyncio


# `playwright install` is required to run this example
# Basic configuration
config = BrowserConfig(headless=False, disable_security=True)


async def main():
    agent = Agent(
        browser=Browser(config=config),
        task="食べログで六本木の焼肉店4つ見つけて",
        llm=ChatOpenAI(model="gpt-4o-mini"),
    )
    result = await agent.run()
    print(result)


asyncio.run(main())
