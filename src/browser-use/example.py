from langchain_openai import ChatOpenAI
from browser_use import Agent, BrowserConfig, Browser
import asyncio


# `playwright install` is required to run this example
# Basic configuration
config = BrowserConfig(headless=True, disable_security=True)


async def main():
    agent = Agent(
        browser=Browser(config=config),
        task="練馬区の明日の天気教えて",
        llm=ChatOpenAI(model="gpt-4o-mini"),
    )
    result = await agent.run()
    print(result)


asyncio.run(main())
