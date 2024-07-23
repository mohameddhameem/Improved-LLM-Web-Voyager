import asyncio
from playwright.async_api import async_playwright
from agent_types import AgentState
from graph import create_graph
from utils import call_agent

async def main():
    browser = await async_playwright().start()
    browser = await browser.chromium.launch(headless=False, args=None)
    page = await browser.new_page()
    await page.goto("https://www.google.com")

    graph = create_graph()

    questions = [
        #"Could you explain the WebVoyager paper (on arxiv)?",
        # "Please explain the today's XKCD comic for me. Why is it funny?",
        # "What are the latest blog posts from langchain?",
        # "Could you check google maps to see when i should leave to get to SFO by 7 o'clock? starting from SF downtown."
        "Find me distance between Singapore to Malaysia"
    ]

    for question in questions:
        res = await call_agent(question, page, graph)
        print(f"Final response: {res}")

    await browser.close()

if __name__ == "__main__":
    asyncio.run(main())