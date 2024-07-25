import os
import asyncio
import platform
import base64
import re
from getpass import getpass
from typing import List, Optional, TypedDict

import nest_asyncio
from langchain import hub
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, chain as chain_decorator
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda
from langgraph.graph import END, StateGraph
from playwright.async_api import Page, async_playwright
from IPython import display

# Apply nest_asyncio for running async playwright in a Jupyter notebook
nest_asyncio.apply()

# Type definitions
class BBox(TypedDict):
    x: float
    y: float
    text: str
    type: str
    ariaLabel: str

class Prediction(TypedDict):
    action: str
    args: Optional[List[str]]

class AgentState(TypedDict):
    page: Page
    input: str
    img: str
    bboxes: List[BBox]
    prediction: Prediction
    scratchpad: List[BaseMessage]
    observation: str

# Tool functions
async def click(state: AgentState):
    page = state["page"]
    click_args = state["prediction"]["args"]
    if click_args is None or len(click_args) != 1:
        return f"Failed to click bounding box labeled as number {click_args}"
    bbox_id = int(click_args[0])
    try:
        bbox = state["bboxes"][bbox_id]
    except Exception:
        return f"Error: no bbox for : {bbox_id}"
    x, y = bbox["x"], bbox["y"]
    await page.mouse.click(x, y)
    return f"Clicked {bbox_id}"

async def type_text(state: AgentState):
    page = state["page"]
    type_args = state["prediction"]["args"]
    if type_args is None or len(type_args) != 2:
        return f"Failed to type in element from bounding box labeled as number {type_args}"
    bbox_id = int(type_args[0])
    bbox = state["bboxes"][bbox_id]
    x, y = bbox["x"], bbox["y"]
    text_content = type_args[1]
    await page.mouse.click(x, y)
    select_all = "Meta+A" if platform.system() == "Darwin" else "Control+A"
    await page.keyboard.press(select_all)
    await page.keyboard.press("Backspace")
    await page.keyboard.type(text_content)
    await page.keyboard.press("Enter")
    return f"Typed {text_content} and submitted"

async def scroll(state: AgentState):
    page = state["page"]
    scroll_args = state["prediction"]["args"]
    if scroll_args is None or len(scroll_args) != 2:
        return "Failed to scroll due to incorrect arguments."
    target, direction = scroll_args
    if target.upper() == "WINDOW":
        scroll_amount = 500
        scroll_direction = -scroll_amount if direction.lower() == "up" else scroll_amount
        await page.evaluate(f"window.scrollBy(0, {scroll_direction})")
        return f"Scrolled {direction} in window"
    else:
        try:
            target_id = int(target)
            if target_id < 0 or target_id >= len(state["bboxes"]):
                return f"Error: Invalid bounding box index {target_id}"
            bbox = state["bboxes"][target_id]
            scroll_amount = 200
            x, y = bbox["x"], bbox["y"]
            scroll_direction = -scroll_amount if direction.lower() == "up" else scroll_amount
            await page.mouse.move(x, y)
            await page.mouse.wheel(0, scroll_direction)
            return f"Scrolled {direction} in element {target_id}"
        except ValueError:
            return f"Error: Invalid target '{target}'. Expected 'WINDOW' or a number."
        except Exception as e:
            return f"Error while scrolling: {str(e)}"

async def wait(state: AgentState):
    sleep_time = 5
    await asyncio.sleep(sleep_time)
    return f"Waited for {sleep_time}s."

async def go_back(state: AgentState):
    page = state["page"]
    await page.go_back()
    return f"Navigated back a page to {page.url}."

async def to_google(state: AgentState):
    page = state["page"]
    await page.goto("https://www.google.com/")
    return "Navigated to google.com."

# Load JavaScript for page marking
with open("mark_page.js") as f:
    mark_page_script = f.read()

@chain_decorator
async def mark_page(page):
    await page.evaluate(mark_page_script)
    for _ in range(10):
        try:
            bboxes = await page.evaluate("markPage()")
            break
        except Exception:
            await asyncio.sleep(3)
    screenshot = await page.screenshot()
    await page.evaluate("unmarkPage()")
    return {
        "img": base64.b64encode(screenshot).decode(),
        "bboxes": bboxes,
    }

async def annotate(state):
    marked_page = await mark_page.with_retry().ainvoke(state["page"])
    return {**state, **marked_page}

def format_descriptions(state):
    labels = []
    for i, bbox in enumerate(state["bboxes"]):
        text = bbox.get("ariaLabel") or ""
        if not text.strip():
            text = bbox["text"]
        el_type = bbox.get("type")
        labels.append(f'{i} (<{el_type}/>): "{text}"')
    bbox_descriptions = "\nValid Bounding Boxes:\n" + "\n".join(labels)
    return {**state, "bbox_descriptions": bbox_descriptions}

def parse(text: str) -> dict:
    lines = text.strip().split("\n")
    for line in reversed(lines):
        if line.startswith("Action:"):
            action_str = line[len("Action:"):].strip()
            if action_str.startswith("ANSWER"):
                return {"action": "ANSWER", "args": [action_str[len("ANSWER"):].strip(";").strip()]}
            split_output = action_str.split(" ", 1)
            action = split_output[0]
            action_input = split_output[1] if len(split_output) > 1 else None
            if action_input is not None:
                action_input = [inp.strip().strip("[]") for inp in action_input.strip().split(";")]
            return {"action": action, "args": action_input}
    return {"action": "retry", "args": f"Could not parse LLM Output: {text}"}

# Set up LLM and agent
prompt = hub.pull("wfh/web-voyager")
llm = ChatOpenAI(model="gpt-4o-mini", max_tokens=4096)
agent = annotate | RunnablePassthrough.assign(
    prediction=format_descriptions | prompt | llm | StrOutputParser() | parse
)

def update_scratchpad(state: AgentState):
    old = state.get("scratchpad")
    if old:
        txt = old[0].content
        last_line = txt.rsplit("\n", 1)[-1]
        step = int(re.match(r"\d+", last_line).group()) + 1
    else:
        txt = "Previous action observations:\n"
        step = 1
    txt += f"\n{step}. {state['observation']}"
    return {**state, "scratchpad": [SystemMessage(content=txt)]}

# Set up graph
graph_builder = StateGraph(AgentState)
graph_builder.add_node("agent", agent)
graph_builder.set_entry_point("agent")
graph_builder.add_node("update_scratchpad", update_scratchpad)
graph_builder.add_edge("update_scratchpad", "agent")

tools = {
    "Click": click,
    "Type": type_text,
    "Scroll": scroll,
    "Wait": wait,
    "GoBack": go_back,
    "Google": to_google,
}

for node_name, tool in tools.items():
    graph_builder.add_node(
        node_name,
        RunnableLambda(tool) | (lambda observation: {"observation": observation}),
    )
    graph_builder.add_edge(node_name, "update_scratchpad")

def select_tool(state: AgentState):
    action = state["prediction"]["action"]
    if action == "ANSWER":
        return END
    if action == "retry":
        return "agent"
    return action

graph_builder.add_conditional_edges("agent", select_tool)
graph = graph_builder.compile()

async def call_agent(question: str, page, max_steps: int = 150):
    event_stream = graph.astream(
        {
            "page": page,
            "input": question,
            "scratchpad": [],
        },
        {
            "recursion_limit": max_steps,
        },
    )
    final_answer = None
    steps = []
    step_counter = 0

    try:
        async for event in event_stream:
            step_counter += 1
            if step_counter > max_steps:
                print("Reached maximum number of steps. Stopping.")
                break

            if "agent" not in event:
                continue

            pred = event["agent"].get("prediction") or {}
            action = pred.get("action")
            action_input = pred.get("args")

            display.clear_output(wait=False)
            steps.append(f"{len(steps) + 1}. {action}: {action_input}")
            print("\n".join(steps))
            display.display(display.Image(base64.b64decode(event["agent"]["img"])))

            if action == "ANSWER":
                final_answer = action_input[0] if action_input else "No detailed answer provided."
                break  # Exit the loop as soon as we get an answer

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        await event_stream.aclose()  # Ensure the stream is closed properly

    return final_answer

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        page = await browser.new_page()
        await page.goto("https://www.google.com")

        # Example usage
        questions = [
            # "Could you explain the WebVoyager paper (on arxiv)?",
            # "Please explain the today's XKCD comic for me. Why is it funny?",
            "What are the latest blog posts from langchain?",
            # "Could you check google maps to see when i should leave to get to SFO by 7 o'clock? starting from SF downtown."
        ]

        for question in questions:
            print(f"\nQuestion: {question}")
            res = await call_agent(question, page)
            print(f"Final response: {res}")

        # Close the browser
        await browser.close()

# Run the main function
if __name__ == "__main__":
    asyncio.run(main())