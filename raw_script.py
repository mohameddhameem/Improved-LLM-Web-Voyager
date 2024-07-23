# Import necessary modules
import os
import asyncio
import platform
import base64
from getpass import getpass
from typing import List, Optional, TypedDict
from langchain_core.messages import BaseMessage, SystemMessage
from playwright.async_api import Page, async_playwright
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda
from langgraph.graph import END, StateGraph
import re
import nest_asyncio

# Install dependencies
os.system('pip install -U --quiet langgraph langsmith langchain_openai playwright > /dev/null')

# Configure environment variables
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Web-Voyager"

def _getpass(env_var: str):
    if not os.environ.get(env_var):
        os.environ[env_var] = getpass(f"{env_var}=")

_getpass("LANGCHAIN_API_KEY")
_getpass("OPENAI_API_KEY")

# Apply nest_asyncio for running async Playwright in a Jupyter notebook
nest_asyncio.apply()

# Define TypedDicts for bounding boxes and agent state
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

# Define tools for the agent
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
    await page.mouse.click(bbox["x"], bbox["y"])
    return f"Clicked {bbox_id}"

async def type_text(state: AgentState):
    page = state["page"]
    type_args = state["prediction"]["args"]
    if type_args is None or len(type_args) != 2:
        return f"Failed to type in element from bounding box labeled as number {type_args}"
    bbox_id = int(type_args[0])
    bbox = state["bboxes"][bbox_id]
    await page.mouse.click(bbox["x"], bbox["y"])
    select_all = "Meta+A" if platform.system() == "Darwin" else "Control+A"
    await page.keyboard.press(select_all)
    await page.keyboard.press("Backspace")
    await page.keyboard.type(type_args[1])
    await page.keyboard.press("Enter")
    return f"Typed {type_args[1]} and submitted"

async def scroll(state: AgentState):
    page = state["page"]
    scroll_args = state["prediction"]["args"]
    if scroll_args is None or len(scroll_args) != 2:
        return "Failed to scroll due to incorrect arguments."

    target, direction = scroll_args
    scroll_amount = 500 if target.upper() == "WINDOW" else 200
    scroll_direction = -scroll_amount if direction.lower() == "up" else scroll_amount
    if target.upper() == "WINDOW":
        await page.evaluate(f"window.scrollBy(0, {scroll_direction})")
    else:
        bbox = state["bboxes"][int(target)]
        await page.mouse.move(bbox["x"], bbox["y"])
        await page.mouse.wheel(0, scroll_direction)

    return f"Scrolled {direction} in {'window' if target.upper() == 'WINDOW' else 'element'}"

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

# Define the graph state and runnable functions
with open("mark_page.js") as f:
    mark_page_script = f.read()

@RunnablePassthrough
async def mark_page(page):
    await page.evaluate(mark_page_script)
    for _ in range(10):
        try:
            bboxes = await page.evaluate("markPage()")
            break
        except Exception:
            asyncio.sleep(3)
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
        text = bbox.get("ariaLabel") or bbox["text"]
        labels.append(f'{i} (<{bbox.get("type")}/>): "{text}"')
    return {**state, "bbox_descriptions": "\nValid Bounding Boxes:\n" + "\n".join(labels)}

def parse(text: str) -> dict:
    action_prefix = "Action: "
    action_block = text.strip().split("\n")[-1]
    if not action_block.startswith(action_prefix):
        return {"action": "retry", "args": f"Could not parse LLM Output: {text}"}
    action_str = action_block[len(action_prefix):]
    split_output = action_str.split(" ", 1)
    action, action_input = (split_output[0], None) if len(split_output) == 1 else split_output
    action_input = [inp.strip().strip("[]") for inp in action_input.strip().split(";")] if action_input else None
    return {"action": action.strip(), "args": action_input}

prompt = hub.pull("wfh/web-voyager")

llm = ChatOpenAI(model="gpt-4-vision-preview", max_tokens=4096)
agent = annotate | RunnablePassthrough.assign(
    prediction=format_descriptions | prompt | llm | StrOutputParser() | parse
)

def update_scratchpad(state: AgentState):
    old = state.get("scratchpad", [])
    txt = old[0].content if old else "Previous action observations:\n"
    step = int(re.match(r"\d+", txt.rsplit("\n", 1)[-1]).group()) + 1 if old else 1
    txt += f"\n{step}. {state['observation']}"
    return {**state, "scratchpad": [SystemMessage(content=txt)]}

# Define the agent graph
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
    return END if action == "ANSWER" else "agent" if action == "retry" else action

graph_builder.add_conditional_edges("agent", select_tool)

graph = graph_builder.compile()

# Run the agent
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
    async for event in event_stream:
        if "agent" not in event:
            continue
        pred = event["agent"].get("prediction") or {}
        action = pred.get("action")
        action_input = pred.get("args")
        steps.append(f"{len(steps) + 1}. {action}: {action_input}")
        if "ANSWER" in action:
            final_answer = action_input[0]
            break
    return final_answer

async def main():
    browser = await async_playwright().start()
    browser = await browser.chromium.launch(headless=False)
    page = await browser.new_page()
    await page.goto("https://www.google.com")

    questions = [
        "Could you explain the WebVoyager paper (on arxiv)?",
        "Please explain the today's XKCD comic for me. Why is it funny?",
        "What are the latest blog posts from Google DeepMind?",
    ]

    for question in questions:
        final_answer = await call_agent(question, page)
        print(final_answer)

    await browser.close()

if __name__ == "__main__":
    asyncio.run(main())
