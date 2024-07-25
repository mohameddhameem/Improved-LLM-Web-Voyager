import os
import asyncio
import platform
import base64
import json
from typing import List, Optional, TypedDict

import nest_asyncio
from playwright.async_api import Page, async_playwright
from IPython import display
from openai import AsyncOpenAI

# Apply nest_asyncio for running async playwright in a Jupyter notebook
nest_asyncio.apply()

# Initialize the OpenAI client
client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


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
    bbox_descriptions: str
    function_args: dict


# Tool functions
async def click(state: AgentState):
    page = state["page"]
    bbox_id = state["function_args"]["bbox_id"]
    try:
        bbox = state["bboxes"][bbox_id]
    except Exception as e:
        print(f"Error accessing bounding box {bbox_id}: {e}")
        return f"Error: no bbox for : {bbox_id}"
    x, y = bbox["x"], bbox["y"]
    await page.mouse.click(x, y)
    print(f"Clicked on bounding box {bbox_id} at ({x}, {y})")
    return f"Clicked {bbox_id}"


async def type_text(state: AgentState):
    page = state["page"]
    bbox_id = state["function_args"]["bbox_id"]
    text_content = state["function_args"]["text"]
    try:
        bbox = state["bboxes"][bbox_id]
    except Exception as e:
        print(f"Error accessing bounding box {bbox_id}: {e}")
        return f"Error: no bbox for : {bbox_id}"
    x, y = bbox["x"], bbox["y"]
    await page.mouse.click(x, y)
    select_all = "Meta+A" if platform.system() == "Darwin" else "Control+A"
    await page.keyboard.press(select_all)
    await page.keyboard.press("Backspace")
    await page.keyboard.type(text_content)
    await page.keyboard.press("Enter")
    print(f"Typed '{text_content}' into bounding box {bbox_id} at ({x}, {y})")
    return f"Typed {text_content} and submitted"


async def scroll(state: AgentState):
    page = state["page"]
    target = state["function_args"]["target"]
    direction = state["function_args"]["direction"]
    if target.upper() == "WINDOW":
        scroll_amount = 500
        scroll_direction = (
            -scroll_amount if direction.lower() == "up" else scroll_amount
        )
        await page.evaluate(f"window.scrollBy(0, {scroll_direction})")
        print(f"Scrolled {direction} in the window")
        return f"Scrolled {direction} in window"
    else:
        try:
            target_id = int(target)
            if target_id < 0 or target_id >= len(state["bboxes"]):
                print(f"Error: Invalid bounding box index {target_id}")
                return f"Error: Invalid bounding box index {target_id}"
            bbox = state["bboxes"][target_id]
            scroll_amount = 200
            x, y = bbox["x"], bbox["y"]
            scroll_direction = (
                -scroll_amount if direction.lower() == "up" else scroll_amount
            )
            await page.mouse.move(x, y)
            await page.mouse.wheel(0, scroll_direction)
            print(f"Scrolled {direction} in element {target_id}")
            return f"Scrolled {direction} in element {target_id}"
        except ValueError:
            print(f"Error: Invalid target '{target}'. Expected 'WINDOW' or a number.")
            return f"Error: Invalid target '{target}'. Expected 'WINDOW' or a number."
        except Exception as e:
            print(f"Error while scrolling: {str(e)}")
            return f"Error while scrolling: {str(e)}"


async def wait(state: AgentState):
    sleep_time = 5
    await asyncio.sleep(sleep_time)
    print(f"Waited for {sleep_time} seconds")
    return f"Waited for {sleep_time}s."


async def go_back(state: AgentState):
    page = state["page"]
    await page.go_back()
    print(f"Navigated back to {page.url}")
    return f"Navigated back a page to {page.url}."


async def to_google(state: AgentState):
    page = state["page"]
    await page.goto("https://www.google.com/")
    print("Navigated to google.com")
    return "Navigated to google.com."


# Load JavaScript for page marking
with open("mark_page.js") as f:
    mark_page_script = f.read()


async def mark_page(page, max_retries=3):
    for attempt in range(max_retries):
        try:
            await page.evaluate(mark_page_script)
            for _ in range(10):
                try:
                    bboxes = await page.evaluate("markPage()")
                    screenshot = await page.screenshot()
                    await page.evaluate("unmarkPage()")
                    print("Page marked and bounding boxes retrieved")
                    return {
                        "img": base64.b64encode(screenshot).decode(),
                        "bboxes": bboxes,
                    }
                except Exception as e:
                    print(f"Error in markPage evaluation: {e}")
                    await asyncio.sleep(1)
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                print("Retrying...")
                await asyncio.sleep(2)
            else:
                print("Max retries reached. Returning empty state.")
                return {
                    "img": "",
                    "bboxes": [],
                }


async def annotate(state):
    marked_page = await mark_page(state["page"])
    print(f"Annotated page state: {marked_page}")
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
    print(f"Bounding box descriptions: {bbox_descriptions}")
    return {**state, "bbox_descriptions": bbox_descriptions}


# Define tools for OpenAI function calling
tools = [
    {
        "type": "function",
        "function": {
            "name": "click",
            "description": "Click on a specified element on the page",
            "parameters": {
                "type": "object",
                "properties": {
                    "bbox_id": {
                        "type": "integer",
                        "description": "The ID of the bounding box to click",
                    }
                },
                "required": ["bbox_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "type_text",
            "description": "Type text into a specified element on the page",
            "parameters": {
                "type": "object",
                "properties": {
                    "bbox_id": {
                        "type": "integer",
                        "description": "The ID of the bounding box to type into",
                    },
                    "text": {"type": "string", "description": "The text to type"},
                },
                "required": ["bbox_id", "text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "scroll",
            "description": "Scroll the page or a specific element",
            "parameters": {
                "type": "object",
                "properties": {
                    "target": {
                        "type": "string",
                        "description": "WINDOW or the ID of the bounding box to scroll",
                    },
                    "direction": {
                        "type": "string",
                        "enum": ["up", "down"],
                        "description": "The direction to scroll",
                    },
                },
                "required": ["target", "direction"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "wait",
            "description": "Wait for a few seconds",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "go_back",
            "description": "Navigate back to the previous page",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "to_google",
            "description": "Navigate to Google homepage",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
]


async def call_openai_with_tools(messages, tools):
    response = await client.chat.completions.create(
        model="gpt-4o-mini",  # or whichever model you're using
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )
    print(f"OpenAI response: {response}")
    return response


async def call_agent(question: str, page, max_steps: int = 150):
    messages = [{"role": "user", "content": question}]
    state = await annotate({"page": page})
    state = format_descriptions(state)
    if "bbox_descriptions" in state and state["bbox_descriptions"]:
        messages.append(
            {
                "role": "system",
                "content": f"Current page state: {state['bbox_descriptions']}",
            }
        )
    else:
        messages.append({"role": "system", "content": "No bounding boxes detected."})

    for step in range(max_steps):
        print(f"Step {step + 1}: Current question: {question}")
        response = await call_openai_with_tools(messages, tools)

        if (
            response.role == "assistant"
            and hasattr(response, "tool_calls")
            and response.tool_calls
        ):
            # Use the tool as suggested by the assistant
            tool_call = response.tool_calls[0]

            # Accessing attributes correctly
            function_name = tool_call.function.name
            arguments = tool_call.arguments  # Correct way to get arguments

            # Parse arguments if they are in string format
            function_args = (
                json.loads(arguments) if isinstance(arguments, str) else arguments
            )

            if function_name in globals() and callable(globals()[function_name]):
                # Call the function with the extracted parameters
                print(
                    f"Calling function '{function_name}' with arguments {function_args}"
                )
                result = await globals()[function_name](
                    {**state, "function_args": function_args}
                )
                messages.append({"role": "tool", "content": result})
            else:
                error_message = f"Error: tool {function_name} not found."
                print(error_message)
                messages.append({"role": "tool", "content": error_message})

        elif response.role == "assistant":
            # If the assistant returns a final answer
            print("Final answer from assistant:", response.content)
            return response.content

        # Prepare for the next iteration
        state = await annotate(state)
        state = format_descriptions(state)
        if "bbox_descriptions" in state and state["bbox_descriptions"]:
            messages.append(
                {
                    "role": "system",
                    "content": f"Updated page state: {state['bbox_descriptions']}",
                }
            )

    print("Reached maximum steps without a solution.")
    return "Reached maximum steps without a solution."


async def main():
    question = "What are the latest blog posts from langchain?"

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        page = await browser.new_page()

        print("Starting agent...")
        res = await call_agent(question, page)

        print(f"Final Result: {res}")

        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
