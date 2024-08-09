import os
import asyncio
import platform
import base64
import re
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
    except Exception:
        return f"Error: no bbox for : {bbox_id}"
    x, y = bbox["x"], bbox["y"]
    await page.mouse.click(x, y)
    return f"Clicked {bbox_id}"

async def type_text(state: AgentState):
    page = state["page"]
    bbox_id = state["function_args"]["bbox_id"]
    text_content = state["function_args"]["text"]
    try:
        bbox = state["bboxes"][bbox_id]
    except Exception:
        return f"Error: no bbox for : {bbox_id}"
    x, y = bbox["x"], bbox["y"]
    await page.mouse.click(x, y)
    select_all = "Meta+A" if platform.system() == "Darwin" else "Control+A"
    await page.keyboard.press(select_all)
    await page.keyboard.press("Backspace")
    await page.keyboard.type(text_content)
    await page.keyboard.press("Enter")
    return f"Typed {text_content} and submitted"

async def scroll(state: AgentState):
    page = state["page"]
    target = state["function_args"]["target"]
    direction = state["function_args"]["direction"]
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

async def mark_page(page, max_retries=3):
    for attempt in range(max_retries):
        try:
            await page.evaluate(mark_page_script)
            for _ in range(10):
                try:
                    bboxes = await page.evaluate("markPage()")
                    screenshot = await page.screenshot()
                    await page.evaluate("unmarkPage()")
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
                        "description": "The ID of the bounding box to click"
                    }
                },
                "required": ["bbox_id"]
            }
        }
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
                        "description": "The ID of the bounding box to type into"
                    },
                    "text": {
                        "type": "string",
                        "description": "The text to type"
                    }
                },
                "required": ["bbox_id", "text"]
            }
        }
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
                        "description": "WINDOW or the ID of the bounding box to scroll"
                    },
                    "direction": {
                        "type": "string",
                        "enum": ["up", "down"],
                        "description": "The direction to scroll"
                    }
                },
                "required": ["target", "direction"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "wait",
            "description": "Wait for a few seconds",
            "parameters": {
                "type": "object",
                "properties": {},
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "go_back",
            "description": "Navigate back to the previous page",
            "parameters": {
                "type": "object",
                "properties": {},
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "to_google",
            "description": "Navigate to Google homepage",
            "parameters": {
                "type": "object",
                "properties": {},
            }
        }
    }
]

async def call_openai_with_tools(messages, tools):
    response = await client.chat.completions.create(
        model="gpt-4o-mini",  # or whichever model you're using
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )
    return response.choices[0].message

async def call_agent(question: str, page, max_steps: int = 150):
    messages = [{"role": "user", "content": question}]
    state = await annotate({"page": page})
    state = format_descriptions(state)
    if 'bbox_descriptions' in state and state['bbox_descriptions']:
        messages.append({"role": "system", "content": f"Current page state: {state['bbox_descriptions']}"})
    else:
        messages.append({"role": "system", "content": "Current page state could not be determined. Please navigate to a relevant page."})

    for step in range(max_steps):
        response = await call_openai_with_tools(messages, tools)
        
        # Check if the response is a final answer
        if not response.tool_calls:
            print(f"Final response: {response.content}")
            return response.content

        # If it's not a final answer, continue with tool calls
        messages.append({"role": "assistant", "content": response.content, "tool_calls": response.tool_calls})
        
        for tool_call in response.tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            
            # Update the state with the current function args
            state["function_args"] = function_args
            
            # Call the appropriate tool function
            if function_name == "click":
                result = await click(state)
            elif function_name == "type_text":
                result = await type_text(state)
            elif function_name == "scroll":
                result = await scroll(state)
            elif function_name == "wait":
                result = await wait(state)
            elif function_name == "go_back":
                result = await go_back(state)
            elif function_name == "to_google":
                result = await to_google(state)
            else:
                result = f"Unknown function: {function_name}"
            
            messages.append({"role": "tool", "tool_call_id": tool_call.id, "name": function_name, "content": result})

        # Update the page state
        state = await annotate({"page": page})
        state = format_descriptions(state)
        if 'bbox_descriptions' in state and state['bbox_descriptions']:
            messages.append({"role": "system", "content": f"Current page state: {state['bbox_descriptions']}"})
        else:
            messages.append({"role": "system", "content": "Current page state could not be determined. Please navigate to a relevant page."})

        # Display the current state (optional)
        display.clear_output(wait=False)
        print(f"Step {step + 1}: {function_name if response.tool_calls else 'No function called'}")
        if 'img' in state and state['img']:
            display.display(display.Image(base64.b64decode(state['img'])))

    return "Max steps reached without a final answer."

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        page = await browser.new_page()
        await page.goto("https://www.google.com")

        # Example usage
        questions = [
            "What are the latest blog posts from openai?",
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