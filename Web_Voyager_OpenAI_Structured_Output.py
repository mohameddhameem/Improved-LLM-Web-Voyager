import os
import asyncio
import platform
import base64
from typing import List, Optional, Union, Literal
from pydantic import BaseModel, Field, PydanticUserError
import nest_asyncio
from playwright.async_api import Page, async_playwright
from IPython import display
from openai import AsyncOpenAI

# Apply nest_asyncio for running async playwright in a Jupyter notebook
nest_asyncio.apply()

# Initialize the OpenAI client
client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Updated Pydantic models for structured outputs
class BBox(BaseModel):
    x: float
    y: float
    text: str
    type: str
    ariaLabel: str

class ClickAction(BaseModel):
    type: Literal["click"] = "click"
    bbox_id: int = Field(..., description="The ID of the bounding box to click")

class TypeTextAction(BaseModel):
    type: Literal["type_text"] = "type_text"
    bbox_id: int = Field(..., description="The ID of the bounding box to type into")
    text: str = Field(..., description="The text to type")

class ScrollAction(BaseModel):
    type: Literal["scroll"] = "scroll"
    target: str = Field(..., description="WINDOW or the ID of the bounding box to scroll")
    direction: Literal["up", "down"] = Field(..., description="The direction to scroll")

class WaitAction(BaseModel):
    type: Literal["wait"] = "wait"

class GoBackAction(BaseModel):
    type: Literal["go_back"] = "go_back"

class ToGoogleAction(BaseModel):
    type: Literal["to_google"] = "to_google"

class AgentAction(BaseModel):
    type: Literal["agent_action"] = "agent_action"
    action: Union[ClickAction, TypeTextAction, ScrollAction, WaitAction, GoBackAction, ToGoogleAction]
    explanation: str = Field(..., description="Explanation for the chosen action")

class FinalResponse(BaseModel):
    type: Literal["final_response"] = "final_response"
    content: str = Field(..., description="The final response content")

class AgentOutput(BaseModel):
    response: Union[AgentAction, FinalResponse] = Field(..., discriminator="type")

# Tool functions (keep the existing implementations)
async def click(page: Page, bbox_id: int, bboxes: List[BBox]):
    try:
        bbox = bboxes[bbox_id]
    except IndexError:
        return f"Error: no bbox for : {bbox_id}"
    x, y = bbox.x, bbox.y
    await page.mouse.click(x, y)
    return f"Clicked {bbox_id}"

async def type_text(page: Page, bbox_id: int, text: str, bboxes: List[BBox]):
    try:
        bbox = bboxes[bbox_id]
    except IndexError:
        return f"Error: no bbox for : {bbox_id}"
    x, y = bbox.x, bbox.y
    await page.mouse.click(x, y)
    select_all = "Meta+A" if platform.system() == "Darwin" else "Control+A"
    await page.keyboard.press(select_all)
    await page.keyboard.press("Backspace")
    await page.keyboard.type(text)
    await page.keyboard.press("Enter")
    return f"Typed {text} and submitted"

async def scroll(page: Page, target: str, direction: str, bboxes: List[BBox]):
    if target.upper() == "WINDOW":
        scroll_amount = 500
        scroll_direction = -scroll_amount if direction.lower() == "up" else scroll_amount
        await page.evaluate(f"window.scrollBy(0, {scroll_direction})")
        return f"Scrolled {direction} in window"
    else:
        try:
            target_id = int(target)
            if target_id < 0 or target_id >= len(bboxes):
                return f"Error: Invalid bounding box index {target_id}"
            bbox = bboxes[target_id]
            scroll_amount = 200
            x, y = bbox.x, bbox.y
            scroll_direction = -scroll_amount if direction.lower() == "up" else scroll_amount
            await page.mouse.move(x, y)
            await page.mouse.wheel(0, scroll_direction)
            return f"Scrolled {direction} in element {target_id}"
        except ValueError:
            return f"Error: Invalid target '{target}'. Expected 'WINDOW' or a number."
        except Exception as e:
            return f"Error while scrolling: {str(e)}"

async def wait():
    sleep_time = 5
    await asyncio.sleep(sleep_time)
    return f"Waited for {sleep_time}s."

async def go_back(page: Page):
    await page.go_back()
    return f"Navigated back a page to {page.url}."

async def to_google(page: Page):
    await page.goto("https://www.google.com/")
    return "Navigated to google.com."

# Load JavaScript for page marking
with open("mark_page.js") as f:
    mark_page_script = f.read()

async def mark_page(page: Page, max_retries=3):
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
                        "bboxes": [BBox(**bbox) for bbox in bboxes],
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

def format_descriptions(bboxes: List[BBox]) -> str:
    labels = []
    for i, bbox in enumerate(bboxes):
        text = bbox.ariaLabel or ""
        if not text.strip():
            text = bbox.text
        labels.append(f'{i} (<{bbox.type}/>): "{text}"')
    return "\nValid Bounding Boxes:\n" + "\n".join(labels)

async def call_openai_with_pydantic(messages: List[dict], page_state: str):
    try:
        response = await client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=messages + [{"role": "system", "content": f"Current page state: {page_state}"}],
            response_format=AgentOutput
        )
        return response.choices[0].message
    except Exception as e:
        print(f"Error calling OpenAI API: {str(e)}")
        return None

async def call_agent(question: str, page: Page, max_steps: int = 150):
    messages = [{"role": "user", "content": question}]
    marked_page = await mark_page(page)
    bboxes = marked_page["bboxes"]
    bbox_descriptions = format_descriptions(bboxes)

    for step in range(max_steps):
        response = await call_openai_with_pydantic(messages, bbox_descriptions)
        
        if response is None:
            print("Failed to get a response from OpenAI. Ending the conversation.")
            return "Failed to get a response from OpenAI."

        if response.refusal:
            print(f"Model refused to respond: {response.refusal}")
            return f"Model refused to respond: {response.refusal}"

        if response.parsed.response.type == "final_response":
            print(f"Final response: {response.parsed.response.content}")
            return response.parsed.response.content

        agent_action = response.parsed.response
        action = agent_action.action
        messages.append({"role": "assistant", "content": agent_action.explanation})

        if isinstance(action, ClickAction):
            result = await click(page, action.bbox_id, bboxes)
        elif isinstance(action, TypeTextAction):
            result = await type_text(page, action.bbox_id, action.text, bboxes)
        elif isinstance(action, ScrollAction):
            result = await scroll(page, action.target, action.direction, bboxes)
        elif isinstance(action, WaitAction):
            result = await wait()
        elif isinstance(action, GoBackAction):
            result = await go_back(page)
        elif isinstance(action, ToGoogleAction):
            result = await to_google(page)
        else:
            result = f"Unknown action: {type(action).__name__}"
        
        messages.append({"role": "function", "name": action.type, "content": result})

        # Update the page state
        marked_page = await mark_page(page)
        bboxes = marked_page["bboxes"]
        bbox_descriptions = format_descriptions(bboxes)

        # Display the current state (optional)
        display.clear_output(wait=False)
        print(f"Step {step + 1}: {action.type}")
        if marked_page["img"]:
            display.display(display.Image(base64.b64decode(marked_page["img"])))

    return "Max steps reached without a final answer."

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        page = await browser.new_page()
        await page.goto("https://www.google.com")

        # Example usage
        questions = [
            "What are the latest blog posts from langchain?",
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