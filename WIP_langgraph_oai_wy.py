import os
import asyncio
import platform
import base64
import json
from typing import List, Optional, TypedDict, TypeVar, Dict
from langgraph.graph import Graph, END
from langgraph.prebuilt import ToolExecutor
from langchain.tools import tool
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
    messages: List[dict]
    bboxes: List[BBox]
    bbox_descriptions: str
    img: str

# Define a new type for the state
StateType = TypeVar("StateType", bound=dict)

# Tool functions
@tool
async def click(bbox_id: int, state: Dict):
    """
    Click on a specified element on the page.
    
    Args:
        bbox_id (int): The ID of the bounding box to click.
        state (dict): The current state of the agent.
    
    Returns:
        str: A message indicating the result of the click action.
    """
    page = state["page"]
    try:
        bbox = state["bboxes"][bbox_id]
    except Exception:
        return f"Error: no bbox for : {bbox_id}"
    x, y = bbox["x"], bbox["y"]
    await page.mouse.click(x, y)
    return f"Clicked {bbox_id}"

@tool
async def type_text(bbox_id: int, text: str, state: Dict):
    """
    Type text into a specified element on the page.
    
    Args:
        bbox_id (int): The ID of the bounding box to type into.
        text (str): The text to type.
        state (dict): The current state of the agent.
    
    Returns:
        str: A message indicating the result of the typing action.
    """
    page = state["page"]
    try:
        bbox = state["bboxes"][bbox_id]
    except Exception:
        return f"Error: no bbox for : {bbox_id}"
    x, y = bbox["x"], bbox["y"]
    await page.mouse.click(x, y)
    select_all = "Meta+A" if platform.system() == "Darwin" else "Control+A"
    await page.keyboard.press(select_all)
    await page.keyboard.press("Backspace")
    await page.keyboard.type(text)
    await page.keyboard.press("Enter")
    return f"Typed {text} and submitted"

@tool
async def scroll(target: str, direction: str, state: Dict):
    """
    Scroll the page or a specific element.
    
    Args:
        target (str): 'WINDOW' or the ID of the bounding box to scroll.
        direction (str): The direction to scroll ('up' or 'down').
        state (dict): The current state of the agent.
    
    Returns:
        str: A message indicating the result of the scroll action.
    """
    page = state["page"]
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

@tool
async def wait(state: Dict):
    """
    Wait for a few seconds.
    
    Args:
        state (dict): The current state of the agent.
    
    Returns:
        str: A message indicating the duration of the wait.
    """
    sleep_time = 5
    await asyncio.sleep(sleep_time)
    return f"Waited for {sleep_time}s."

@tool
async def go_back(state: Dict):
    """
    Navigate back to the previous page.
    
    Args:
        state (dict): The current state of the agent.
    
    Returns:
        str: A message indicating the result of the navigation action.
    """
    page = state["page"]
    await page.go_back()
    return f"Navigated back a page to {page.url}."

@tool
async def to_google(state: Dict):
    """
    Navigate to Google homepage.
    
    Args:
        state (dict): The current state of the agent.
    
    Returns:
        str: A message indicating the result of the navigation action.
    """
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
                    "bbox_id": {"type": "integer", "description": "The ID of the bounding box to click"},
                    "state": {"type": "object", "description": "The current state of the agent"}
                },
                "required": ["bbox_id", "state"]
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
                    "bbox_id": {"type": "integer", "description": "The ID of the bounding box to type into"},
                    "text": {"type": "string", "description": "The text to type"},
                    "state": {"type": "object", "description": "The current state of the agent"}
                },
                "required": ["bbox_id", "text", "state"]
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
                    "target": {"type": "string", "description": "WINDOW or the ID of the bounding box to scroll"},
                    "direction": {"type": "string", "enum": ["up", "down"], "description": "The direction to scroll"},
                    "state": {"type": "object", "description": "The current state of the agent"}
                },
                "required": ["target", "direction", "state"]
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
                "properties": {
                    "state": {"type": "object", "description": "The current state of the agent"}
                },
                "required": ["state"]
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
                "properties": {
                    "state": {"type": "object", "description": "The current state of the agent"}
                },
                "required": ["state"]
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
                "properties": {
                    "state": {"type": "object", "description": "The current state of the agent"}
                },
                "required": ["state"]
            },
        },
    },
]

async def call_openai_with_tools(messages, tools):
    response = await client.chat.completions.create(
        model="gpt-4-0613",  # or whichever model you're using
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )
    return response.choices[0].message

def create_workflow(page):
    # Create a ToolExecutor with your tools
    tool_executor = ToolExecutor([click, type_text, scroll, wait, go_back, to_google])

    # Define the workflow
    workflow = Graph()

    # Define the agent node
    def agent(state: AgentState) -> dict:
        messages = state["messages"]
        response = asyncio.run(call_openai_with_tools(messages, tools))
        if not response.tool_calls:
            return {"messages": messages + [response], "response": response, "type": END}
        return {"messages": messages + [response], "response": response, "type": "tool_executor"}

    # Define the tool execution node
    from langgraph.prebuilt import ToolInvocation

    def tool_executor_node(state: AgentState) -> AgentState:
        response = state["response"]
        for tool_call in response.tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            
            # Create a ToolInvocation
            invocation = ToolInvocation(
                tool=function_name,
                tool_input=function_args
            )
            
            # Execute the tool
            tool_result = asyncio.run(tool_executor.ainvoke(invocation, state))
            
            state["messages"].append({"role": "tool", "tool_call_id": tool_call.id, "name": function_name, "content": tool_result})
        return {"type": "update_state", **state}

    # Define the state update node
    def update_state(state: AgentState) -> AgentState:
        new_state = asyncio.run(annotate({"page": state["page"]}))
        new_state = format_descriptions(new_state)
        state.update(new_state)
        state["messages"].append({"role": "system", "content": f"Current page state: {state['bbox_descriptions']}"})
        return {"type": "agent", **state}

    # Add nodes to the graph
    workflow.add_node("agent", agent)
    workflow.add_node("tool_executor", tool_executor_node)
    workflow.add_node("update_state", update_state)

    # Define the edges of the graph
    workflow.add_edge("agent", "tool_executor")
    workflow.add_edge("tool_executor", "update_state")
    workflow.add_edge("update_state", "agent")

    # Set the entry point
    workflow.set_entry_point("agent")

    return workflow

async def call_agent(question: str, page, max_steps: int = 150):
    workflow = create_workflow(page)
    
    initial_state = await annotate({"page": page})
    initial_state = format_descriptions(initial_state)
    
    messages = [{"role": "user", "content": question}]
    if 'bbox_descriptions' in initial_state and initial_state['bbox_descriptions']:
        messages.append({"role": "system", "content": f"Current page state: {initial_state['bbox_descriptions']}"})
    else:
        messages.append({"role": "system", "content": "Current page state could not be determined. Please navigate to a relevant page."})

    initial_state["messages"] = messages

    # Compile the workflow
    app = workflow.compile()

    # Run the compiled workflow
    step = 0
    async for event in app.astream(initial_state):
        # The event now directly contains the state
        state = event
        
        if "response" in state:
            response = state["response"]
            if not response.tool_calls:
                print(f"Final response: {response.content}")
                return response.content
        
        # Display the current state (optional)
        display.clear_output(wait=False)
        print(f"Step {step}: {state.get('type', 'unknown')}")
        if 'img' in state and state['img']:
            display.display(display.Image(base64.b64decode(state['img'])))
        
        step += 1
        if step >= max_steps:
            return "Max steps reached without a final answer."

    return "Workflow completed without a final answer."

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