import asyncio
import platform
from agent_types import AgentState


async def click(state: AgentState):
    page = state["page"]
    action = state["prediction"]
    if action.action != "Click" or len(action.args) != 1:
        return "Failed to click bounding box: invalid action or arguments"
    bbox_id = int(action.args[0])
    try:
        bbox = state["bboxes"][bbox_id]
    except Exception:
        return f"Error: no bbox for : {bbox_id}"
    x, y = bbox["x"], bbox["y"]
    await page.mouse.click(x, y)
    return f"Clicked {bbox_id}"


async def type_text(state: AgentState):
    page = state["page"]
    action = state["prediction"]

    if action.action != "Type" or len(action.args) != 2:
        return "Failed to type: invalid action or arguments"

    bbox_id, text_content = action.args
    bbox_id = int(bbox_id)

    try:
        bbox = state["bboxes"][bbox_id]
    except Exception:
        return f"Error: no bbox for : {bbox_id}"

    x, y = bbox["x"], bbox["y"]
    await page.mouse.click(x, y)

    # Check if MacOS
    select_all = "Meta+A" if platform.system() == "Darwin" else "Control+A"
    await page.keyboard.press(select_all)
    await page.keyboard.press("Backspace")
    await page.keyboard.type(text_content)
    await page.keyboard.press("Enter")

    return f"Typed '{text_content}' into element {bbox_id} and submitted"


async def scroll(state: AgentState):
    page = state["page"]
    action = state["prediction"]

    if action.action != "Scroll" or len(action.args) != 2:
        return "Failed to scroll: invalid action or arguments"

    target, direction = action.args

    if target.upper() == "WINDOW":
        scroll_amount = 500
        scroll_direction = (
            -scroll_amount if direction.lower() == "up" else scroll_amount
        )
        await page.evaluate(f"window.scrollBy(0, {scroll_direction})")
    else:
        scroll_amount = 200
        try:
            target_id = int(target)
            bbox = state["bboxes"][target_id]
        except Exception:
            return f"Error: invalid target for scrolling: {target}"

        x, y = bbox["x"], bbox["y"]
        scroll_direction = (
            -scroll_amount if direction.lower() == "up" else scroll_amount
        )
        await page.mouse.move(x, y)
        await page.mouse.wheel(0, scroll_direction)

    return f"Scrolled {direction} in {'window' if target.upper() == 'WINDOW' else f'element {target}'}"


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
