import asyncio
import base64
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, chain
from langchain_openai import ChatOpenAI
from langchain_core.runnables import chain as chain_decorator
from langchain.prompts.chat import ChatPromptTemplate
from langchain.prompts.chat import SystemMessagePromptTemplate
from langchain.prompts.chat import MessagesPlaceholder
from langchain.prompts.chat import HumanMessagePromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts.image import ImagePromptTemplate

# Load mark_page_script
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
            # May be loading...
            asyncio.sleep(3)
    screenshot = await page.screenshot()
    # Ensure the bboxes don't follow us around
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
    action_prefix = "Action: "
    if not text.strip().split("\n")[-1].startswith(action_prefix):
        return {"action": "retry", "args": f"Could not parse LLM Output: {text}"}
    action_block = text.strip().split("\n")[-1]

    action_str = action_block[len(action_prefix) :]
    split_output = action_str.split(" ", 1)
    if len(split_output) == 1:
        action, action_input = split_output[0], None
    else:
        action, action_input = split_output
    action = action.strip()
    if action_input is not None:
        action_input = [
            inp.strip().strip("[]") for inp in action_input.strip().split(";")
        ]
    return {"action": action, "args": action_input}


prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate(
            prompt=[
                PromptTemplate.from_template("""Imagine you are a robot browsing the web, just like humans. 
                                     Now you need to complete a task. In each iteration, you will receive an Observation that includes a screenshot of a webpage and some texts. 
                                     This screenshot will\nfeature Numerical Labels placed in the TOP LEFT corner of each Web Element. 
                                     Carefully analyze the visual\ninformation to identify the Numerical Label corresponding to the Web Element that requires interaction, then follow the guidelines and choose one of the following actions:\n\n
                                     1. Click a Web Element.\n
                                     2. Delete existing content in a textbox and then type content.\n
                                     3. Scroll up or down.\n
                                     4. Wait \n
                                     5. Go back\n
                                     7. Return to google to start over.\n
                                     8. Respond with the final answer\n\n
                                     Correspondingly, Action should STRICTLY follow the format:\n\n- Click [Numerical_Label] \n- Type [Numerical_Label]; [Content] \n- Scroll [Numerical_Label or WINDOW]; [up or down] \n- Wait \n- GoBack\n- Google\n- ANSWER; [content]\n\n
                                     Key Guidelines You MUST follow:
                                     \n\n* Action guidelines *\n1) Execute only one action per iteration.\n
                                     2) When clicking or typing, ensure to select the correct bounding box.\n
                                     3) Numeric labels lie in the top-left corner of their corresponding bounding boxes and are colored the same.\n\n
                                     * Web Browsing Guidelines *\n
                                     1) Don't interact with useless web elements like Login, Sign-in, donation that appear in Webpages\n
                                     2) Select strategically to minimize time wasted.\n\n
                                     Your reply should strictly follow the format:\n\nThought: {{Your brief thoughts (briefly summarize the info that will help ANSWER)}}\nAction: {{One Action format you choose}}\nThen the User will provide:\nObservation: {{A labeled screenshot Given by User}}\n"""),
            ],
        ),
        MessagesPlaceholder(
            optional=True,
            variable_name="scratchpad",
        ),
        HumanMessagePromptTemplate(
            prompt=[
                ImagePromptTemplate(
                    template={"url": "data:image/png;base64,{img}"},
                    input_variables=[
                        "img",
                    ],
                ),
                PromptTemplate.from_template("{bbox_descriptions}"),
                PromptTemplate.from_template("{input}"),
            ],
        ),
    ],
    input_variables=[
        "bbox_descriptions",
        "img",
        "input",
    ],
    partial_variables={"scratchpad": []},
)

llm = ChatOpenAI(model="gpt-4o-mini", max_tokens=4096)

agent = annotate | RunnablePassthrough.assign(
    prediction=format_descriptions | prompt | llm | StrOutputParser() | parse
)
