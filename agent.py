import asyncio
from typing import List
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
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
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# Load mark_page_script
with open("mark_page.js") as f:
    mark_page_script = f.read()

from typing import Union

def parse_with_step_count(x: str, step_count: Union[int, str]):
    return AgentAction.parse_raw(x, int(step_count))

class AgentAction(BaseModel):
    action: str = Field(description="The action to take")
    args: List[str] = Field(description="Arguments for the action")
    thought: str = Field(description="Reasoning behind the action")
    step_count: int = Field(default=0, description="Number of steps taken")

    @classmethod
    def parse_raw(cls, raw_string: str, step_count: int) -> 'AgentAction':
        # Split the input string into thought and action parts
        parts = raw_string.split("\n")
        thought = ""
        action = ""
        args = []
        
        for part in parts:
            if part.startswith("Thought:"):
                thought = part.replace("Thought:", "").strip()
            elif part.startswith("Action:"):
                action_part = part.replace("Action:", "").strip()
                action_split = action_part.split(maxsplit=1)
                action = action_split[0]
                if len(action_split) > 1:
                    args = [arg.strip() for arg in action_split[1].split(';')]
        
        return cls(action=action, args=args, thought=thought, step_count=int(step_count))



output_parser = PydanticOutputParser(pydantic_object=AgentAction)


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
                PromptTemplate.from_template(
                    """Imagine you are a robot browsing the web, just like humans. 
                    Your task is to answer the user's question about Google Business.
                    In each iteration, you will receive an Observation that includes a screenshot of a webpage and some texts.
                    This screenshot will feature Numerical Labels placed in the TOP LEFT corner of each Web Element.
                    Carefully analyze the visual information to identify the Numerical Label corresponding to the Web Element that requires interaction, then follow the guidelines and choose one of the following actions:

                    1. Click a Web Element.
                    2. Delete existing content in a textbox and then type content.
                    3. Scroll up or down.
                    4. Wait 
                    5. Go back
                    6. Return to google to start over.
                    7. Respond with the final answer

                    Key Guidelines You MUST follow:

                    * Action guidelines *
                    1) Execute only one action per iteration.
                    2) When clicking or typing, ensure to select the correct bounding box.
                    3) Numeric labels lie in the top-left corner of their corresponding bounding boxes and are colored the same.
                    4) After gathering sufficient information, provide an ANSWER.
                    5) If you've taken more than 8 steps, you MUST provide an ANSWER based on the information gathered so far.

                    * Web Browsing Guidelines *
                    1) Don't interact with useless web elements like Login, Sign-in, donation that appear in Webpages
                    2) Select strategically to minimize time wasted.

                    Your response should be in the following format:
                    Thought: [Your reasoning]
                    Action: [Action] [Arguments separated by semicolons if any]

                    Remember, your goal is to answer the question about Google Business. If you have gathered enough information, use the ANSWER action to provide the final response.
                    """
                ),
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
    partial_variables={
        "scratchpad": [],
    },
)

llm = ChatOpenAI(model="gpt-4o-mini", max_tokens=4096)



def increment_step(state):
    step_count = state.get("step_count", 0) + 1
    return {**state, "step_count": step_count}



agent = (
    annotate 
    | RunnablePassthrough.assign(step_count=increment_step)
    | RunnablePassthrough.assign(
        prediction=RunnablePassthrough.assign(
            parsed_output=format_descriptions 
            | prompt 
            | llm 
            | StrOutputParser()
        )
        | RunnableLambda(lambda x: parse_with_step_count(x["parsed_output"], x["step_count"]["step_count"]))
    )
)