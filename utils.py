import base64
from IPython import display
from typing import List, Optional

async def call_agent(question: str, page, graph, max_steps: int = 150):
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
    final_answer: Optional[str] = None
    steps: List[str] = []
    
    async for event in event_stream:
        if "agent" not in event:
            continue
        
        pred = event["agent"].get("prediction")
        if not pred:
            continue
        
        action = pred.action
        args = pred.args
        thought = pred.thought

        display.clear_output(wait=False)
        steps.append(f"{len(steps) + 1}. Thought: {thought}\nAction: {action}\nArgs: {args}")
        print("\n".join(steps))
        display.display(display.Image(base64.b64decode(event["agent"]["img"])))
        
        if action == "ANSWER":
            final_answer = args[0] if args else None
            break
    
    return final_answer