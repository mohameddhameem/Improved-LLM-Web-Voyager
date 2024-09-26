import os
import base64
from typing import TypedDict, Annotated, Sequence
from langgraph.graph import Graph, END
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

# Set your OpenAI API key
# os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY")

# Define state
class State(TypedDict):
    messages: Annotated[Sequence[str], "The messages in the conversation"]
    image: Annotated[str, "The base64 encoded image"]
    extracted_text: Annotated[str, "The text extracted from the image"]
    summary: Annotated[str, "The summary of the extracted text"]

# Function to encode image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Node functions
def process_image(state: State):
    image = state['image']
    llm = ChatOpenAI(model="gpt-4o-mini", max_tokens=1000)  # Changed from "gpt-4o-mini" to "gpt-4-vision-preview"
    message = HumanMessage(
        content=[
            {"type": "text", "text": "Extract all the text from this image."},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image}"
                }
            }
        ]
    )
    response = llm.invoke([message])
    state['extracted_text'] = response.content
    print(f"Extracted text is: {state['extracted_text']}")
    return state

def summarize_text(state: State):
    extracted_text = state['extracted_text']
    llm = ChatOpenAI(model="gpt-3.5-turbo", max_tokens=300)
    message = HumanMessage(content=f"Summarize the following text:\n\n{extracted_text}")
    response = llm.invoke([message])
    state['summary'] = response.content
    print(f"Summary: {state['summary']}")
    return state

# Define the graph
workflow = Graph()

# Add nodes
workflow.add_node("process_image", process_image)
workflow.add_node("summarize_text", summarize_text)

# Set the entry point
workflow.set_entry_point("process_image")

# Add edges
workflow.add_edge("process_image", "summarize_text")
workflow.add_edge("summarize_text", END)

# Compile the graph
app = workflow.compile()

def main(image_path: str):
    # Encode the image
    base64_image = encode_image(image_path)
    
    # Initial state
    initial_state = State(
        messages=[],
        image=base64_image,
        extracted_text="",
        summary=""
    )
    
    # Run the graph
    final_output = app.invoke(initial_state)
    
    print("Final output:", final_output)
    if 'summary' in final_output:
        print("Final Summary:", final_output['summary'])
    else:
        print("No summary found in the final output.")

if __name__ == "__main__":
    image_path = "numbers-table.png"  
    main(image_path)