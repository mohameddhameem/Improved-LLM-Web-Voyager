import openai
from langchain.prompts import PromptTemplate
from langchain.agents import AgentExecutor, create_openai_functions_agent
# from langchain.chains import SimpleChain
# from langgraph import LangGraph
from jinja2 import Environment, FileSystemLoader, meta
from PIL import Image
import io
import os

# Jinja-based Template Handling
def infer_required_keys(template_path):
    env = Environment(loader=FileSystemLoader("."))
    with open(template_path) as file:
        template_source = file.read()
    parsed_content = env.parse(template_source)
    return meta.find_undeclared_variables(parsed_content)

def render_template(template_path, values):
    required_keys = infer_required_keys(template_path)
    
    missing_keys = required_keys - values.keys()
    if missing_keys:
        raise ValueError(f"Missing required keys: {', '.join(missing_keys)}")
    
    env = Environment(loader=FileSystemLoader("."))
    template = env.get_template(template_path)
    return template.render(values)

# Define the LangChain agent
class ImageTextAgent:
    def __init__(self, openai_api_key, model_name="gpt-4o-mini"):
        self.openai_api_key = openai_api_key
        openai.api_key = self.openai_api_key
        self.agent = create_openai_function_agent(model_name=model_name)  # Using GPT-4o-mini model

    def describe_image(self, image_data, additional_text):
        """
        Takes image data and additional text and returns a description using GPT-4o-mini.
        """
        # Generate a prompt using Jinja-based template
        template_path = "image_text_prompt.jinja"
        values = {
            "image_description": "This is an image with various elements that need to be described.",
            "additional_text": additional_text
        }
        
        try:
            prompt = render_template(template_path, values)
        except ValueError as e:
            print(f"Error: {e}")
            return
        
        # Call GPT-4o-mini for image description
        response = self.agent.run(prompt=prompt)
        return response

# Create a LangGraph-based Chain for image and text handling
def create_image_chain(agent):
    def handle_image(image_data, text):
        return agent.describe_image(image_data, text)
    
    graph = LangGraph()
    graph.add_node("input_image", handle_image, dependencies=[])
    return graph

# Example usage of the agent
if __name__ == "__main__":
    # Initialize the LangChain agent with OpenAI API key
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    agent = ImageTextAgent(openai_api_key=openai_api_key)
    
    # Load an example image
    image_path = "numbers-table.png"
    with open(image_path, "rb") as image_file:
        image_data = image_file.read()
    
    # Define additional text input to be used in the prompt
    additional_text = "This image shows a busy street with various vehicles."

    # Create the LangGraph chain to handle image and text
    image_chain = create_image_chain(agent)
    
    # Execute the chain and get the description
    result = image_chain.execute({"input_image": (image_data, additional_text)})
    
    # Print the result description
    print("Image Description:", result)
