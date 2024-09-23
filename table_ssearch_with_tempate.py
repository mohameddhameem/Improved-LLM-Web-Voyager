import openai
from langchain.prompts import PromptTemplate
from jinja2 import Environment, FileSystemLoader, meta
from PIL import Image
import io

# Function to infer required keys from the Jinja template
def infer_required_keys(template_path):
    env = Environment(loader=FileSystemLoader("."))
    with open(template_path) as file:
        template_source = file.read()
    parsed_content = env.parse(template_source)
    return meta.find_undeclared_variables(parsed_content)

# Function to render the Jinja template with provided values
def render_template(template_path, values):
    required_keys = infer_required_keys(template_path)
    
    # Validate that all required keys are present in the provided values
    missing_keys = required_keys - values.keys()
    if missing_keys:
        raise ValueError(f"Missing required keys: {', '.join(missing_keys)}")
    
    # Load and render the template
    env = Environment(loader=FileSystemLoader("."))
    template = env.get_template(template_path)
    return template.render(values)

# Function to send image and prompt to OpenAI's vision-enabled model
def send_to_openai(image_data, prompt_text):
    response = openai.Image.create(
        image=image_data,
        prompt=prompt_text,
        n=1,
        response_format="structured"
    )
    return response

# Example usage with LangChain to pass an image and get structured output
def process_image_and_text(image_path, target_text):
    # Load the image
    with open(image_path, "rb") as image_file:
        image_data = image_file.read()
    
    # Render the prompt from the Jinja template
    template_path = "prompt_image_text.jinja"
    values = {
        "target_text": target_text,
        "image_description": "A photo containing various objects and text elements."
    }
    
    try:
        prompt_text = render_template(template_path, values)
        print(f"Rendered Prompt:\n{prompt_text}")
    except ValueError as e:
        print(f"Error: {e}")
        return
    
    # Send to OpenAI's image-based model and get structured output
    response = send_to_openai(image_data, prompt_text)
    print("Structured Response from OpenAI:")
    print(response)

# Sample usage
if __name__ == "__main__":
    # OpenAI API key setup
    openai.api_key = "your-openai-api-key"

    # Path to the image and the text to find in the image
    image_path = "sample_image.png"
    target_text = "Sample Text"

    process_image_and_text(image_path, target_text)
