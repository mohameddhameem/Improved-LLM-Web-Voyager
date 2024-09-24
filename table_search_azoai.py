
import os
from azure.ai.openai import OpenAIClient
from azure.core.credentials import AzureKeyCredential
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

# Function to send image and prompt to Azure OpenAI's vision-enabled model
def send_to_azure_openai(client, image_data, prompt_text):
    # In Azure OpenAI, we pass the image data and prompt for structured output
    response = client.get_completion(
        deployment_id="your-deployment-id",  # The deployment name for your model
        prompt=prompt_text,
        files=[{"image": image_data}],  # Image passed as file input
        n=1,
        response_format="structured"
    )
    return response

# Example usage to pass an image and get structured output
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
    
    # Initialize Azure OpenAI Client
    azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")  # Azure OpenAI endpoint
    azure_openai_key = os.getenv("AZURE_OPENAI_KEY")  # Azure OpenAI API key
    
    client = OpenAIClient(
        endpoint=azure_openai_endpoint,
        credential=AzureKeyCredential(azure_openai_key)
    )
    
    # Send to Azure OpenAI's image-based model and get structured output
    response = send_to_azure_openai(client, image_data, prompt_text)
    print("Structured Response from Azure OpenAI:")
    print(response)

# Sample usage
if __name__ == "__main__":
    # Make sure to set environment variables for AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_KEY
    # or hardcode them for testing purposes:
    # os.environ["AZURE_OPENAI_ENDPOINT"] = "https://<your-endpoint>.openai.azure.com"
    # os.environ["AZURE_OPENAI_KEY"] = "<your-api-key>"

    # Path to the image and the text to find in the image
    image_path = "sample_image.png"
    target_text = "Sample Text"

    process_image_and_text(image_path, target_text)
