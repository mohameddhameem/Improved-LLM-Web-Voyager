from jinja2 import Environment, FileSystemLoader, meta, Template
import os

# Function to infer required keys from the Jinja template
def infer_required_keys(template_path):
    env = Environment(loader=FileSystemLoader(os.path.dirname(template_path)))
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
    env = Environment(loader=FileSystemLoader(os.path.dirname(template_path)))
    template = env.get_template(os.path.basename(template_path))
    return template.render(values)

# Example usage
if __name__ == "__main__":
    # Path to the Jinja template
    template_path = "prompt_template.jinja"
    
    # Values to be used for rendering the template
    values = {
        "user_name": "John Doe",
        "task_name": "Monthly Report",
        "deadline": "2024-09-30",
        "priority": "High"
    }
    
    # Render the template
    try:
        rendered_prompt = render_template(template_path, values)
        print(rendered_prompt)
    except ValueError as e:
        print(f"Error: {e}")
