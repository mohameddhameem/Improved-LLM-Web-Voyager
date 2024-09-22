import os
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
import base64
from openai import OpenAI
import json

# OpenAI API configuration
OPENAI_API_KEY = "your_openai_api_key_here"

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def take_full_page_screenshot(page):
    return page.screenshot(full_page=True)

def encode_image(image_bytes):
    return base64.b64encode(image_bytes).decode('utf-8')

def query_gpt4v(image, query):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": query},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}"}}
                ]
            }
        ],
        max_tokens=300,
        response_format={"type": "json_object"}
    )
    return response

def parse_gpt4v_response(response):
    try:
        content = json.loads(response.choices[0].message.content)
        return content.get('bounding_box')
    except (KeyError, json.JSONDecodeError) as e:
        print(f"Error parsing GPT-4V response: {e}")
        return None

def interact_with_element(page, coordinates):
    if coordinates:
        center_x = (coordinates['x1'] + coordinates['x2']) // 2
        center_y = (coordinates['y1'] + coordinates['y2']) // 2
        page.mouse.click(center_x, center_y)
        return True
    return False

def scan_page_for_content(page, search_criteria):
    viewport_height = page.viewport_size['height']
    last_height = page.evaluate('document.body.scrollHeight')
    
    while True:
        # Take screenshot of current viewport
        screenshot = take_full_page_screenshot(page)
        encoded_image = encode_image(screenshot)
        
        # Query GPT-4V
        query = f"""Analyze the image and find a cell containing {search_criteria}. 
        If found, provide the bounding box coordinates of this cell. 
        If not found, respond with "not_found".
        Respond with a JSON object in the following format:
        {{
            "bounding_box": {{
                "x1": int,
                "y1": int,
                "x2": int,
                "y2": int
            }}
        }}
        or
        {{
            "result": "not_found"
        }}
        """
        gpt4v_response = query_gpt4v(encoded_image, query)
        coordinates = parse_gpt4v_response(gpt4v_response)
        
        if coordinates:
            return coordinates
        
        # Scroll down
        page.evaluate(f'window.scrollBy(0, {viewport_height});')
        page.wait_for_timeout(1000)  # Wait for any dynamic content to load
        
        new_height = page.evaluate('document.body.scrollHeight')
        if new_height == last_height:
            break
        last_height = new_height
    
    return None

def main():
    url = "https://www.w3schools.com/html/html_tables.asp"
    search_criteria = "find Alfreds Futterkiste	in Germany"
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context(no_viewport=True)
        page = context.new_page()
        
        # Maximize the browser window
        # page.set_viewport_size({"width": 1920, "height": 1080})  # Set to a large size
        # page.evaluate("() => { window.moveTo(0, 0); window.resizeTo(screen.width, screen.height); }")

        page.goto(url, timeout=60000)
        # page.wait_for_load_state('networkidle')
        
        coordinates = scan_page_for_content(page, search_criteria)
        
        if coordinates:
            print(f"Found content at coordinates: {coordinates}")
            interact_with_element(page, coordinates)
            page.wait_for_timeout(5000)
        else:
            print("Content not found on the page.")
        
        browser.close()

if __name__ == "__main__":
    main()