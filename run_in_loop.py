import pandas as pd
import openai
import asyncio
from concurrent.futures import ProcessPoolExecutor

# Sample DataFrame
data = {
    'task_name': ['task1', 'task2', 'task3'],
    'json_path': ['path1.json', 'path2.json', 'path3.json'],
    'expected_results': ['result1', 'result2', 'result3']
}

df = pd.DataFrame(data)

# Set your OpenAI API key
openai.api_key = "YOUR_OPENAI_API_KEY"

# Async function to call OpenAI API
async def process_row(task_name, json_path, expected_results):
    prompt = f"Process task {task_name} with data from {json_path}. Expected results: {expected_results}."
    try:
        # Example OpenAI API call (you might need to adjust this to your specific use case)
        response = await openai.ChatCompletion.acreate(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {str(e)}"

# Synchronous wrapper
def run_async_function(task_name, json_path, expected_results):
    loop = asyncio.new_event_loop()  # Create a new event loop for each process
    asyncio.set_event_loop(loop)
    return loop.run_until_complete(process_row(task_name, json_path, expected_results))

# Function to process each row and return results
def process_row_sync(row):
    task_name = row['task_name']
    json_path = row['json_path']
    expected_results = row['expected_results']
    result = run_async_function(task_name, json_path, expected_results)
    return {
        'task_name': task_name,
        'json_path': json_path,
        'expected_results': expected_results,
        'result': result
    }

# Set the number of parallel processes
max_workers = 2  # Limit the number of parallel processes

# Create a ProcessPoolExecutor with a limited number of workers
with ProcessPoolExecutor(max_workers=max_workers) as executor:
    # Map the process_row_sync function to each row of the DataFrame
    results = list(executor.map(process_row_sync, [row for _, row in df.iterrows()]))

# Create a new DataFrame with the results
results_df = pd.DataFrame(results)

# Display the results DataFrame
print(results_df)
