import requests
import json

# Test the API endpoint
url = "http://localhost:8000/api/"

# Prepare files
files = {
    'questions_txt': ('questions.txt', open('questions.txt', 'rb')),
    'files': ('edges.csv', open('edges.csv', 'rb'))
}

try:
    response = requests.post(url, files=files)
    print(f"Status Code: {response.status_code}")
    print(f"Response Headers: {response.headers}")
    print("Response Body:")
    print(json.dumps(response.json(), indent=2))
except Exception as e:
    print(f"Error: {e}")
    print(f"Raw response: {response.text}")
finally:
    # Close file handles
    for file_obj in files.values():
        if hasattr(file_obj[1], 'close'):
            file_obj[1].close()
