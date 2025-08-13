from fastapi.testclient import TestClient
from main import app
import json

# Create test client
client = TestClient(app)

def test_cricket_api():
    """Test the API with cricket questions"""
    
    # Read the sample question file
    with open("New folder/question.txt", "r") as f:
        questions_content = f.read()
    
    print("Questions content:")
    print(questions_content)
    print("\n" + "="*50 + "\n")
    
    # Prepare the files
    files = {
        "questions_txt": ("question.txt", questions_content, "text/plain")
    }
    
    # Make API request
    response = client.post("/api/", files=files)
    
    print(f"Response status: {response.status_code}")
    print(f"Response content: {response.json()}")
    
    return response.json()

if __name__ == "__main__":
    print("Testing Cricket API endpoint...")
    result = test_cricket_api()
    
    if isinstance(result, list):
        print("\n✓ API returned JSON array as expected!")
        print("Final result:", json.dumps(result, indent=2))
    else:
        print("\n✗ API returned unexpected format")
        print("Result type:", type(result))
