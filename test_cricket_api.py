#!/usr/bin/env python3
"""
Test the updated generic Wikipedia scraping with cricket questions
"""

from fastapi.testclient import TestClient
from main import app
import json

def test_cricket_api():
    """Test the API with cricket questions"""
    
    # Create test client
    client = TestClient(app)
    
    # Read the cricket question file
    with open("cricket_questions.txt", "r") as f:
        questions_content = f.read()
    
    print("Questions content:")
    print(questions_content)
    print("\n" + "="*50 + "\n")
    
    # Prepare the files for API request
    files = {
        "questions_txt": ("cricket_questions.txt", questions_content, "text/plain")
    }
    
    # Make API request
    response = client.post("/api/", files=files)
    
    print(f"Response status: {response.status_code}")
    
    try:
        result = response.json()
        print(f"Response type: {type(result)}")
        
        if isinstance(result, list):
            print("✓ API returned JSON array as expected!")
            print("Cricket match answers:")
            for i, answer in enumerate(result, 1):
                print(f"{i}. {answer}")
        else:
            print("✗ API returned unexpected format")
            print("Result:", result)
            
        return result
        
    except Exception as e:
        print(f"Error parsing response: {e}")
        print("Raw response:", response.text)
        return None

if __name__ == "__main__":
    print("=== Testing Generic Wikipedia Cricket Scraping ===\n")
    
    result = test_cricket_api()
    
    if result and isinstance(result, list):
        print("\n✓ Test passed! The API successfully returned cricket data in JSON array format.")
        print("Final JSON result:", json.dumps(result, indent=2))
    else:
        print("\n✗ Test failed!")
