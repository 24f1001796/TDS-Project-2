#!/usr/bin/env python3
"""
Test script for generic Wikipedia scraping functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import scrape_wikipedia_page
import json

def test_cricket_scraping():
    """Test cricket match scraping"""
    print("Testing Cricket World Cup 1983 scraping...")
    
    questions = """
1. What was the total number of runs scored by India in the match?
2. What was the total number of runs scored by West Indies in the match? 
3. Which Indian bowler took the most wickets while conceding the fewest runs?
4. Which team won the match, and by what margin runs?
"""
    
    url = 'https://en.wikipedia.org/wiki/1983_Cricket_World_Cup_Final'
    
    try:
        answers = scrape_wikipedia_page(url, questions)
        print("✓ Cricket scraping successful!")
        print("Questions and Answers:")
        question_lines = [q.strip() for q in questions.split('\n') if q.strip() and any(char.isdigit() for char in q[:3])]
        
        for i, (question, answer) in enumerate(zip(question_lines, answers), 1):
            print(f"{i}. {question}")
            print(f"   Answer: {answer}")
        
        # Return as JSON array as required
        result = json.dumps(answers)
        print(f"\nJSON Result: {result}")
        return answers
        
    except Exception as e:
        print(f"✗ Cricket scraping failed: {e}")
        return None

def test_generic_scraping():
    """Test with a different Wikipedia page"""
    print("\nTesting generic Wikipedia scraping with different content...")
    
    questions = """
1. When was the company founded?
2. Who is the current CEO?
3. How many employees does the company have?
4. What is the company's main product?
"""
    
    url = 'https://en.wikipedia.org/wiki/Microsoft'
    
    try:
        answers = scrape_wikipedia_page(url, questions)
        print("✓ Generic scraping successful!")
        print("Questions and Answers:")
        question_lines = [q.strip() for q in questions.split('\n') if q.strip() and any(char.isdigit() for char in q[:3])]
        
        for i, (question, answer) in enumerate(zip(question_lines, answers), 1):
            print(f"{i}. {question}")
            print(f"   Answer: {answer}")
        
        result = json.dumps(answers)
        print(f"\nJSON Result: {result}")
        return answers
        
    except Exception as e:
        print(f"✗ Generic scraping failed: {e}")
        return None

if __name__ == "__main__":
    print("=== Generic Wikipedia Scraping Test ===\n")
    
    # Test cricket scraping (original requirement)
    cricket_results = test_cricket_scraping()
    
    # Test generic scraping
    generic_results = test_generic_scraping()
    
    print("\n=== Summary ===")
    if cricket_results:
        print("✓ Cricket test passed")
    else:
        print("✗ Cricket test failed")
        
    if generic_results:
        print("✓ Generic test passed")
    else:
        print("✗ Generic test failed")
    
    print("\nThe main.py file now supports generic Wikipedia scraping for any page and questions!")
