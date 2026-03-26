import requests
import json
import uuid

BASE_URL = "http://localhost:8000"

def test_legal_reasoning():
    session_id = str(uuid.uuid4())
    payload = {
        "user_input": "The seller is ignoring me, I bought a phone 2 months ago and it exploded yesterday causing a small burn on my hand. I want my money back and compensation.",
        "session_id": session_id,
        "llm_model": "llama3.1:8b"
    }
    
    print(f"--- Sending initial request (Session: {session_id}) ---")
    response = requests.post(f"{BASE_URL}/legal_reasoning", json=payload)
    
    if response.status_code == 200:
        data = response.json()
        print(f"Response:\n{data['text']}")
    else:
        print(f"Error {response.status_code}: {response.text}")

if __name__ == "__main__":
    # Note: Requires the API to be running on localhost:8000
    print("This script helps verify the /legal_reasoning endpoint.")
    print("Ensure the API is running before executing.")
    # test_legal_reasoning() 
