#!/usr/bin/env python3
"""
Test script to verify OpenAI integration works correctly
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

def test_openai_integration():
    """Test OpenAI API integration"""
    print("Testing OpenAI Integration...")
    
    # Get API key from environment or prompt user
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        api_key = input("Enter your OpenAI API key: ").strip()
    
    if not api_key:
        print("❌ No API key provided")
        return False
    
    try:
        # Test with GPT-4o-mini (the model you requested)
        print("Testing with gpt-4o-mini...")
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            openai_api_key=api_key,
            temperature=0.7,
            max_tokens=100
        )
        
        # Test message
        response = llm.invoke("Hello! Please respond with a short greeting.")
        print(f"✅ Success! Response: {response.content}")
        return True
        
    except Exception as e:
        error_msg = str(e)
        if "401" in error_msg or "unauthorized" in error_msg.lower():
            print("❌ Invalid or expired API key")
        elif "429" in error_msg:
            print("❌ Rate limit exceeded - but API key is valid")
        else:
            print(f"❌ Test failed: {error_msg}")
        return False

if __name__ == "__main__":
    test_openai_integration()