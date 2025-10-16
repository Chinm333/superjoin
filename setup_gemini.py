#!/usr/bin/env python3
"""
Setup script for Gemini AI integration
"""

import os
import sys
from pathlib import Path

def setup_gemini_api_key():
    """Setup Gemini API key in environment"""
    print("🔧 Setting up Gemini AI integration...")
    print("=" * 50)
    
    # Check if API key is already set
    current_key = os.getenv('GEMINI_API_KEY')
    if current_key:
        print(f"✅ GEMINI_API_KEY is already set: {current_key[:10]}...")
        return True
    
    # Get API key from user
    print("To use Gemini AI for enhanced semantic understanding, you need a Google AI Studio API key.")
    print("Get your free API key from: https://aistudio.google.com/app/apikey")
    print()
    
    api_key = input("Enter your Gemini API key (or press Enter to skip): ").strip()
    
    if not api_key:
        print("⚠️  Skipping Gemini setup. The system will use rule-based semantic analysis.")
        return False
    
    # Set environment variable for current session
    os.environ['GEMINI_API_KEY'] = api_key
    print(f"✅ GEMINI_API_KEY set for current session: {api_key[:10]}...")
    
    # Create .env file for persistence
    env_file = Path('.env')
    try:
        with open(env_file, 'w') as f:
            f.write(f"GEMINI_API_KEY={api_key}\n")
            f.write("GEMINI_MODEL=gemini-1.5-flash\n")
            f.write("GEMINI_MAX_TOKENS=1000\n")
            f.write("GEMINI_TEMPERATURE=0.3\n")
        print(f"✅ Created {env_file} for persistent configuration")
    except Exception as e:
        print(f"⚠️  Could not create .env file: {e}")
        print("You'll need to set GEMINI_API_KEY manually for each session")
    
    return True

def test_gemini_connection():
    """Test Gemini API connection"""
    print("\n🧪 Testing Gemini connection...")
    
    try:
        import google.generativeai as genai
        
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            print("❌ No API key found")
            return False
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Simple test
        response = model.generate_content("Say 'Hello from Gemini'")
        print(f"✅ Gemini connection successful: {response.text}")
        return True
        
    except ImportError:
        print("❌ google-generativeai not installed. Run: pip install google-generativeai")
        return False
    except Exception as e:
        print(f"❌ Gemini connection failed: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 SuperJoin - Gemini AI Setup")
    print("=" * 50)
    
    # Setup API key
    if setup_gemini_api_key():
        # Test connection
        if test_gemini_connection():
            print("\n🎉 Gemini AI integration setup complete!")
            print("Your semantic search engine is now enhanced with AI-powered understanding.")
        else:
            print("\n⚠️  Setup incomplete. Check your API key and try again.")
    else:
        print("\n📋 System will use rule-based semantic analysis.")
        print("You can run this setup script again later to enable Gemini AI.")
    
    print("\nNext steps:")
    print("1. Run the demo: python demo/demo_non_interactive.py")
    print("2. Test with your own data: python src/main.py --file your_file.xlsx --query 'find revenue calculations'")

if __name__ == "__main__":
    main()
