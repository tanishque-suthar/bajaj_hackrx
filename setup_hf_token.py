#!/usr/bin/env python3
"""
Setup script for Hugging Face token
"""

import os
from dotenv import load_dotenv

def setup_huggingface_token():
    """Help user set up Hugging Face token"""
    print("🤗 Hugging Face Token Setup")
    print("=" * 40)
    print()
    print("To use meta-llama/Llama-3.1-8B-Instruct, you need a Hugging Face token.")
    print()
    print("Steps to get your token:")
    print("1. Go to https://huggingface.co/settings/tokens")
    print("2. Create a new token (Read access is sufficient)")
    print("3. Accept the Llama license at: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct")
    print("4. Copy your token and paste it below")
    print()
    
    # Check if token already exists
    load_dotenv()
    existing_token = os.getenv('HUGGINGFACE_TOKEN')
    
    if existing_token and existing_token != 'your_hf_token_here':
        print(f"✅ Token already configured: {existing_token[:8]}...")
        choice = input("Do you want to update it? (y/N): ").strip().lower()
        if choice != 'y':
            return existing_token
    
    # Get new token
    token = input("Enter your Hugging Face token: ").strip()
    
    if not token:
        print("❌ No token provided. Exiting.")
        return None
    
    # Update .env file
    env_file = ".env"
    if os.path.exists(env_file):
        with open(env_file, 'r') as f:
            content = f.read()
        
        # Replace token line
        if 'HUGGINGFACE_TOKEN=' in content:
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if line.startswith('HUGGINGFACE_TOKEN='):
                    lines[i] = f'HUGGINGFACE_TOKEN={token}'
                    break
            content = '\n'.join(lines)
        else:
            content = f'HUGGINGFACE_TOKEN={token}\n' + content
        
        with open(env_file, 'w') as f:
            f.write(content)
    else:
        with open(env_file, 'w') as f:
            f.write(f'HUGGINGFACE_TOKEN={token}\n')
    
    print(f"✅ Token saved to {env_file}")
    print()
    print("🎉 Setup complete! You can now run the local RAG system.")
    print("Run: python test_local_rag.py")
    
    return token

def test_token():
    """Test if the token works"""
    try:
        from transformers import AutoTokenizer
        import os
        from dotenv import load_dotenv
        
        load_dotenv()
        token = os.getenv('HUGGINGFACE_TOKEN')
        
        if not token:
            print("❌ No token found in .env file")
            return False
        
        print("🔍 Testing token access to Llama-3.1-8B-Instruct...")
        
        # Try to load tokenizer (lightweight test)
        tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-3.1-8B-Instruct",
            token=token
        )
        
        print("✅ Token works! You have access to Llama-3.1-8B-Instruct")
        return True
        
    except Exception as e:
        print(f"❌ Token test failed: {e}")
        print()
        print("Make sure you:")
        print("1. Have a valid Hugging Face token")
        print("2. Accepted the Llama license")
        print("3. Have internet connection")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Setup Hugging Face token")
    parser.add_argument("--test", action="store_true", help="Test existing token")
    args = parser.parse_args()
    
    if args.test:
        test_token()
    else:
        setup_huggingface_token()
