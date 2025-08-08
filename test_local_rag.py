"""
Test script for Local RAG Service
Tests local models configured in .env file
"""

import asyncio
import logging
import os
from dotenv import load_dotenv
from core.local_rag import LocalRAGService, LocalModelConfig

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_local_rag():
    """Test the local RAG service with proper models"""
    
    # Check for Hugging Face token
    hf_token = os.getenv('HUGGINGFACE_TOKEN')
    if not hf_token or hf_token == 'your_hf_token_here':
        print("❌ Hugging Face token not found!")
        print("Run: python setup_hf_token.py")
        return
    
    print("🚀 Testing Local RAG Service")
    print("=" * 50)
    
    # Read configuration from environment variables
    required_env_vars = {
        "LOCAL_EMBEDDING_MODEL": os.getenv("LOCAL_EMBEDDING_MODEL"),
        "LOCAL_LLM_MODEL": os.getenv("LOCAL_LLM_MODEL"),
        "DEVICE": os.getenv("DEVICE"),
        "USE_QUANTIZATION": os.getenv("USE_QUANTIZATION"),
        "QUANTIZATION_BITS": os.getenv("QUANTIZATION_BITS"),
        "MAX_NEW_TOKENS": os.getenv("MAX_NEW_TOKENS"),
        "TEMPERATURE": os.getenv("TEMPERATURE"),
        "TOP_K": os.getenv("TOP_K"),
        "TOP_K_CHUNKS": os.getenv("TOP_K_CHUNKS")
    }
    
    # Check for missing environment variables
    missing_vars = [var for var, value in required_env_vars.items() if value is None]
    if missing_vars:
        print(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        print("Please add them to your .env file.")
        return
    
    # Create config from environment variables
    config = LocalModelConfig(
        embedding_model=required_env_vars["LOCAL_EMBEDDING_MODEL"],
        llm_model=required_env_vars["LOCAL_LLM_MODEL"],
        device=required_env_vars["DEVICE"],
        use_quantization=required_env_vars["USE_QUANTIZATION"].lower() == "true",
        quantization_bits=int(required_env_vars["QUANTIZATION_BITS"]),
        max_new_tokens=int(required_env_vars["MAX_NEW_TOKENS"]),
        temperature=float(required_env_vars["TEMPERATURE"]),
        top_k=int(required_env_vars["TOP_K"]),
        top_k_chunks=int(required_env_vars["TOP_K_CHUNKS"])
    )
    
    print(f"📋 Configuration loaded from .env:")
    print(f"  Embedding Model: {config.embedding_model}")
    print(f"  LLM Model: {config.llm_model}")
    print(f"  Device: {config.device}")
    print(f"  Quantization: {config.use_quantization} ({config.quantization_bits}-bit)")
    
    # Initialize RAG service
    rag_service = LocalRAGService(config)
    
    try:
        print("📚 Initializing Local RAG Service...")
        await rag_service.initialize()
        
        # Test documents
        test_chunks = [
            "The health insurance policy covers medical expenses up to $10,000 per year.",
            "Deductible amount is $500 per claim for outpatient services.",
            "Emergency room visits are covered 100% after deductible.",
            "Prescription drugs are covered with 20% co-pay.",
            "The policy renewal period is 12 months from the date of purchase.",
            "Pre-existing conditions are covered after 24 months waiting period.",
            "Dental coverage includes cleaning and basic procedures up to $1,500 annually."
        ]
        
        print(f"🔍 Building vector store with {len(test_chunks)} chunks...")
        await rag_service.build_vector_store(test_chunks)
        
        # Test questions
        test_questions = [
            "What is the annual coverage limit?",
            "How much is the deductible?",
            "Are emergency room visits covered?",
            "What is the waiting period for pre-existing conditions?"
        ]
        
        print("❓ Testing question answering...")
        for i, question in enumerate(test_questions, 1):
            print(f"\n📝 Question {i}: {question}")
            
            try:
                answer = await rag_service.answer_question(question)
                print(f"💡 Answer: {answer}")
            except Exception as e:
                print(f"❌ Error: {e}")
        
        # System info
        print("\n📊 System Information:")
        info = rag_service.get_system_info()
        for key, value in info.items():
            print(f"  {key}: {value}")
            
    except Exception as e:
        logger.error(f"Test failed: {e}")
        print(f"❌ Test failed: {e}")
        
        # Provide helpful error messages
        if "401" in str(e) or "authentication" in str(e).lower():
            print("\n💡 This looks like an authentication error.")
            print("Make sure you:")
            print("1. Have a valid Hugging Face token")
            print("2. Accepted the Llama license at: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct")
            print("3. Run: python setup_hf_token.py --test")
        
    finally:
        # Cleanup
        rag_service.cleanup()
        print("\n✅ Test completed and cleaned up!")

if __name__ == "__main__":
    asyncio.run(test_local_rag())
