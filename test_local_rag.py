"""
Test script for Local RAG Service
Tests Qwen/Qwen3-Embedding-0.6B + meta-llama/Llama-3.1-8B-Instruct
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
    
    # Create config with your requested models
    config = LocalModelConfig(
        embedding_model="Qwen/Qwen3-Embedding-0.6B",
        llm_model="meta-llama/Llama-3.1-8B-Instruct",
        device="auto",  # Will use CPU for testing (128MB VRAM)
        use_quantization=True,  # Enable for large models
        quantization_bits=8,
        max_new_tokens=256,
        temperature=0.2,
        top_k_chunks=3  # Fewer chunks for testing
    )
    
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
