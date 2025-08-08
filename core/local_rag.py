"""
Local RAG Service - Pure Local Inference
No cloud APIs, everything runs locally
"""

import torch
import numpy as np
import faiss
import re
import logging
import asyncio
import os
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

# Load environment variables first
from dotenv import load_dotenv
load_dotenv()

# Set HF_HOME from environment variable if provided
if os.getenv('HF_HOME'):
    os.environ['HF_HOME'] = os.getenv('HF_HOME')
    os.environ['TRANSFORMERS_CACHE'] = os.getenv('HF_HOME')
    os.environ['HUGGINGFACE_HUB_CACHE'] = os.getenv('HF_HOME')
    print(f"🗂️ Using Hugging Face cache directory: {os.getenv('HF_HOME')}")

# Local inference imports
try:
    from transformers import (
        AutoTokenizer, AutoModel, AutoModelForCausalLM,
        BitsAndBytesConfig, pipeline
    )
    from sentence_transformers import SentenceTransformer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.error("Transformers not available. Install with: pip install transformers sentence-transformers")

logger = logging.getLogger(__name__)

@dataclass
class LocalModelConfig:
    """Configuration for local models - Using your requested models"""
    # Your requested models
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"    # Qwen3 embedding model
    llm_model: str = "meta-llama/Llama-3.1-8B-Instruct"   # Llama 3.1 for excellent QA
    
    # Hardware settings
    device: str = "auto"  # auto, cuda, cpu
    use_quantization: bool = True   # Enable quantization for large models
    quantization_bits: int = 8      # 8-bit quantization to fit in your VRAM
    
    # Generation settings (optimized for Llama-3.1)
    max_new_tokens: int = 256
    temperature: float = 0.2        # Lower for more focused answers
    top_p: float = 0.9
    top_k: int = 40
    do_sample: bool = True
    
    # RAG settings
    top_k_chunks: int = 5
    embedding_dim: int = 1024  # Qwen3-Embedding-0.6B actual dimension


class LocalEmbeddingService:
    """Local embedding service using SentenceTransformers"""
    
    def __init__(self, config: LocalModelConfig):
        self.config = config
        self.device = self._determine_device()
        self.model = None
        self.is_loaded = False
        
    def _determine_device(self) -> str:
        """Determine the best device to use"""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            return "cpu"
        return self.config.device
    
    async def load_model(self) -> bool:
        """Load the embedding model"""
        if self.is_loaded:
            return True
            
        if not TRANSFORMERS_AVAILABLE:
            logger.error("Transformers not available for local inference")
            return False
            
        try:
            logger.info(f"Loading embedding model: {self.config.embedding_model} on {self.device}")
            
            # Get Hugging Face token from environment
            hf_token = os.getenv('HUGGINGFACE_TOKEN')
            if not hf_token or hf_token == 'your_hf_token_here':
                logger.warning("No Hugging Face token found. Some models may not be accessible.")
                hf_token = None
            
            # Use SentenceTransformer for Qwen3-Embedding with proper configuration
            model_kwargs = {
                "attn_implementation": "flash_attention_2" if torch.cuda.is_available() and self.device == "cuda" else None,
                "device_map": "auto" if self.device == "cuda" else None,
            }
            # Remove None values
            model_kwargs = {k: v for k, v in model_kwargs.items() if v is not None}
            
            tokenizer_kwargs = {
                "padding_side": "left"  # Recommended for Qwen3-Embedding
            }
            
            self.model = SentenceTransformer(
                self.config.embedding_model,
                device=self.device,
                model_kwargs=model_kwargs,
                tokenizer_kwargs=tokenizer_kwargs,
                trust_remote_code=True,
                token=hf_token
            )
            
            self.is_loaded = True
            logger.info(f"Embedding model loaded successfully on {self.device}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            self.is_loaded = False
            return False
    
    async def create_embeddings(self, texts: List[str], is_query: bool = False) -> np.ndarray:
        """Create embeddings for texts
        
        Args:
            texts: List of texts to embed
            is_query: If True, uses query prompt for better search performance
        """
        if not self.is_loaded:
            success = await self.load_model()
            if not success:
                raise Exception("Failed to load embedding model")
        
        try:
            logger.info(f"Creating embeddings for {len(texts)} texts on {self.device} (query={is_query})")
            
            # Use query prompt for search queries as recommended by Qwen3-Embedding
            encode_kwargs = {
                "convert_to_numpy": True, 
                "show_progress_bar": False
            }
            
            if is_query:
                encode_kwargs["prompt_name"] = "query"
            
            # Run in thread pool for CPU, direct for GPU
            if self.device == "cpu":
                loop = asyncio.get_event_loop()
                with ThreadPoolExecutor(max_workers=1) as executor:
                    embeddings = await loop.run_in_executor(
                        executor, 
                        lambda: self.model.encode(texts, **encode_kwargs)
                    )
            else:
                embeddings = self.model.encode(texts, **encode_kwargs)
            
            logger.info(f"Created embeddings with shape: {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error creating local embeddings: {e}")
            raise Exception(f"Local embedding error: {e}")
    
    def unload_model(self):
        """Unload model to free memory"""
        if self.model:
            del self.model
            self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.is_loaded = False
        logger.info("Embedding model unloaded")


class LocalLLMService:
    """Local LLM service using Transformers"""
    
    def __init__(self, config: LocalModelConfig):
        self.config = config
        self.device = self._determine_device()
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.is_loaded = False
        
    def _determine_device(self) -> str:
        """Determine the best device to use"""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            return "cpu"
        return self.config.device
    
    async def load_model(self) -> bool:
        """Load the LLM model"""
        if self.is_loaded:
            return True
            
        if not TRANSFORMERS_AVAILABLE:
            logger.error("Transformers not available for local inference")
            return False
            
        try:
            logger.info(f"Loading LLM model: {self.config.llm_model} on {self.device}")
            
            # Get Hugging Face token from environment
            hf_token = os.getenv('HUGGINGFACE_TOKEN')
            if not hf_token or hf_token == 'your_hf_token_here':
                logger.warning("No Hugging Face token found. Llama models require authentication.")
                hf_token = None
            
            # Configure quantization for GPU
            quantization_config = None
            if self.device == "cuda" and self.config.use_quantization:
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True if self.config.quantization_bits == 8 else False,
                    load_in_4bit=True if self.config.quantization_bits == 4 else False,
                )
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.llm_model,
                trust_remote_code=True,
                token=hf_token
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.llm_model,
                quantization_config=quantization_config,
                device_map="auto" if self.device == "cuda" else None,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                token=hf_token
            )
            
            if self.device == "cpu":
                self.model = self.model.to("cpu")
            
            # Create pipeline for easier inference
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            
            self.is_loaded = True
            logger.info(f"LLM model loaded successfully on {self.device}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load LLM model: {e}")
            self.is_loaded = False
            return False
    
    async def generate_answer(self, prompt: str) -> str:
        """Generate answer using local LLM"""
        if not self.is_loaded:
            success = await self.load_model()
            if not success:
                raise Exception("Failed to load LLM model")
        
        try:
            logger.info(f"Generating answer on {self.device}")
            
            # Configure generation parameters
            generation_params = {
                "max_new_tokens": self.config.max_new_tokens,
                "temperature": self.config.temperature,
                "do_sample": True,
                "top_p": self.config.top_p,
                "top_k": self.config.top_k,
                "repetition_penalty": 1.1,
                "pad_token_id": self.tokenizer.eos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "return_full_text": False  # Only return generated text
            }
            
            # Run in thread pool for CPU
            if self.device == "cpu":
                loop = asyncio.get_event_loop()
                with ThreadPoolExecutor(max_workers=1) as executor:
                    result = await loop.run_in_executor(
                        executor,
                        lambda: self.pipeline(prompt, **generation_params)
                    )
            else:
                result = self.pipeline(prompt, **generation_params)
            
            # Extract generated text
            if isinstance(result, list) and len(result) > 0:
                generated_text = result[0]['generated_text']
            else:
                generated_text = str(result)
            
            logger.info("Answer generated successfully")
            return generated_text.strip()
            
        except Exception as e:
            logger.error(f"Error generating local answer: {e}")
            raise Exception(f"Local LLM error: {e}")
    
    def unload_model(self):
        """Unload model to free memory"""
        if self.pipeline:
            del self.pipeline
            self.pipeline = None
        if self.model:
            del self.model
            self.model = None
        if self.tokenizer:
            del self.tokenizer
            self.tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.is_loaded = False
        logger.info("LLM model unloaded")


class LocalRAGService:
    """Local RAG Service - Pure local inference"""
    
    def __init__(self, config: Optional[LocalModelConfig] = None):
        self.config = config or LocalModelConfig()
        
        # Initialize services
        self.embedding_service = LocalEmbeddingService(self.config)
        self.llm_service = LocalLLMService(self.config)
        
        # Vector store
        self.vector_store = None
        self.chunks = []
        
        logger.info(f"Local RAG Service initialized with device: {self.config.device}")
    
    async def initialize(self):
        """Initialize all services"""
        logger.info("Initializing Local RAG Service...")
        
        # Load embedding model first (lighter)
        embedding_success = await self.embedding_service.load_model()
        if not embedding_success:
            raise Exception("Failed to initialize embedding service")
        
        logger.info("Local RAG Service initialized successfully")
        return True
    
    async def build_vector_store(self, chunks: List[str]) -> None:
        """Build FAISS vector store from document chunks"""
        try:
            logger.info(f"Building vector store from {len(chunks)} chunks")
            self.chunks = chunks
            
            # Create embeddings for all chunks
            chunk_embeddings = await self.embedding_service.create_embeddings(chunks)
            
            # Build FAISS index
            embedding_dim = chunk_embeddings.shape[1]
            self.config.embedding_dim = embedding_dim  # Update config with actual dimension
            
            self.vector_store = faiss.IndexFlatIP(embedding_dim)
            faiss.normalize_L2(chunk_embeddings)
            self.vector_store.add(chunk_embeddings)
            
            logger.info(f"Vector store built with {len(chunks)} chunks, embedding dim: {embedding_dim}")
        except Exception as e:
            raise Exception(f"Error building vector store: {str(e)}")
    
    async def retrieve_relevant_chunks(self, query: str) -> List[Dict]:
        """Retrieve top-k relevant chunks for a query"""
        try:
            if not self.vector_store or not self.chunks:
                raise ValueError("Vector store not initialized. Call build_vector_store first.")
            
            logger.info(f"Retrieving relevant chunks for query: {query[:50]}...")
            
            # Create query embedding (use query prompt for better retrieval)
            query_embedding = await self.embedding_service.create_embeddings([query], is_query=True)
            faiss.normalize_L2(query_embedding)
            
            # Search in vector store
            scores, indices = self.vector_store.search(query_embedding, self.config.top_k_chunks)
            
            # Format results
            results = [
                {"chunk": self.chunks[idx], "score": float(score), "index": int(idx)}
                for score, idx in zip(scores[0], indices[0]) if idx < len(self.chunks)
            ]
            
            logger.info(f"Retrieved {len(results)} relevant chunks")
            return results
        except Exception as e:
            raise Exception(f"Error retrieving relevant chunks: {str(e)}")
    
    def _create_prompt(self, query: str, context: str) -> str:
        """Create prompt for Llama-3.1-8B-Instruct (uses specific instruction format)"""
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant that answers questions based on the provided context. Use only the information in the context to answer the question. If the answer cannot be found in the context, say so clearly.

<|eot_id|><|start_header_id|>user<|end_header_id|>

Context:
{context}

Question: {query}

<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        return prompt
    
    def _clean_answer(self, raw_answer: str) -> str:
        """Clean and format the generated answer"""
        try:
            if not raw_answer or not raw_answer.strip():
                return "I couldn't generate a response. Please try rephrasing your question."
            
            answer = raw_answer.strip()
            
            # Remove common prefixes
            answer = re.sub(r'^(ANSWER|Answer):\s*', '', answer, flags=re.IGNORECASE)
            answer = re.sub(r'^(Based on the provided|According to the|From the document)', '', answer, flags=re.IGNORECASE).strip()
            
            # Clean formatting
            answer = re.sub(r'\*\*(.*?)\*\*', r'\1', answer)  # Remove bold
            answer = re.sub(r'\*(.*?)\*', r'\1', answer)      # Remove italic
            answer = re.sub(r'`(.*?)`', r'\1', answer)        # Remove code blocks
            
            # Clean spacing
            answer = re.sub(r'\s{2,}', ' ', answer)
            answer = re.sub(r'\s+([.,;:!?])', r'\1', answer)
            
            # Ensure proper capitalization
            if answer and not answer[0].isupper():
                answer = answer[0].upper() + answer[1:]
            
            # Ensure proper ending
            if answer and answer[-1] not in '.!?':
                answer += '.'
            
            answer = answer.strip()
            
            # Check if answer is too short
            if len(answer) < 10:
                return "The information requested could not be found in the provided document sections."
            
            return answer
            
        except Exception as e:
            logger.error(f"Error cleaning answer: {str(e)}")
            return raw_answer.strip() if raw_answer else "Unable to process the response."
    
    async def answer_question(self, query: str) -> str:
        """Answer a question using local RAG"""
        try:
            # Retrieve relevant chunks
            relevant_chunks = await self.retrieve_relevant_chunks(query)
            
            # Create context from chunks
            context = "\n\n".join([
                f"[Document Section {i+1}]:\n{chunk['chunk']}" 
                for i, chunk in enumerate(relevant_chunks)
            ])
            
            # Create prompt
            prompt = self._create_prompt(query, context)
            
            # Load LLM if not loaded (lazy loading)
            if not self.llm_service.is_loaded:
                logger.info("Loading LLM model for answer generation...")
                llm_success = await self.llm_service.load_model()
                if not llm_success:
                    raise Exception("Failed to load LLM model")
            
            # Generate answer
            raw_answer = await self.llm_service.generate_answer(prompt)
            
            # Clean and return answer
            clean_answer = self._clean_answer(raw_answer)
            
            logger.info("Question answered successfully")
            return clean_answer
            
        except Exception as e:
            logger.error(f"Error answering question '{query}': {str(e)}")
            return f"Sorry, I couldn't answer this question due to an error: {str(e)}"
    
    async def answer_multiple_questions(self, questions: List[str]) -> List[str]:
        """Answer multiple questions efficiently"""
        try:
            logger.info(f"Processing {len(questions)} questions locally")
            
            # Process questions sequentially to manage memory
            answers = []
            for i, question in enumerate(questions):
                logger.info(f"Processing question {i+1}/{len(questions)}")
                answer = await self.answer_question(question)
                answers.append(answer)
            
            return answers
            
        except Exception as e:
            logger.error(f"Error processing multiple questions: {str(e)}")
            return [f"Error processing questions: {str(e)}" for _ in questions]
    
    def get_system_info(self) -> Dict:
        """Get system information"""
        return {
            "service": "Local RAG",
            "embedding_model": self.config.embedding_model,
            "llm_model": self.config.llm_model,
            "device": self.config.device,
            "use_quantization": self.config.use_quantization,
            "embedding_loaded": self.embedding_service.is_loaded,
            "llm_loaded": self.llm_service.is_loaded,
            "vector_store_ready": self.vector_store is not None,
            "num_chunks": len(self.chunks),
            "cuda_available": torch.cuda.is_available(),
        }
    
    def cleanup(self):
        """Clean up all loaded models"""
        logger.info("Cleaning up Local RAG Service...")
        self.embedding_service.unload_model()
        self.llm_service.unload_model()
        self.vector_store = None
        self.chunks = []
        logger.info("Local RAG Service cleaned up")


def create_local_rag_service(config: Optional[LocalModelConfig] = None) -> LocalRAGService:
    """Factory function to create local RAG service"""
    return LocalRAGService(config)
