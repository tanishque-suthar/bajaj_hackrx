import google.generativeai as genai
import numpy as np
import faiss
import re
from typing import List, Dict
import logging
import os
from dotenv import load_dotenv
import asyncio
from sentence_transformers import SentenceTransformer, CrossEncoder
import torch
from core.api_key_manager import GeminiAPIKeyManager, ModelManager
load_dotenv()
logger = logging.getLogger(__name__)

class RAGService:
    """Hybrid RAG service using local embeddings/reranking and Gemini for generation"""
    
    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_k: int = 8,
        rerank_top_k: int = 4,
        gemma_model: str = "gemini-1.5-flash",
        device: str = "cpu",
        enable_reranking: bool = True,  # Option to disable reranking entirely
        rerank_batch_size: int = 4      # Smaller batches for CPU
    ):
        self.top_k = top_k
        self.rerank_top_k = max(rerank_top_k, top_k)  # Retrieve more for reranking
        self.gemma_model = gemma_model
        self.embedding_model = embedding_model
        self.rerank_model = rerank_model
        self.device = device
        self.enable_reranking = enable_reranking
        self.rerank_batch_size = rerank_batch_size
        
        # Set device based on availability (optimized for i5 10th gen + integrated GPU)
        if torch.cuda.is_available():
            self.device = "cuda"
            logger.info("CUDA available, using GPU")
        else:
            self.device = "cpu"
            logger.info("Using CPU for local inference")
        
        # Initialize local models (lazy loading)
        self.embedding_model_instance = None
        self.rerank_model_instance = None
        
        # --- Initialize Gemini API Key Manager (keep only this for generation) ---
        try:
            self.gemini_key_manager = GeminiAPIKeyManager()
            logger.info(f"Initialized Gemini API Key Manager with {len(self.gemini_key_manager.api_keys)} keys")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini API Key Manager: {str(e)}")
            # Fallback to single key approach
            google_api_key = os.getenv("GOOGLE_API_KEY")
            if not google_api_key:
                raise ValueError("Neither GEMINI_API_TOKENS nor GOOGLE_API_KEY found in environment variables")
            self.gemini_key_manager = None
            logger.warning("Using fallback single Gemini API key")
        
        # --- Initialize Model Manager ---
        self.gemini_models = [
            "gemini-1.5-flash",
            "gemma-3-12b-it", 
            "gemma-3-4b-it"
        ]
        self.model_manager = ModelManager(self.gemini_models, rate_limit_cooldown=120)
        
        # Determine embedding dimensions based on model
        self.embedding_dim = self._get_embedding_dimension()
        
        # --- Initialize Google AI Generative Model (Lazy Loading) ---
        self.model = None
        self.generation_config = None
        
        self.vector_store = None
        self.chunks = []

    def _get_embedding_dimension(self) -> int:
        """Get embedding dimension based on model name"""
        model_dims = {
            "sentence-transformers/all-MiniLM-L6-v2": 384,
            "sentence-transformers/all-mpnet-base-v2": 768,
            "sentence-transformers/e5-base-v2": 768,
            "sentence-transformers/paraphrase-MiniLM-L6-v2": 384,
        }
        return model_dims.get(self.embedding_model, 384)  # Default to 384

    def _initialize_local_models(self):
        """Initialize local embedding and reranking models lazily"""
        if self.embedding_model_instance is None:
            logger.info(f"Loading local embedding model: {self.embedding_model}")
            try:
                # For i5 10th gen, use CPU with optimized settings
                self.embedding_model_instance = SentenceTransformer(
                    self.embedding_model, 
                    device=self.device
                )
                # Enable optimizations for CPU inference
                if self.device == "cpu":
                    self.embedding_model_instance.eval()
                    torch.set_num_threads(4)  # Optimize for i5 quad-core
                
                logger.info(f"✅ Embedding model loaded on {self.device}")
            except Exception as e:
                logger.error(f"Failed to load embedding model: {str(e)}")
                raise
        
        if self.rerank_model_instance is None:
            logger.info(f"Loading local reranking model: {self.rerank_model}")
            try:
                self.rerank_model_instance = CrossEncoder(
                    self.rerank_model, 
                    device=self.device
                )
                logger.info(f"✅ Reranking model loaded on {self.device}")
            except Exception as e:
                logger.error(f"Failed to load reranking model: {str(e)}")
                raise

    def _initialize_gemini_model(self):
        """Initialize Gemini model lazily with fallback support"""
        if self.model is None:
            if not self.gemini_key_manager:
                # Fallback to single key
                google_api_key = os.getenv("GOOGLE_API_KEY")
                if not google_api_key:
                    raise ValueError("GOOGLE_API_KEY not found in environment variables")
                genai.configure(api_key=google_api_key)
                self.model = genai.GenerativeModel(self.gemma_model)
            else:
                # Will be configured per request with different API keys
                # Initial configuration with first available key
                initial_key = self.gemini_key_manager.get_next_available_key()
                if initial_key:
                    genai.configure(api_key=initial_key)
                    self.model = genai.GenerativeModel(self.gemma_model)
                else:
                    raise ValueError("No available Gemini API keys")
            
            self.generation_config = genai.types.GenerationConfig(
                temperature=0.1,
                top_p=1.0,
                top_k=12,
                max_output_tokens=200,
            )

    async def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """Create embeddings using local SentenceTransformer model"""
        try:
            self._initialize_local_models()
            
            logger.info(f"Creating local embeddings for {len(texts)} texts using {self.embedding_model}")
            
            # Use asyncio to run CPU-intensive embedding creation in thread pool
            loop = asyncio.get_event_loop()
            
            def _create_embeddings_sync():
                with torch.no_grad():  # Save memory and speed up inference
                    embeddings = self.embedding_model_instance.encode(
                        texts,
                        convert_to_numpy=True,
                        show_progress_bar=len(texts) > 10,  # Show progress for large batches
                        batch_size=32 if self.device == "cpu" else 64,  # Optimize batch size for CPU
                        normalize_embeddings=True  # Normalize for cosine similarity
                    )
                return embeddings
            
            # Run in thread pool to avoid blocking the event loop
            embeddings_array = await loop.run_in_executor(None, _create_embeddings_sync)
            
            logger.info(f"✅ Created local embeddings with shape: {embeddings_array.shape}")
            return embeddings_array.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Error creating local embeddings: {str(e)}")
            raise Exception(f"Error creating local embeddings: {str(e)}")

    async def build_vector_store(self, chunks: List[str]) -> None:
        """Build FAISS vector store from document chunks using local embeddings"""
        try:
            logger.info(f"Building local vector store from {len(chunks)} chunks")
            self.chunks = chunks
            
            chunk_embeddings = await self.create_embeddings(chunks)
            
            # Use IndexFlatIP for cosine similarity (embeddings are already normalized)
            self.vector_store = faiss.IndexFlatIP(self.embedding_dim)
            self.vector_store.add(chunk_embeddings)
            
            logger.info(f"✅ Vector store built locally with {len(chunks)} chunks")
        except Exception as e:
            raise Exception(f"Error building vector store: {str(e)}")

    async def retrieve_relevant_chunks(self, query: str) -> List[Dict]:
        """Retrieve and optionally rerank relevant chunks using local models"""
        try:
            if not self.vector_store or not self.chunks:
                raise ValueError("Vector store not initialized. Call build_vector_store first.")
            
            logger.info(f"Retrieving relevant chunks for query: {query[:50]}...")
            
            # Step 1: Initial retrieval
            if self.enable_reranking:
                # Retrieve more candidates for reranking
                retrieve_k = self.rerank_top_k
            else:
                # Retrieve exactly what we need
                retrieve_k = self.top_k
            
            query_embedding = await self.create_embeddings([query])
            scores, indices = self.vector_store.search(query_embedding, retrieve_k)
            
            initial_results = [
                {"chunk": self.chunks[idx], "score": float(score), "index": int(idx)}
                for score, idx in zip(scores[0], indices[0]) if idx < len(self.chunks)
            ]
            
            logger.info(f"📊 Initial retrieval: {len(initial_results)} chunks")
            
            # Step 2: Optional reranking with performance optimizations
            if self.enable_reranking and len(initial_results) > self.rerank_top_k:
                logger.info("🔄 Applying fast local reranking...")
                start_time = asyncio.get_event_loop().time()
                
                # Smart reranking: only rerank if we have enough candidates
                if len(initial_results) <= 3:  # If we have 3 or fewer, skip reranking
                    logger.info("⚡ Skipping reranking (too few candidates for meaningful improvement)")
                    final_results = initial_results[:self.top_k]
                else:
                    reranked_results = await self._rerank_chunks_fast(query, initial_results)
                    final_results = reranked_results[:self.top_k]
                
                end_time = asyncio.get_event_loop().time()
                logger.info(f"✅ Reranking completed in {end_time - start_time:.2f}s, selected top {len(final_results)} chunks")
            else:
                final_results = initial_results[:self.top_k]
                if not self.enable_reranking:
                    logger.info("⚡ Reranking disabled - using semantic similarity only")
                else:
                    logger.info("⏭️ Skipping reranking (not enough candidates)")
            
            return final_results
            
        except Exception as e:
            raise Exception(f"Error retrieving relevant chunks: {str(e)}")

    async def _rerank_chunks_fast(self, query: str, chunks: List[Dict]) -> List[Dict]:
        """Fast reranking with optimizations for i5 10th gen CPU"""
        try:
            self._initialize_local_models()
            query_chunk_pairs = [(query, chunk['chunk']) for chunk in chunks]

            loop = asyncio.get_event_loop()
            
            def _rerank_sync():
                with torch.no_grad():
                    # Optimization 2: Use smaller batch size for CPU
                    rerank_scores = self.rerank_model_instance.predict(
                        query_chunk_pairs,
                        show_progress_bar=False,
                        batch_size=self.rerank_batch_size,  # Smaller batches
                        convert_to_numpy=True,  # Faster than tensor operations
                        apply_softmax=False     # Skip softmax for speed (we only need rankings)
                    )
                return rerank_scores
            
            # Run reranking in thread pool
            rerank_scores = await loop.run_in_executor(None, _rerank_sync)
            
            # Update chunks with rerank scores and sort
            for chunk, rerank_score in zip(chunks, rerank_scores):
                chunk['rerank_score'] = float(rerank_score)
            
            # Sort by rerank score (higher is better)
            reranked_chunks = sorted(chunks, key=lambda x: x['rerank_score'], reverse=True)
            
            logger.info(f"⚡ Fast-reranked {len(chunks)} chunks (batch_size={self.rerank_batch_size})")
            return reranked_chunks
            
        except Exception as e:
            logger.error(f"Error in fast reranking: {str(e)}")
            # Fallback to original retrieval scores
            logger.info("⚠️ Falling back to semantic similarity scores")
            return sorted(chunks, key=lambda x: x['score'], reverse=True)
            
    def generate_answer(self, query: str, relevant_chunks: List[Dict]) -> str:
        """Generate answer using Google Gemini with model-first fallback strategy"""
        if not self.gemini_key_manager:
            # Fallback to single client
            return self._generate_answer_single_client(query, relevant_chunks)
        
        logger.info("🤖 Generating answer using Google Gemini with model-first fallback")
        
        # Create context with rerank scores if available
        context_parts = []
        for i, chunk in enumerate(relevant_chunks):
            score_info = ""
            if 'rerank_score' in chunk:
                score_info = f" (relevance: {chunk['rerank_score']:.3f})"
            context_parts.append(f"[Document Section {i+1}{score_info}]:\n{chunk['chunk']}")
        
        context = "\n\n".join(context_parts)
        prompt = self._create_gemma_prompt(query, context)
        
        # Model-first strategy: All Keys try flash → All Keys try 12b → All Keys try 4b
        for model_name in self.gemini_models:
            # Check if this model is available
            if (model_name in self.model_manager.failed_models or 
                model_name in self.model_manager.rate_limited_models):
                logger.info(f"⏭️ Skipping model {model_name} (not available)")
                continue
                
            logger.info(f"🔄 Trying model: {model_name}")
            
            # Try this model with all available API keys
            for attempt in range(len(self.gemini_key_manager.api_keys)):
                api_key = self.gemini_key_manager.get_next_available_key()
                
                if not api_key:
                    logger.warning(f"No available API keys for model {model_name}")
                    break
                
                try:
                    # Configure Gemini with current API key and model
                    genai.configure(api_key=api_key)
                    model = genai.GenerativeModel(model_name)
                    
                    # Make API call
                    response = model.generate_content(prompt, generation_config=self.generation_config)
                    raw_answer = response.text.strip()
                    cleaned_answer = self._clean_answer(raw_answer)
                    
                    # Success - mark both key and model as successful
                    self.gemini_key_manager.mark_key_successful(api_key)
                    self.model_manager.mark_model_successful(model_name)
                    
                    logger.info(f"✅ Answer generated successfully using model: {model_name} with API key: {self.gemini_key_manager._mask_key(api_key)}")
                    return cleaned_answer
                    
                except Exception as e:
                    error_msg = str(e)
                    
                    logger.warning(f"❌ Failed with model {model_name} and key {self.gemini_key_manager._mask_key(api_key)}: {error_msg}")
                    
                    # Check if it's a model-specific rate limit error
                    if self.gemini_key_manager.is_rate_limit_error(error_msg):
                        # Could be model-specific or key-specific rate limit
                        if "model" in error_msg.lower() or model_name in error_msg.lower():
                            self.model_manager.mark_model_rate_limited(model_name)
                            logger.info(f"🔄 Model {model_name} rate limited, trying next model")
                            break  # Try next model
                        else:
                            self.gemini_key_manager.mark_key_rate_limited(api_key)
                            logger.info(f"🔄 API key rate limited, trying next key with same model")
                            continue  # Try next key with same model
                    
                    # Check if it's an authentication error
                    elif self.gemini_key_manager.is_auth_error(error_msg):
                        self.gemini_key_manager.mark_key_failed(api_key)
                        logger.warning(f"🚫 Authentication error, marking key as failed and trying next key")
                        continue
                    
                    # Check if it's a model-specific error
                    elif any(keyword in error_msg.lower() for keyword in ['model not found', 'invalid model', 'model unavailable']):
                        self.model_manager.mark_model_failed(model_name)
                        logger.warning(f"🚫 Model {model_name} failed permanently, trying next model")
                        break  # Try next model
                    
                    # For other errors, continue with next key
                    else:
                        logger.warning(f"⚠️ Unknown error, trying next key: {error_msg}")
                        continue
        
        # All models and keys exhausted
        key_stats = self.gemini_key_manager.get_statistics()
        model_stats = self.model_manager.get_statistics()
        logger.error(f"🔴 All Gemini models and API keys exhausted. Key stats: {key_stats}, Model stats: {model_stats}")
        
        return "Sorry, all Gemini models and API keys are currently unavailable. Please try again later."
    
    def _generate_answer_single_client(self, query: str, relevant_chunks: List[Dict]) -> str:
        """Fallback method for single client (backward compatibility)"""
        try:
            # Initialize Gemini model if not already done
            self._initialize_gemini_model()
            
            logger.info("Generating answer using single Gemini client")
            context = "\n\n".join([
                f"[Document Section {i+1}]:\n{chunk['chunk']}" 
                for i, chunk in enumerate(relevant_chunks)
            ])
            prompt = self._create_gemma_prompt(query, context)
            response = self.model.generate_content(prompt, generation_config=self.generation_config)
            raw_answer = response.text.strip()
            cleaned_answer = self._clean_answer(raw_answer)
            logger.info("Answer generated and cleaned successfully using Gemini")
            return cleaned_answer
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return f"Sorry, I couldn't generate an answer: {str(e)}"

    def _create_gemma_prompt(self, query: str, context: str) -> str:
        """Create prompt for Gemini model"""
        prompt = f"""You are a precision-focused AI analyst. Your task is to answer the question with extreme accuracy based ONLY on the provided POLICY SECTIONS.

POLICY SECTIONS:
{context}

QUESTION: {query}

INSTRUCTIONS:
1. The sections are ordered by relevance - prioritize information from higher-ranked sections.
2. Extract specific facts, such as time periods (e.g., 30 days, 24 months), monetary values, percentages, and conditions.
3. You must cite your source at the end of the answer, like this: (Source: SECTION 1).
4. If multiple sections provide related information, synthesize them coherently.
5. Your answer must be a maximum of three (3) sentences.
6. If the answer cannot be found, respond with exactly: "The answer cannot be found in the provided policy sections."
7. Answer in clean, continuous prose without bullet points or formatting marks.

ANSWER:"""
        return prompt

    def _clean_answer(self, raw_answer: str) -> str:
        """Clean and format the generated answer"""
        try:
            if not raw_answer or not raw_answer.strip():
                return "I couldn't generate a response. Please try rephrasing your question."
            answer = raw_answer.strip()
            answer = re.sub(r'^(ANSWER|Answer):\s*', '', answer, flags=re.IGNORECASE)
            answer = re.sub(r'\*\*(.*?)\*\*', r'\1', answer)
            answer = re.sub(r'\*(.*?)\*', r'\1', answer)
            answer = re.sub(r'`(.*?)`', r'\1', answer)
            answer = re.sub(r'[^\w\s\.,;:!?()\-\%\$\@\&\#\*\+\=\[\]\{\}<>/"\']', ' ', answer)
            answer = re.sub(r'\s+([.,;:!?])', r'\1', answer)
            answer = re.sub(r'([.,;:!?])\s*([A-Z])', r'\1 \2', answer)
            answer = re.sub(r'\s{2,}', ' ', answer)
            answer = re.sub(r'^(Based on the provided|According to the|From the document|The document states|In the policy)', '', answer, flags=re.IGNORECASE).strip()
            if answer and not answer[0].isupper():
                answer = answer[0].upper() + answer[1:]
            if answer and answer[-1] not in '.!?':
                answer += '.'
            answer = answer.strip()
            if len(answer) < 10:
                return "The information requested could not be found in the provided document sections."
            return answer
        except Exception as e:
            logger.error(f"Error cleaning answer: {str(e)}")
            return raw_answer.strip() if raw_answer else "Unable to process the response."

    async def answer_question(self, query: str, chunks: List[str]) -> str:
        """Main method to answer a single question"""
        try:
            relevant_chunks = await self.retrieve_relevant_chunks(query)
            answer = self.generate_answer(query, relevant_chunks)
            return answer
        except Exception as e:
            logger.error(f"Error answering question '{query}': {str(e)}")
            return f"Sorry, I couldn't answer this question due to an error: {str(e)}"

    async def answer_multiple_questions(self, questions: List[str], chunks: List[str]) -> List[str]:
        """Answer multiple questions efficiently"""
        try:
            logger.info(f"Processing {len(questions)} questions with Gemini")
            # Build vector store once for all questions
            await self.build_vector_store(chunks)
            
            tasks = [self.answer_question(q, chunks) for q in questions]
            answers = await asyncio.gather(*tasks, return_exceptions=True)
            
            final_answers = []
            for i, result in enumerate(answers):
                if isinstance(result, Exception):
                    logger.error(f"Error processing question {i+1}: {str(result)}")
                    final_answers.append("Sorry, an error occurred for this question.")
                else:
                    final_answers.append(result)
            return final_answers
        except Exception as e:
            logger.error(f"Error processing multiple questions: {str(e)}")
            return [f"Error processing questions: {str(e)}" for _ in questions]

    def get_api_key_statistics(self) -> Dict:
        """Get statistics about API key usage and model usage"""
        stats = {}
        
        # Local model information
        stats["local_models"] = {
            "embedding_model": self.embedding_model,
            "rerank_model": self.rerank_model,
            "device": self.device,
            "embedding_dim": self.embedding_dim,
            "models_loaded": {
                "embeddings": self.embedding_model_instance is not None,
                "reranking": self.rerank_model_instance is not None
            }
        }
        
        # Gemini API key statistics
        if self.gemini_key_manager:
            stats["gemini_keys"] = self.gemini_key_manager.get_statistics()
        else:
            stats["gemini_keys"] = {
                "service": "Gemini",
                "mode": "single_key_fallback", 
                "message": "Using single GOOGLE_API_KEY (no multi-key manager)"
            }
        
        # Gemini model statistics
        stats["gemini_models"] = self.model_manager.get_statistics()
        
        return stats
