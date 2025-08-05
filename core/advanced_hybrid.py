"""
Advanced Hybrid RAG Service
Combines hybrid search (dense + sparse) with RAG functionality
"""

import google.generativeai as genai
import asyncio
import logging
import os
import re
import time
from typing import List, Dict, Optional
from dotenv import load_dotenv

from core.hybrid_retrieval import HybridSearchService
from core.hybrid_config import HybridSearchConfig
from core.api_key_manager import GeminiAPIKeyManager, ModelManager

load_dotenv()
logger = logging.getLogger(__name__)

class AdvancedHybridRAGService:
    """
    Advanced RAG service using hybrid search (dense + sparse vectors) for optimal retrieval
    Combines semantic and lexical search for better context retrieval
    """
    
    def __init__(
        self,
        config: Optional[HybridSearchConfig] = None,
        gemma_model: str = "gemini-1.5-flash",
        auto_cleanup: bool = True
    ):
        # Use provided config or default
        self.config = config or HybridSearchConfig.from_env()
        self.config.validate()
        
        self.gemma_model = gemma_model
        self.auto_cleanup = auto_cleanup
        
        # Initialize hybrid search service
        logger.info("Initializing Advanced Hybrid RAG Service...")
        self.hybrid_search = HybridSearchService(
            embedding_model=self.config.external_embedding_model,
            dense_model=self.config.dense_model,
            sparse_model=self.config.sparse_model,
            top_k_per_index=self.config.top_k_per_index,
            final_top_k=self.config.final_top_k,
            auto_cleanup=self.config.auto_cleanup
        )
        
        # Initialize Gemini components
        self._initialize_gemini_components()
        
        # Session management
        self.current_session_id = None
        self.chunks = []
        
        logger.info("✅ Advanced Hybrid RAG Service initialized successfully")

    def _initialize_gemini_components(self):
        """Initialize Gemini API key manager and model manager"""
        try:
            # --- Initialize Gemini API Key Manager ---
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
        
        # --- Initialize Google AI Generative Model (Lazy Loading) ---
        self.model = None
        self.generation_config = None

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

    async def build_vector_store(self, chunks: List[str]) -> None:
        """Build hybrid vector store from document chunks"""
        try:
            logger.info(f"Building hybrid vector store from {len(chunks)} chunks")
            if len(chunks) == 0:
                raise ValueError("No chunks provided to build vector store")
            
            self.chunks = chunks
            
            # Build hybrid vector store
            await self.hybrid_search.build_hybrid_vector_store(chunks)
            
            # Update session tracking
            self.current_session_id = self.hybrid_search.current_session_id
            
            # Debug: Check if data was actually uploaded
            debug_info = self.hybrid_search.debug_index_contents()
            logger.info(f"Vector store debug info: {debug_info}")
            
            logger.info(f"✅ Hybrid vector store built with {len(chunks)} chunks for session: {self.current_session_id}")
            
        except Exception as e:
            logger.error(f"Error building hybrid vector store: {str(e)}")
            raise Exception(f"Error building hybrid vector store: {str(e)}")

    async def retrieve_relevant_chunks(self, query: str) -> List[Dict]:
        """Retrieve top-k relevant chunks using hybrid search"""
        try:
            if not self.current_session_id:
                raise ValueError("Vector store not initialized. Call build_vector_store first.")
            
            logger.info(f"Retrieving relevant chunks using hybrid search for query: {query[:100]}...")
            logger.info(f"Session: {self.current_session_id}, Available chunks: {len(self.chunks)}")
            
            # Perform hybrid search
            results = await self.hybrid_search.hybrid_search(query)
            
            if len(results) == 0:
                logger.warning("No results returned from hybrid search")
                # Debug the index state
                debug_info = self.hybrid_search.debug_index_contents()
                logger.warning(f"Index debug info: {debug_info}")
            else:
                logger.info(f"✅ Retrieved {len(results)} relevant chunks using hybrid search")
                # Log first result for debugging
                if results:
                    first_result = results[0]
                    logger.info(f"Top result score: {first_result.get('rerank_score', first_result.get('score', 0)):.3f}")
                    logger.info(f"Top result text: {first_result['chunk'][:200]}...")
            
            return results
            
        except Exception as e:
            logger.error(f"Error retrieving relevant chunks: {str(e)}")
            raise Exception(f"Error retrieving relevant chunks: {str(e)}")

    def generate_answer(self, query: str, relevant_chunks: List[Dict]) -> str:
        """Generate answer using Google Gemini with model-first fallback strategy"""
        if not self.gemini_key_manager:
            # Fallback to single client
            return self._generate_answer_single_client(query, relevant_chunks)
        
        logger.info("Generating answer using Google Gemini with hybrid search context")
        
        # Enhanced context with hybrid search scores
        context_parts = []
        for i, chunk in enumerate(relevant_chunks):
            score_info = ""
            if 'rerank_score' in chunk:
                score_info = f" (Relevance: {chunk['rerank_score']:.3f})"
            
            context_parts.append(
                f"[Document Section {i+1}{score_info}]:\n{chunk['chunk']}"
            )
        
        context = "\n\n".join(context_parts)
        prompt = self._create_hybrid_rag_prompt(query, context)
        
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
                    
                    logger.info(f"✅ Answer generated successfully using model: {model_name}")
                    return cleaned_answer
                    
                except Exception as e:
                    error_msg = str(e)
                    
                    logger.warning(f"❌ Failed with model {model_name}: {error_msg}")
                    
                    # Handle different types of errors
                    if self.gemini_key_manager.is_rate_limit_error(error_msg):
                        if "model" in error_msg.lower() or model_name in error_msg.lower():
                            self.model_manager.mark_model_rate_limited(model_name)
                            break  # Try next model
                        else:
                            self.gemini_key_manager.mark_key_rate_limited(api_key)
                            continue  # Try next key with same model
                    elif self.gemini_key_manager.is_auth_error(error_msg):
                        self.gemini_key_manager.mark_key_failed(api_key)
                        continue
                    elif any(keyword in error_msg.lower() for keyword in ['model not found', 'invalid model', 'model unavailable']):
                        self.model_manager.mark_model_failed(model_name)
                        break  # Try next model
                    else:
                        continue
        
        # All models and keys exhausted
        logger.error("🔴 All Gemini models and API keys exhausted")
        return "Sorry, all Gemini models and API keys are currently unavailable. Please try again later."

    def _generate_answer_single_client(self, query: str, relevant_chunks: List[Dict]) -> str:
        """Fallback method for single client (backward compatibility)"""
        try:
            # Initialize Gemini model if not already done
            self._initialize_gemini_model()
            
            logger.info("Generating answer using single Gemini client with hybrid search context")
            
            # Enhanced context with hybrid search scores
            context_parts = []
            for i, chunk in enumerate(relevant_chunks):
                score_info = ""
                if 'rerank_score' in chunk:
                    score_info = f" (Relevance: {chunk['rerank_score']:.3f})"
                
                context_parts.append(
                    f"[Document Section {i+1}{score_info}]:\n{chunk['chunk']}"
                )
            
            context = "\n\n".join(context_parts)
            prompt = self._create_hybrid_rag_prompt(query, context)
            
            response = self.model.generate_content(prompt, generation_config=self.generation_config)
            raw_answer = response.text.strip()
            cleaned_answer = self._clean_answer(raw_answer)
            
            logger.info("✅ Answer generated successfully using single Gemini client")
            return cleaned_answer
            
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return f"Sorry, I couldn't generate an answer: {str(e)}"

    def _create_hybrid_rag_prompt(self, query: str, context: str) -> str:
        """Create enhanced prompt for hybrid RAG"""
        prompt = f"""You are an expert insurance policy analyst with access to advanced hybrid search results. Your task is to provide a precise and factual answer based ONLY on the provided policy sections, which have been retrieved using both semantic and lexical search for maximum relevance.

POLICY SECTIONS (Retrieved via Hybrid Search):
{context}

QUESTION: {query}

INSTRUCTIONS:
1. Answer the question using only the information from the 'POLICY SECTIONS' above.
2. The sections are ordered by relevance - prioritize information from higher-ranked sections.
3. Extract specific facts, such as time periods (e.g., 30 days, 24 months), monetary values, percentages, and conditions.
4. If multiple sections provide related information, synthesize them coherently.
5. If the answer cannot be found in the provided sections, respond with exactly: "The answer cannot be found in the provided policy sections."
6. Be direct and concise while being comprehensive.
7. Answer in clean, continuous prose without bullet points or formatting marks.

ANSWER:"""
        return prompt

    def _clean_answer(self, raw_answer: str) -> str:
        """Clean and format the generated answer"""
        try:
            if not raw_answer or not raw_answer.strip():
                return "I couldn't generate a response. Please try rephrasing your question."
            
            answer = raw_answer.strip()
            
            # Remove common prefixes
            answer = re.sub(r'^(ANSWER|Answer):\s*', '', answer, flags=re.IGNORECASE)
            
            # Remove markdown formatting
            answer = re.sub(r'\*\*(.*?)\*\*', r'\1', answer)
            answer = re.sub(r'\*(.*?)\*', r'\1', answer)
            answer = re.sub(r'`(.*?)`', r'\1', answer)
            
            # Clean up special characters (but preserve important punctuation)
            answer = re.sub(r'[^\w\s\.,;:!?()\-\%\$\@\&\#\*\+\=\[\]\{\}<>/"\']', ' ', answer)
            
            # Fix spacing around punctuation
            answer = re.sub(r'\s+([.,;:!?])', r'\1', answer)
            answer = re.sub(r'([.,;:!?])\s*([A-Z])', r'\1 \2', answer)
            answer = re.sub(r'\s{2,}', ' ', answer)
            
            # Remove common response prefixes
            answer = re.sub(r'^(Based on the provided|According to the|From the document|The document states|In the policy)', '', answer, flags=re.IGNORECASE).strip()
            
            # Ensure proper capitalization
            if answer and not answer[0].isupper():
                answer = answer[0].upper() + answer[1:]
            
            # Ensure proper ending punctuation
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
        """Answer a single question using hybrid search and RAG"""
        try:
            # Retrieve relevant chunks using hybrid search
            relevant_chunks = await self.retrieve_relevant_chunks(query)
            
            # Generate answer
            answer = self.generate_answer(query, relevant_chunks)
            
            return answer
            
        except Exception as e:
            logger.error(f"Error answering question '{query}': {str(e)}")
            return f"Sorry, I couldn't answer this question due to an error: {str(e)}"

    async def answer_multiple_questions(self, questions: List[str], chunks: List[str]) -> List[str]:
        """Answer multiple questions efficiently with automatic cleanup"""
        try:
            logger.info("=" * 80)
            logger.info(f"🚀 PROCESSING {len(questions)} QUESTIONS WITH ADVANCED HYBRID RAG")
            logger.info("=" * 80)
            logger.info(f"📚 Document chunks: {len(chunks)}")
            logger.info(f"🔍 Search mode: Hybrid (Dense + Sparse + Reranking)")
            logger.info(f"⚙️ Config: {self.config.final_top_k} final results, reranking: {self.config.enable_reranking}")
            
            # Build hybrid vector store once for all questions
            await self.build_vector_store(chunks)
            
            # Process questions with detailed logging
            final_answers = []
            for i, question in enumerate(questions, 1):
                logger.info("-" * 60)
                logger.info(f"🔍 PROCESSING QUESTION {i}/{len(questions)}")
                logger.info(f"❓ Question: {question}")
                logger.info("-" * 60)
                
                try:
                    # Get relevant chunks
                    start_time = time.time()
                    relevant_chunks = await self.retrieve_relevant_chunks(question)
                    retrieval_time = time.time() - start_time
                    
                    logger.info(f"⏱️ Retrieval time: {retrieval_time:.2f}s")
                    logger.info(f"📄 Retrieved {len(relevant_chunks)} relevant chunks")
                    
                    # Log top retrieved chunks
                    if relevant_chunks:
                        logger.info("🔝 Top retrieved chunks:")
                        for j, chunk in enumerate(relevant_chunks[:3], 1):
                            score = chunk.get('rerank_score', chunk.get('score', 0))
                            logger.info(f"   {j}. Score: {score:.3f} - {chunk['chunk'][:150]}...")
                    
                    # Generate answer
                    start_time = time.time()
                    answer = self.generate_answer(question, relevant_chunks)
                    generation_time = time.time() - start_time
                    
                    logger.info(f"⏱️ Generation time: {generation_time:.2f}s")
                    logger.info(f"🤖 Generated Answer: {answer}")
                    
                    final_answers.append(answer)
                    
                except Exception as e:
                    logger.error(f"❌ Error processing question {i}: {str(e)}")
                    error_answer = "Sorry, an error occurred for this question."
                    final_answers.append(error_answer)
                    logger.info(f"🤖 Error Answer: {error_answer}")
            
            # Automatic cleanup after processing all questions (if enabled)
            if self.auto_cleanup:
                try:
                    logger.info("🧹 Automatically cleaning up hybrid search vectors after request completion")
                    self.cleanup_session()
                    logger.info("✅ Hybrid search vectors cleaned up successfully")
                except Exception as cleanup_error:
                    logger.warning(f"Failed to cleanup vectors (non-critical): {str(cleanup_error)}")
            else:
                logger.debug("Auto-cleanup disabled, vectors will persist")
            
            return final_answers
            
        except Exception as e:
            logger.error(f"Error processing multiple questions: {str(e)}")
            # Attempt cleanup even if there was an error (if auto-cleanup enabled)
            if self.auto_cleanup:
                try:
                    self.cleanup_session()
                    logger.info("Vectors cleaned up after error")
                except Exception as cleanup_error:
                    logger.warning(f"Failed to cleanup vectors after error (non-critical): {str(cleanup_error)}")
            return [f"Error processing questions: {str(e)}" for _ in questions]

    def cleanup_session(self, session_id: Optional[str] = None):
        """Clean up hybrid search session"""
        try:
            target_session = session_id or self.current_session_id
            self.hybrid_search.cleanup_session(target_session)
            
            if target_session == self.current_session_id:
                self.current_session_id = None
                self.chunks = []
            
            logger.info(f"✅ Advanced Hybrid RAG session cleanup completed: {target_session}")
            
        except Exception as e:
            logger.error(f"Error cleaning up Advanced Hybrid RAG session: {str(e)}")

    def get_api_key_statistics(self) -> Dict:
        """Get comprehensive statistics about all services"""
        stats = {}
        
        # Hybrid search statistics
        stats["hybrid_search"] = self.hybrid_search.get_statistics()
        
        # Gemini API key statistics
        if self.gemini_key_manager:
            stats["gemini_keys"] = self.gemini_key_manager.get_statistics()
        else:
            stats["gemini_keys"] = {
                "service": "Gemini",
                "mode": "single_key_fallback", 
                "message": "Using single GOOGLE_API_KEY"
            }
        
        # Gemini model statistics
        stats["gemini_models"] = self.model_manager.get_statistics()
        
        # Configuration
        stats["config"] = {
            "service": "AdvancedHybridRAGService",
            "top_k_per_index": self.config.top_k_per_index,
            "final_top_k": self.config.final_top_k,
            "dense_model": self.config.dense_model,
            "sparse_model": self.config.sparse_model,
            "rerank_model": self.config.rerank_model,
            "enable_reranking": self.config.enable_reranking,
            "auto_cleanup": self.auto_cleanup,
            "current_session": self.current_session_id
        }
        
        return stats


def create_advanced_hybrid_rag_service(config: Optional[HybridSearchConfig] = None) -> AdvancedHybridRAGService:
    """Factory function to create Advanced Hybrid RAG service"""
    if config is None:
        config = HybridSearchConfig.from_env()
    
    auto_cleanup = config.auto_cleanup
    return AdvancedHybridRAGService(config=config, auto_cleanup=auto_cleanup)
