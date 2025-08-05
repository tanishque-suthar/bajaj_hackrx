import google.generativeai as genai
import numpy as np
from pinecone import Pinecone, ServerlessSpec
import re
from typing import List, Dict, Optional
import logging
import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
import asyncio
import uuid
import time
from core.api_key_manager import HuggingFaceAPIKeyManager, GeminiAPIKeyManager, ModelManager, PineconeAPIKeyManager
from core.advanced_hybrid import AdvancedHybridRAGService, create_advanced_hybrid_rag_service
from core.hybrid_config import HybridSearchConfig
load_dotenv()
logger = logging.getLogger(__name__)

class RAGService:
    """RAG service using Hugging Face Inference API for embeddings and Pinecone for vector storage"""
    
    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        top_k: int = 5,
        gemma_model: str = "gemini-1.5-flash",
        auto_cleanup: bool = True
    ):
        self.top_k = top_k
        self.gemma_model = gemma_model
        self.embedding_model = embedding_model
        self.embedding_dim = 384  # Dimension for all-MiniLM-L6-v2
        self.auto_cleanup = auto_cleanup  # Control automatic cleanup behavior
        
        # --- Initialize Hugging Face API Key Manager ---
        try:
            self.hf_key_manager = HuggingFaceAPIKeyManager()
            logger.info(f"Initialized HF API Key Manager with {len(self.hf_key_manager.api_keys)} keys")
        except Exception as e:
            logger.error(f"Failed to initialize HF API Key Manager: {str(e)}")
            # Fallback to single key approach
            hf_api_token = os.getenv("HF_API_TOKEN")
            if not hf_api_token:
                raise ValueError("Neither HF_API_TOKENS nor HF_API_TOKEN found in environment variables")
            self.hf_key_manager = None
            self.hf_client = InferenceClient(api_key=hf_api_token)
            logger.warning("Using fallback single HF API token")
        
        # --- Initialize Gemini API Key Manager ---
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
        
        # --- Initialize Pinecone ---
        try:
            pinecone_api_key = os.getenv("PINECONE_API_KEY")
            if not pinecone_api_key:
                raise ValueError("PINECONE_API_KEY not found in environment variables")
            
            logger.info("Initializing Pinecone client...")
            self.pinecone_client = Pinecone(api_key=pinecone_api_key)
            self.pinecone_environment = os.getenv("PINECONE_ENVIRONMENT", "us-east-1-aws")
            self.pinecone_index_name = os.getenv("PINECONE_INDEX_NAME", "bajaj-hackrx-index")
            
            logger.info(f"Pinecone client created for environment: {self.pinecone_environment}")
            logger.info(f"Target index name: {self.pinecone_index_name}")
            
            # Initialize or connect to index
            self._initialize_pinecone_index()
            
            logger.info("Pinecone initialization completed successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {str(e)}")
            self.pinecone_client = None
            self.pinecone_index = None
            raise ValueError(f"Pinecone initialization failed: {str(e)}")
        
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
        
        # Session management (index is initialized above in Pinecone section)
        self.current_session_id = None
        self.chunks = []
        
        # Debug: Verify Pinecone index state after initialization
        if hasattr(self, 'pinecone_index') and self.pinecone_index is not None:
            logger.info("✅ Pinecone index successfully initialized and ready for use")
        else:
            logger.error("❌ Pinecone index is None after initialization - this will cause errors")

    def _initialize_pinecone_index(self):
        """Initialize or connect to Pinecone index"""
        try:
            logger.info("Starting Pinecone index initialization...")
            
            # Check if index exists
            logger.info("Listing existing Pinecone indexes...")
            existing_indexes = self.pinecone_client.list_indexes()
            index_names = [index.name for index in existing_indexes]
            logger.info(f"Existing indexes: {index_names}")
            
            if self.pinecone_index_name not in index_names:
                logger.info(f"Creating new Pinecone index: {self.pinecone_index_name}")
                
                # Parse region correctly - handle both "us-east-1" and "us-east-1-aws" formats
                region = self.pinecone_environment
                if not region.endswith("-aws"):
                    region = f"{region}-aws"
                logger.info(f"Using region: {region}")
                
                # Create new index with serverless spec
                self.pinecone_client.create_index(
                    name=self.pinecone_index_name,
                    dimension=self.embedding_dim,
                    metric="cosine",  # Equivalent to IP with L2 normalization
                    spec=ServerlessSpec(
                        cloud="aws",
                        region=region
                    )
                )
                logger.info("Index creation request sent, waiting for index to be ready...")
                
                # Wait for index to be ready
                while not self.pinecone_client.describe_index(self.pinecone_index_name).status['ready']:
                    logger.info("Waiting for Pinecone index to be ready...")
                    time.sleep(1)
                
                logger.info(f"Pinecone index created successfully: {self.pinecone_index_name}")
            else:
                logger.info(f"Pinecone index already exists: {self.pinecone_index_name}")
            
            # Connect to index
            logger.info(f"Connecting to Pinecone index: {self.pinecone_index_name}")
            self.pinecone_index = self.pinecone_client.Index(self.pinecone_index_name)
            
            # Verify connection by getting index stats
            logger.info("Verifying index connection...")
            stats = self.pinecone_index.describe_index_stats()
            logger.info(f"Index stats - Total vectors: {stats.total_vector_count}, Dimension: {stats.dimension}")
            logger.info(f"Successfully connected to Pinecone index: {self.pinecone_index_name}")
            
        except Exception as e:
            logger.error(f"Error initializing Pinecone index: {str(e)}")
            logger.error(f"Exception type: {type(e).__name__}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            self.pinecone_index = None  # Ensure it's set to None on failure
            raise Exception(f"Failed to initialize Pinecone index: {str(e)}")

    def _generate_session_id(self) -> str:
        """Generate a unique session ID for document processing"""
        return f"session_{int(time.time())}_{uuid.uuid4().hex[:8]}"

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
        """Create embeddings for a list of texts using Hugging Face Inference API with fallback"""
        if not self.hf_key_manager:
            # Fallback to single client
            return await self._create_embeddings_single_client(texts)
        
        logger.info(f"Creating embeddings for {len(texts)} texts via Hugging Face API with fallback")
        
        max_attempts = len(self.hf_key_manager.api_keys)
        last_error = None
        
        for attempt in range(max_attempts):
            api_key = self.hf_key_manager.get_next_available_key()
            
            if not api_key:
                logger.error("No available Hugging Face API keys")
                break
            
            try:
                # Create client with current API key
                client = InferenceClient(api_key=api_key)
                
                # Make API call
                embeddings = client.feature_extraction(
                    text=texts,
                    model=self.embedding_model
                )
                
                # Success - mark key as successful and return results
                self.hf_key_manager.mark_key_successful(api_key)
                embeddings_array = np.array(embeddings, dtype=np.float32)
                logger.info(f"Created embeddings with shape: {embeddings_array.shape}")
                return embeddings_array
                
            except Exception as e:
                error_msg = str(e)
                last_error = e
                
                logger.warning(f"HF API attempt {attempt + 1} failed with key {self.hf_key_manager._mask_key(api_key)}: {error_msg}")
                
                # Check if it's a rate limit error
                if self.hf_key_manager.is_rate_limit_error(error_msg):
                    self.hf_key_manager.mark_key_rate_limited(api_key)
                    logger.info(f"Rate limit detected, trying next key...")
                    continue
                
                # Check if it's an authentication error
                elif self.hf_key_manager.is_auth_error(error_msg):
                    self.hf_key_manager.mark_key_failed(api_key)
                    logger.warning(f"Authentication error, marking key as failed and trying next...")
                    continue
                
                # For other errors, try next key without marking as failed
                else:
                    logger.warning(f"Unknown error, trying next key: {error_msg}")
                    continue
        
        # All attempts failed
        stats = self.hf_key_manager.get_statistics()
        logger.error(f"All Hugging Face API keys exhausted. Stats: {stats}")
        
        if last_error:
            raise Exception(f"All Hugging Face API keys failed. Last error: {str(last_error)}")
        else:
            raise Exception("All Hugging Face API keys are unavailable")
    
    async def _create_embeddings_single_client(self, texts: List[str]) -> np.ndarray:
        """Fallback method for single client (backward compatibility)"""
        try:
            logger.info(f"Creating embeddings for {len(texts)} texts via single HF client")
            embeddings = self.hf_client.feature_extraction(
                text=texts,
                model=self.embedding_model
            )
            embeddings_array = np.array(embeddings, dtype=np.float32)
            logger.info(f"Created embeddings with shape: {embeddings_array.shape}")
            return embeddings_array
        except Exception as e:
            logger.error(f"Error creating Hugging Face embeddings: {str(e)}")
            raise Exception(f"Error creating Hugging Face embeddings: {str(e)}")

    async def build_vector_store(self, chunks: List[str]) -> None:
        """Build Pinecone vector store from document chunks"""
        try:
            if not self.pinecone_index:
                raise ValueError("Pinecone index not initialized. Check Pinecone configuration and connection.")
            
            logger.info("Building vector store in Pinecone from chunks")
            self.chunks = chunks
            
            # Generate a new session ID for this document processing
            self.current_session_id = self._generate_session_id()
            
            # Create embeddings for all chunks
            chunk_embeddings = await self.create_embeddings(chunks)
            
            # Prepare vectors for Pinecone upsert
            vectors_to_upsert = []
            for i, (chunk, embedding) in enumerate(zip(chunks, chunk_embeddings)):
                vector_id = f"{self.current_session_id}_chunk_{i}"
                metadata = {
                    "session_id": self.current_session_id,
                    "chunk_index": i,
                    "text": chunk[:1000],  # Limit text size for metadata
                    "chunk_length": len(chunk),
                    "timestamp": int(time.time())
                }
                
                vectors_to_upsert.append({
                    "id": vector_id,
                    "values": embedding.tolist(),
                    "metadata": metadata
                })
            
            # Upsert vectors to Pinecone in batches
            batch_size = 100  # Pinecone recommended batch size
            for i in range(0, len(vectors_to_upsert), batch_size):
                batch = vectors_to_upsert[i:i + batch_size]
                self.pinecone_index.upsert(vectors=batch)
                logger.info(f"Upserted batch {i // batch_size + 1}/{(len(vectors_to_upsert) + batch_size - 1) // batch_size}")
            
            logger.info(f"Vector store built in Pinecone with {len(chunks)} chunks for session: {self.current_session_id}")
            
        except Exception as e:
            logger.error(f"Error building vector store: {str(e)}")
            raise Exception(f"Error building vector store: {str(e)}")

    async def retrieve_relevant_chunks(self, query: str) -> List[Dict]:
        """Retrieve top-k relevant chunks for a query using Pinecone"""
        try:
            if not self.pinecone_index or not self.current_session_id:
                raise ValueError("Vector store not initialized. Call build_vector_store first.")
            
            logger.info(f"Retrieving relevant chunks for query: {query[:50]}...")
            
            # Create query embedding
            query_embedding = await self.create_embeddings([query])
            
            # Query Pinecone index
            query_response = self.pinecone_index.query(
                vector=query_embedding[0].tolist(),
                top_k=self.top_k,
                include_metadata=True,
                filter={"session_id": self.current_session_id}  # Filter by current session
            )
            
            # Format results
            results = []
            for match in query_response.matches:
                # Get the full chunk text from our stored chunks or metadata
                chunk_index = match.metadata.get("chunk_index", 0)
                # Ensure chunk_index is an integer (Pinecone may return it as float)
                chunk_index = int(chunk_index)
                chunk_text = ""
                
                if chunk_index < len(self.chunks):
                    chunk_text = self.chunks[chunk_index]
                else:
                    # Fallback to metadata text if chunks not available
                    chunk_text = match.metadata.get("text", "")
                
                results.append({
                    "chunk": chunk_text,
                    "score": float(match.score),
                    "index": chunk_index,
                    "vector_id": match.id
                })
            
            logger.info(f"Retrieved {len(results)} relevant chunks from Pinecone")
            return results
            
        except Exception as e:
            logger.error(f"Error retrieving relevant chunks: {str(e)}")
            raise Exception(f"Error retrieving relevant chunks: {str(e)}")
            
    def generate_answer(self, query: str, relevant_chunks: List[Dict]) -> str:
        """Generate answer using Google Gemini with model-first fallback strategy"""
        if not self.gemini_key_manager:
            # Fallback to single client
            return self._generate_answer_single_client(query, relevant_chunks)
        
        logger.info("Generating answer using Google Gemini with model-first fallback")
        
        context = "\n\n".join([
            f"[Document Section {i+1}]:\n{chunk['chunk']}" 
            for i, chunk in enumerate(relevant_chunks)
        ])
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
        prompt = f"""You are a meticulous insurance policy analyst. Your task is to provide a precise and factual answer to the question based ONLY on the provided policy sections.

POLICY SECTIONS:
{context}

QUESTION: {query}

INSTRUCTIONS:
1. Answer the question using only the information from the 'POLICY SECTIONS' above.
2. Extract specific facts, such as time periods (e.g., 30 days, 24 months), monetary values, percentages, and conditions.
3. If the answer cannot be found in the provided sections, respond with exactly: "The answer cannot be found in the provided policy sections."
4. Be direct and concise.
5. Answer in clean, continuous prose without bullet points or formatting marks

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
        """Answer multiple questions efficiently with automatic cleanup"""
        try:
            logger.info("=" * 80)
            logger.info(f"📚 PROCESSING {len(questions)} QUESTIONS WITH TRADITIONAL RAG")
            logger.info("=" * 80)
            logger.info(f"📄 Document chunks: {len(chunks)}")
            logger.info(f"🔍 Search mode: Traditional (Dense only)")
            logger.info(f"⚙️ Config: {self.top_k} top-k results")
            
            # Build vector store once for all questions
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
                            score = chunk.get('score', 0)
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
                    logger.info("🧹 Automatically cleaning up vectors after request completion")
                    self.cleanup_session()
                    logger.info("✅ Vectors cleaned up successfully")
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
        """Clean up vectors from a specific session in Pinecone"""
        try:
            if not self.pinecone_index:
                logger.warning("Pinecone index not initialized, skipping cleanup")
                return
                
            target_session = session_id or self.current_session_id
            if not target_session:
                logger.warning("No session ID provided for cleanup")
                return
            
            logger.info(f"Cleaning up Pinecone vectors for session: {target_session}")
            
            # Delete vectors by session filter
            self.pinecone_index.delete(filter={"session_id": target_session})
            
            if target_session == self.current_session_id:
                self.current_session_id = None
                self.chunks = []
            
            logger.info(f"Cleaned up session: {target_session}")
            
        except Exception as e:
            target_session = session_id or self.current_session_id
            logger.error(f"Error cleaning up session {target_session}: {str(e)}")

    def get_api_key_statistics(self) -> Dict:
        """Get statistics about API key usage and model usage for all services"""
        stats = {}
        
        # Hugging Face statistics
        if self.hf_key_manager:
            stats["huggingface"] = self.hf_key_manager.get_statistics()
        else:
            stats["huggingface"] = {
                "service": "HuggingFace",
                "mode": "single_key_fallback",
                "message": "Using single HF_API_TOKEN (no multi-key manager)"
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
        
        # Pinecone statistics
        try:
            index_stats = self.pinecone_index.describe_index_stats()
            stats["pinecone"] = {
                "service": "Pinecone",
                "index_name": self.pinecone_index_name,
                "environment": self.pinecone_environment,
                "current_session": self.current_session_id,
                "total_vectors": index_stats.total_vector_count,
                "dimension": self.embedding_dim,
                "index_fullness": index_stats.index_fullness,
                "namespaces": dict(index_stats.namespaces) if hasattr(index_stats, 'namespaces') else {}
            }
        except Exception as e:
            stats["pinecone"] = {
                "service": "Pinecone",
                "error": f"Unable to get Pinecone stats: {str(e)}"
            }
        
        return stats


def create_rag_service() -> RAGService:
    """Factory function to create RAG service"""
    # Read auto-cleanup setting from environment
    auto_cleanup = os.getenv("AUTO_CLEANUP_VECTORS", "True").lower() in ("true", "1", "yes", "on")
    return RAGService(auto_cleanup=auto_cleanup)


def create_hybrid_rag_service() -> AdvancedHybridRAGService:
    """Factory function to create Advanced Hybrid RAG service"""
    return create_advanced_hybrid_rag_service()


def get_rag_service(use_hybrid: bool = None) -> RAGService:
    """
    Get RAG service based on configuration
    
    Args:
        use_hybrid: If True, use hybrid search. If False, use traditional RAG.
                   If None, read from environment variable ENABLE_HYBRID_SEARCH
    
    Returns:
        RAGService or AdvancedHybridRAGService instance
    """
    if use_hybrid is None:
        use_hybrid = os.getenv("ENABLE_HYBRID_SEARCH", "False").lower() in ("true", "1", "yes", "on")
    
    if use_hybrid:
        logger.info("🚀 Creating Advanced Hybrid RAG Service (Dense + Sparse Search)")
        return create_hybrid_rag_service()
    else:
        logger.info("📚 Creating Traditional RAG Service (Dense Search Only)")
        return create_rag_service()
