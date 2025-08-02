import google.generativeai as genai
import numpy as np
import faiss
import re
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple, Optional
import logging
import os
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

class RAGService:
    """RAG (Retrieval-Augmented Generation) service for question answering using Gemma 3n"""
    
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        top_k: int = 5,
        gemma_model: str = "gemma-3n-e4b-it"  # Latest experimental model
    ):
        self.top_k = top_k
        self.gemma_model = gemma_model
        
        # Initialize embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        
        # Initialize Google AI
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        
        genai.configure(api_key=google_api_key)
        
        # Initialize the model
        self.model = genai.GenerativeModel(self.gemma_model)
        
        # Configure generation parameters
        self.generation_config = genai.types.GenerationConfig(
            temperature=0.1,  # Low temperature for factual responses
            top_p=1.0,
            top_k=12,
            max_output_tokens=200,
            stop_sequences=None,
        )
        
        # Vector store (will be populated when processing documents)
        self.vector_store = None
        self.chunks = []
        self.chunk_embeddings = None
        
    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """Create embeddings for a list of texts"""
        try:
            logger.info(f"Creating embeddings for {len(texts)} texts")
            embeddings = self.embedding_model.encode(texts, convert_to_tensor=False)
            logger.info(f"Created embeddings with shape: {embeddings.shape}")
            return embeddings
        except Exception as e:
            raise Exception(f"Error creating embeddings: {str(e)}")
    
    def build_vector_store(self, chunks: List[str]) -> None:
        """Build FAISS vector store from document chunks"""
        try:
            logger.info("Building vector store from chunks")
            
            # Store chunks
            self.chunks = chunks
            
            # Create embeddings for all chunks
            self.chunk_embeddings = self.create_embeddings(chunks)
            
            # Initialize FAISS index
            self.vector_store = faiss.IndexFlatIP(self.embedding_dim)  # Inner Product for cosine similarity
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(self.chunk_embeddings)
            
            # Add embeddings to index
            self.vector_store.add(self.chunk_embeddings.astype(np.float32))
            
            logger.info(f"Vector store built with {len(chunks)} chunks")
            
        except Exception as e:
            raise Exception(f"Error building vector store: {str(e)}")
    
    def retrieve_relevant_chunks(self, query: str) -> List[Dict]:
        """Retrieve top-k relevant chunks for a query"""
        try:
            if not self.vector_store or not self.chunks:
                raise ValueError("Vector store not initialized. Call build_vector_store first.")
            
            logger.info(f"Retrieving relevant chunks for query: {query[:50]}...")
            
            # Create query embedding
            query_embedding = self.create_embeddings([query])
            faiss.normalize_L2(query_embedding)
            
            # Search for similar chunks
            scores, indices = self.vector_store.search(
                query_embedding.astype(np.float32), 
                self.top_k
            )
            
            # Prepare results
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx < len(self.chunks):  # Ensure valid index
                    results.append({
                        "chunk": self.chunks[idx],
                        "score": float(score),
                        "index": int(idx),
                        "rank": i + 1
                    })
            
            logger.info(f"Retrieved {len(results)} relevant chunks")
            return results
            
        except Exception as e:
            raise Exception(f"Error retrieving relevant chunks: {str(e)}")
    
    def generate_answer(self, query: str, relevant_chunks: List[Dict]) -> str:
        """Generate answer using Google Gemma 3n based on relevant chunks"""
        try:
            logger.info("Generating answer using Google Gemma 3n")
            
            # Prepare context from relevant chunks
            context = "\n\n".join([
                f"[Document Section {i+1}]:\n{chunk['chunk']}" 
                for i, chunk in enumerate(relevant_chunks)
            ])
            
            # Create prompt optimized for Gemma
            prompt = self._create_gemma_prompt(query, context)
            
            # Generate response using Google AI
            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config
            )
            
            # Extract answer from response
            raw_answer = response.text.strip()
            
            # Clean the answer
            cleaned_answer = self._clean_answer(raw_answer)
            logger.info("Answer generated and cleaned successfully using Gemma 3n")
            
            return cleaned_answer
            
        except Exception as e:
            logger.error(f"Error generating answer with Gemma: {str(e)}")
            raise Exception(f"Error generating answer: {str(e)}")
    
    def _create_gemma_prompt(self, query: str, context: str) -> str:
        """Create a well-structured prompt optimized for Gemma 3n"""
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
        """Clean and format the LLM-generated answer"""
        try:
            # Remove the raw answer if empty or whitespace only
            if not raw_answer or not raw_answer.strip():
                return "I couldn't generate a response. Please try rephrasing your question."
            
            answer = raw_answer.strip()
            
            # Remove common LLM artifacts and formatting issues
            # Remove leading "Answer:" or "ANSWER:" if present
            answer = re.sub(r'^(ANSWER|Answer):\s*', '', answer, flags=re.IGNORECASE)
            
            # Remove markdown-style formatting
            answer = re.sub(r'\*\*(.*?)\*\*', r'\1', answer)  # Remove bold **text**
            answer = re.sub(r'\*(.*?)\*', r'\1', answer)      # Remove italic *text*
            answer = re.sub(r'`(.*?)`', r'\1', answer)        # Remove code `text`
            
            # Remove excessive special characters and normalize
            answer = re.sub(r'[^\w\s\.,;:!?()\-\%\$\@\&\#\*\+\=\[\]\{\}<>/"\']', ' ', answer)
            
            # Fix spacing issues
            answer = re.sub(r'\s+([.,;:!?])', r'\1', answer)  # Remove space before punctuation
            answer = re.sub(r'([.,;:!?])\s*([A-Z])', r'\1 \2', answer)  # Ensure space after punctuation
            answer = re.sub(r'\s{2,}', ' ', answer)  # Remove multiple spaces
            
            # Remove common prefixes that might leak from prompts
            answer = re.sub(r'^(Based on the provided|According to the|From the document|The document states|In the policy)', 
                          '', answer, flags=re.IGNORECASE).strip()
            
            # Ensure proper sentence structure
            if answer and not answer[0].isupper():
                answer = answer[0].upper() + answer[1:]
            
            # Ensure proper ending punctuation
            if answer and answer[-1] not in '.!?':
                answer += '.'
            
            # Final cleanup
            answer = answer.strip()
            
            # Validate minimum length
            if len(answer) < 10:
                return "The information requested could not be found in the provided document sections."
            
            return answer
            
        except Exception as e:
            logger.error(f"Error cleaning answer: {str(e)}")
            return raw_answer.strip() if raw_answer else "Unable to process the response."
    
    async def answer_question(self, query: str, chunks: List[str]) -> str:
        """Main method to answer a single question"""
        try:
            # Build vector store if not already built or if chunks changed
            if not self.vector_store or len(self.chunks) != len(chunks):
                self.build_vector_store(chunks)
            
            # Retrieve relevant chunks
            relevant_chunks = self.retrieve_relevant_chunks(query)
            
            # Generate answer
            answer = self.generate_answer(query, relevant_chunks)
            
            return answer
            
        except Exception as e:
            logger.error(f"Error answering question '{query}': {str(e)}")
            return f"Sorry, I couldn't answer this question due to an error: {str(e)}"
    
    async def answer_multiple_questions(self, questions: List[str], chunks: List[str]) -> List[str]:
        """Answer multiple questions efficiently"""
        try:
            logger.info(f"Processing {len(questions)} questions with Gemma 3n")
            
            # Build vector store once for all questions
            self.build_vector_store(chunks)
            
            # Process each question
            answers = []
            for i, question in enumerate(questions):
                logger.info(f"Processing question {i+1}/{len(questions)} with Gemma")
                try:
                    answer = await self.answer_question(question, chunks)
                    answers.append(answer)
                except Exception as e:
                    logger.error(f"Error processing question {i+1}: {str(e)}")
                    answers.append(f"Sorry, I couldn't process this question: {str(e)}")
            
            logger.info("All questions processed successfully with Gemma 3n")
            return answers
            
        except Exception as e:
            logger.error(f"Error processing multiple questions: {str(e)}")
            # Return error messages for all questions
            return [f"Error processing questions: {str(e)}" for _ in questions]

# Utility functions
def create_rag_service() -> RAGService:
    """Factory function to create RAG service with Gemma 3n"""
    return RAGService(
        embedding_model="all-MiniLM-L6-v2",
        top_k=5,
        gemma_model="gemma-3n-e4b-it"  # Using latest experimental model
    )
