import google.generativeai as genai
import numpy as np
import faiss
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
        top_k: int = 3,
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
            top_k=40,
            max_output_tokens=500,
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
            answer = response.text.strip()
            logger.info("Answer generated successfully using Gemma 3n")
            
            return answer
            
        except Exception as e:
            logger.error(f"Error generating answer with Gemma: {str(e)}")
            raise Exception(f"Error generating answer: {str(e)}")
    
    def _create_gemma_prompt(self, query: str, context: str) -> str:
        """Create a well-structured prompt optimized for Gemma 3n"""
        prompt = f"""You are an expert insurance policy analyst. Your task is to answer questions accurately based on provided policy documents.

POLICY DOCUMENT EXCERPTS:
{context}

QUESTION: {query}

INSTRUCTIONS:
- Answer based ONLY on the information provided in the document excerpts above
- Be specific and include relevant details (time periods, amounts, conditions)
- If information is not available in the provided excerpts, clearly state "This information is not available in the provided document sections"
- Keep your response clear, professional, and concise
- Do not make assumptions beyond what is explicitly stated
- Focus on factual accuracy over lengthy explanations

ANSWER:"""
        
        return prompt
    
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
        top_k=3,
        gemma_model="gemma-3n-e4b-it"  # Using latest experimental model
    )
