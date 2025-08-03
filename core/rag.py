import google.generativeai as genai
import numpy as np
import faiss
import re
from typing import List, Dict
import logging
import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
import asyncio
load_dotenv()
logger = logging.getLogger(__name__)

class RAGService:
    """RAG service using Hugging Face Inference API for embeddings and Gemini for generation"""
    
    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        top_k: int = 5,
        gemma_model: str = "gemini-1.5-flash"
    ):
        self.top_k = top_k
        self.gemma_model = gemma_model
        self.embedding_model = embedding_model
        
        # --- Hugging Face Inference Client Configuration ---
        hf_api_token = os.getenv("HF_API_TOKEN")
        if not hf_api_token:
            raise ValueError("HF_API_TOKEN not found in environment variables")
        
        self.hf_client = InferenceClient(
            api_key=hf_api_token,
        )
        self.embedding_dim = 384  # Dimension for all-MiniLM-L6-v2
        
        # --- Initialize Google AI Generative Model (Lazy Loading) ---
        self.model = None
        self.generation_config = None
        
        self.vector_store = None
        self.chunks = []

    def _initialize_gemini_model(self):
        """Initialize Gemini model lazily"""
        if self.model is None:
            google_api_key = os.getenv("GOOGLE_API_KEY")
            if not google_api_key:
                raise ValueError("GOOGLE_API_KEY not found in environment variables")
            
            genai.configure(api_key=google_api_key)
            self.model = genai.GenerativeModel(self.gemma_model)
            
            self.generation_config = genai.types.GenerationConfig(
                temperature=0.1,
                top_p=1.0,
                top_k=12,
                max_output_tokens=200,
            )

    async def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """Create embeddings for a list of texts using Hugging Face Inference API"""
        try:
            logger.info(f"Creating embeddings for {len(texts)} texts via Hugging Face API")
            
            # Use feature extraction with huggingface_hub
            embeddings = self.hf_client.feature_extraction(
                text=texts,
                model=self.embedding_model
            )
            
            # Convert to numpy array
            embeddings_array = np.array(embeddings, dtype=np.float32)
            logger.info(f"Created embeddings with shape: {embeddings_array.shape}")
            return embeddings_array

        except Exception as e:
            logger.error(f"Error creating Hugging Face embeddings: {str(e)}")
            raise Exception(f"Error creating Hugging Face embeddings: {str(e)}")

    async def build_vector_store(self, chunks: List[str]) -> None:
        """Build FAISS vector store from document chunks"""
        try:
            logger.info("Building vector store from chunks")
            self.chunks = chunks
            
            chunk_embeddings = await self.create_embeddings(chunks)
            
            self.vector_store = faiss.IndexFlatIP(self.embedding_dim)
            faiss.normalize_L2(chunk_embeddings)
            self.vector_store.add(chunk_embeddings)
            
            logger.info(f"Vector store built with {len(chunks)} chunks")
        except Exception as e:
            raise Exception(f"Error building vector store: {str(e)}")

    async def retrieve_relevant_chunks(self, query: str) -> List[Dict]:
        """Retrieve top-k relevant chunks for a query"""
        try:
            if not self.vector_store or not self.chunks:
                raise ValueError("Vector store not initialized. Call build_vector_store first.")
            
            logger.info(f"Retrieving relevant chunks for query: {query[:50]}...")
            
            query_embedding = await self.create_embeddings([query])
            faiss.normalize_L2(query_embedding)
            
            scores, indices = self.vector_store.search(query_embedding, self.top_k)
            
            results = [
                {"chunk": self.chunks[idx], "score": float(score), "index": int(idx)}
                for score, idx in zip(scores[0], indices[0]) if idx < len(self.chunks)
            ]
            
            logger.info(f"Retrieved {len(results)} relevant chunks")
            return results
        except Exception as e:
            raise Exception(f"Error retrieving relevant chunks: {str(e)}")
            
    def generate_answer(self, query: str, relevant_chunks: List[Dict]) -> str:
        """Generate answer using Google Gemini"""
        try:
            # Initialize Gemini model if not already done
            self._initialize_gemini_model()
            
            logger.info("Generating answer using Google Gemini")
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


def create_rag_service() -> RAGService:
    """Factory function to create RAG service"""
    return RAGService()
