from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import os
from dotenv import load_dotenv
from models import HackRXRequest, HackRXResponse, HealthResponse
from core.document import DocumentProcessor, get_document_stats
from core.rag import RAGService, get_rag_service
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- CHANGE #2: Modern lifespan context manager ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events
    """
    # Startup
    logger.info("Application startup: Initializing services...")
    try:
        logger.info("Initializing DocumentProcessor...")
        app.state.document_processor = DocumentProcessor(chunk_size=500, chunk_overlap=80)
        logger.info("DocumentProcessor initialized successfully")
        
        logger.info("Initializing RAG service...")
        # Use the new factory function that supports hybrid search
        app.state.rag_service = get_rag_service()
        logger.info("RAG service initialized successfully")
        
        logger.info("All services initialized successfully.")
    except Exception as e:
        logger.error(f"Error during service initialization: {str(e)}")
        raise e
    
    yield
    
    # Shutdown (if needed)
    logger.info("Application shutdown")

app = FastAPI(
    title="HackRX LLM Query-Retrieval System",
    description="Intelligent document processing and question answering system",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Auth configuration
VALID_TOKEN = os.getenv("API_TOKEN", " ")

# --- CHANGE #1: Remove the global service initializations from here ---
# These lines were causing the blocking issue
# document_processor = DocumentProcessor(chunk_size=500, chunk_overlap=80)
# rag_service = create_rag_service()


async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify bearer token authentication"""
    if credentials.credentials != VALID_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    logger.info("Health check endpoint called")
    return HealthResponse(
        status="healthy",
        message="HackRX API is running",
        version="1.0.0"
    )

@app.get("/simple-health")
async def simple_health_check():
    """Simple health check without Pydantic model"""
    logger.info("Simple health check endpoint called")
    return {"status": "ok", "message": "API is running"}

@app.get("/api-stats")
async def get_api_statistics(req: Request, token: str = Depends(verify_token)):
    """Get API key usage statistics"""
    try:
        rag_service = req.app.state.rag_service
        stats = rag_service.get_api_key_statistics()
        return {"status": "success", "statistics": stats}
    except AttributeError as e:
        logger.error(f"RAG service not initialized: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="RAG service not properly initialized"
        )
    except Exception as e:
        logger.error(f"Error getting API statistics: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving statistics: {str(e)}"
        )

@app.post("/cleanup-session")
async def cleanup_session(req: Request, session_id: str = None, token: str = Depends(verify_token)):
    """Clean up Pinecone vectors for a specific session"""
    try:
        rag_service = req.app.state.rag_service
        rag_service.cleanup_session(session_id)
        return {"status": "success", "message": f"Session {session_id or 'current'} cleaned up successfully"}
    except AttributeError as e:
        logger.error(f"RAG service not initialized: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="RAG service not properly initialized"
        )
    except Exception as e:
        logger.error(f"Error cleaning up session: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error cleaning up session: {str(e)}"
        )

@app.post("/hackrx/run", response_model=HackRXResponse)
async def process_hackrx_request(
    # --- CHANGE #3: Access services via the 'Request' object ---
    request: HackRXRequest,
    req: Request, # Get the main FastAPI request object
    token: str = Depends(verify_token)
):
    """
    Main endpoint for processing documents and answering questions
    """
    # Get the initialized services from the application state
    try:
        document_processor = req.app.state.document_processor
        rag_service = req.app.state.rag_service
    except AttributeError as e:
        logger.error(f"Services not properly initialized: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Services not properly initialized. Please restart the application."
        )

    try:
        # === REQUEST SUMMARY ===
        logger.info("=" * 80)
        logger.info("📋 REQUEST SUMMARY")
        logger.info("=" * 80)
        logger.info(f"📄 Document URL: {request.documents}")
        logger.info(f"❓ Number of Questions: {len(request.questions)}")
        for i, question in enumerate(request.questions, 1):
            logger.info(f"❓ Q{i}: {question}")
        logger.info("=" * 80)

        # Step 1: Process document
        logger.info("Step 1: Processing document...")
        document_chunks = await document_processor.process_document_from_url(request.documents)

        # Log document processing stats
        stats = get_document_stats(document_chunks)
        logger.info(f"Document processing stats: {stats}")

        # Step 2: Answer questions using RAG
        logger.info("Step 2: Answering questions using RAG...")
        answers = await rag_service.answer_multiple_questions(request.questions, document_chunks)

        # === RESPONSE SUMMARY ===
        logger.info("=" * 80)
        logger.info("🤖 RESPONSE SUMMARY")
        logger.info("=" * 80)
        logger.info(f"📄 Document: {request.documents}")
        logger.info(f"✅ Questions Processed: {len(request.questions)}")
        for i, (question, answer) in enumerate(zip(request.questions, answers), 1):
            logger.info(f"❓ Q{i}: {question}")
            logger.info(f"🤖 A{i}: {answer}")
            logger.info("-" * 40)
        logger.info("=" * 80)

        logger.info("Request processed successfully")
        return HackRXResponse(answers=answers)

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "HackRX LLM Query-Retrieval System",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "simple_health": "/simple-health",
            "main": "/hackrx/run",
            "api_stats": "/api-stats",
            "cleanup_session": "/cleanup-session"
        },
        "features": [
            "PDF document processing",
            "Semantic search with embeddings",
            "Question answering with Gemini",
            "Multiple API key fallback support",
            "Multi-model fallback (gemini-1.5-flash → gemma-3-12b-it → gemma-3-4b-it)",
            "Pinecone vector storage with session management"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )