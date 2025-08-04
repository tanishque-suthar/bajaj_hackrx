from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import os
from dotenv import load_dotenv
from models import HackRXRequest, HackRXResponse, HealthResponse
from core.document import DocumentProcessor, get_document_stats
from core.rag import RAGService, create_rag_service
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
        app.state.rag_service = create_rag_service()
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
        logger.info(f"Processing request with {len(request.questions)} questions")
        logger.info(f"Document URL: {request.documents}")

        # Step 1: Process document
        logger.info("Step 1: Processing document...")
        document_chunks = await document_processor.process_document_from_url(request.documents)

        # Log document processing stats
        stats = get_document_stats(document_chunks)
        logger.info(f"Document processing stats: {stats}")

        # Step 2: Answer questions using RAG
        logger.info("Step 2: Answering questions using RAG...")
        answers = await rag_service.answer_multiple_questions(request.questions, document_chunks)

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
            "api_stats": "/api-stats"
        },
        "features": [
            "PDF document processing",
            "Semantic search with embeddings",
            "Question answering with Gemini",
            "Multiple API key fallback support",
            "Multi-model fallback (gemini-1.5-flash → gemma-3-12b-it → gemma-3-4b-it)"
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