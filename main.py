from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
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

app = FastAPI(
    title="HackRX LLM Query-Retrieval System",
    description="Intelligent document processing and question answering system",
    version="1.0.0"
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
VALID_TOKEN = os.getenv("API_TOKEN", "[REDACTED]")

# Initialize services
document_processor = DocumentProcessor(chunk_size=500, chunk_overlap=80)
rag_service = create_rag_service()

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
    return HealthResponse(
        status="healthy",
        message="HackRX API is running",
        version="1.0.0"
    )

@app.post("/hackrx/run", response_model=HackRXResponse)
async def process_hackrx_request(
    request: HackRXRequest,
    token: str = Depends(verify_token)
):
    """
    Main endpoint for processing documents and answering questions
    """
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
            "main": "/hackrx/run"
        },
        "features": [
            "PDF document processing",
            "Semantic search with embeddings",
            "Question answering with GPT-3.5-turbo"
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
