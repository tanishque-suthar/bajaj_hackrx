from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import os
from dotenv import load_dotenv
from models import HackRXRequest, HackRXResponse, HealthResponse
from core.document import DocumentProcessor, get_document_stats
from core.local_rag import LocalRAGService, create_local_rag_service, LocalModelConfig
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
    logger.info("Application startup: Initializing local services...")
    try:
        logger.info("Initializing DocumentProcessor...")
        app.state.document_processor = DocumentProcessor(chunk_size=500, chunk_overlap=80)
        logger.info("DocumentProcessor initialized successfully")
        
        logger.info("Initializing Local RAG service...")
        # Create config from environment variables (required - no fallbacks)
        required_env_vars = {
            "LOCAL_EMBEDDING_MODEL": os.getenv("LOCAL_EMBEDDING_MODEL"),
            "LOCAL_LLM_MODEL": os.getenv("LOCAL_LLM_MODEL"),
            "DEVICE": os.getenv("DEVICE"),
            "USE_QUANTIZATION": os.getenv("USE_QUANTIZATION"),
            "QUANTIZATION_BITS": os.getenv("QUANTIZATION_BITS"),
            "MAX_NEW_TOKENS": os.getenv("MAX_NEW_TOKENS"),
            "TEMPERATURE": os.getenv("TEMPERATURE"),
            "TOP_P": os.getenv("TOP_P"),
            "TOP_K": os.getenv("TOP_K"),
            "TOP_K_CHUNKS": os.getenv("TOP_K_CHUNKS")
        }
        
        # Check for missing environment variables
        missing_vars = [var for var, value in required_env_vars.items() if value is None]
        if missing_vars:
            error_msg = f"Missing required environment variables: {', '.join(missing_vars)}. Please add them to your .env file."
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        local_config = LocalModelConfig(
            embedding_model=required_env_vars["LOCAL_EMBEDDING_MODEL"],
            llm_model=required_env_vars["LOCAL_LLM_MODEL"],
            device=required_env_vars["DEVICE"],
            use_quantization=required_env_vars["USE_QUANTIZATION"].lower() == "true",
            quantization_bits=int(required_env_vars["QUANTIZATION_BITS"]),
            max_new_tokens=int(required_env_vars["MAX_NEW_TOKENS"]),
            temperature=float(required_env_vars["TEMPERATURE"]),
            top_p=float(required_env_vars["TOP_P"]),
            top_k=int(required_env_vars["TOP_K"]),
            top_k_chunks=int(required_env_vars["TOP_K_CHUNKS"])
        )
        app.state.rag_service = create_local_rag_service(local_config)
        
        # Initialize the service
        await app.state.rag_service.initialize()
        logger.info("Local RAG service initialized successfully")
        
        logger.info("All services initialized successfully.")
    except Exception as e:
        logger.error(f"Error during service initialization: {str(e)}")
        raise e
    
    yield
    
    # Shutdown
    logger.info("Application shutdown: Cleaning up...")
    try:
        if hasattr(app.state, 'rag_service'):
            app.state.rag_service.cleanup()
        logger.info("Cleanup completed")
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")
    logger.info("Application shutdown complete")

app = FastAPI(
    title="Local RAG System - Pure Local Inference",
    description="Intelligent document processing and question answering system running completely locally",
    version="2.0.0",
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
        message="Local RAG API is running",
        version="2.0.0"
    )

@app.get("/simple-health")
async def simple_health_check():
    """Simple health check without Pydantic model"""
    logger.info("Simple health check endpoint called")
    return {"status": "ok", "message": "API is running"}

@app.get("/api-stats")
async def get_system_info(req: Request, token: str = Depends(verify_token)):
    """Get system and model information"""
    try:
        rag_service = req.app.state.rag_service
        info = rag_service.get_system_info()
        return {"status": "success", "system_info": info}
    except AttributeError as e:
        logger.error(f"RAG service not initialized: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="RAG service not properly initialized"
        )
    except Exception as e:
        logger.error(f"Error getting system info: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving system info: {str(e)}"
        )

@app.post("/hackrx/run", response_model=HackRXResponse)
async def process_hackrx_request(
    # --- CHANGE #3: Access services via the 'Request' object ---
    request: HackRXRequest,
    req: Request, # Get the main FastAPI request object
    token: str = Depends(verify_token)
):
    """
    Main endpoint for processing documents and answering questions using local RAG
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

        # Step 2: Build vector store from document chunks
        logger.info("Step 2: Building vector store...")
        await rag_service.build_vector_store(document_chunks)

        # Step 3: Answer questions using local RAG
        logger.info("Step 3: Answering questions using local RAG...")
        answers = await rag_service.answer_multiple_questions(request.questions)

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
        "message": "Local RAG System - Pure Local Inference",
        "version": "2.0.0",
        "endpoints": {
            "health": "/health",
            "simple_health": "/simple-health", 
            "main": "/hackrx/run",
            "api_stats": "/api-stats"
        },
        "configuration": {
            "embedding_model": os.getenv("LOCAL_EMBEDDING_MODEL") or "❌ NOT CONFIGURED - Add LOCAL_EMBEDDING_MODEL to .env",
            "llm_model": os.getenv("LOCAL_LLM_MODEL") or "❌ NOT CONFIGURED - Add LOCAL_LLM_MODEL to .env",
            "device": os.getenv("DEVICE") or "❌ NOT CONFIGURED - Add DEVICE to .env",
            "quantization": os.getenv("USE_QUANTIZATION") or "❌ NOT CONFIGURED - Add USE_QUANTIZATION to .env",
            "quantization_bits": os.getenv("QUANTIZATION_BITS") or "❌ NOT CONFIGURED - Add QUANTIZATION_BITS to .env"
        },
        "features": [
            "100% Local Processing - No Cloud APIs",
            "GPU/CPU auto-detection",
            "8-bit quantization for memory efficiency",
            "PDF/DOCX document processing",
            "FAISS vector search",
            "Strict environment-based configuration"
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