# HackRX LLM Query-Retrieval System

An intelligent document processing and question answering system built with FastAPI, featuring PDF document processing, **hybrid search** (dense + sparse vectors), and LLM-powered question answering using Google's Gemma model.

## Features

- **PDF & DOCX Document Processing**: Download and extract text from PDF and DOCX documents via URL
- **Intelligent Text Chunking**: Smart text segmentation with overlapping chunks for better context preservation
- **🚀 Hybrid Search**: Advanced retrieval combining semantic (dense) and lexical (sparse) search for optimal relevance
- **Semantic Search**: Vector-based document retrieval using sentence transformers and Pinecone
- **LLM Question Answering**: Powered by Google's Gemma models for accurate, contextual responses
- **Answer Post-processing**: Intelligent cleaning and formatting of AI-generated responses
- **RESTful API**: Clean, documented API endpoints with authentication
- **Insurance Domain Optimized**: Specifically tuned for insurance policy document analysis
- **Automatic Reranking**: Uses Pinecone's reranking models for unified relevance scoring

## Technology Stack

- **Backend**: FastAPI (Python 3.8+)
- **Document Processing**: PyMuPDF (fitz) for PDF text extraction, python-docx for DOCX processing
- **Embeddings**: SentenceTransformers with all-MiniLM-L6-v2 model + Pinecone integrated models
- **Vector Database**: Pinecone with separate dense and sparse indexes for hybrid search
- **LLM**: Google Gemma models via Google AI API
- **Reranking**: Pinecone's BGE reranker for optimal result ordering

## Prerequisites

- Python 3.8 or higher
- Google AI API key (for Gemma model access)
- Pinecone API key (for vector storage)
- Required Python packages (see requirements.txt)

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd bajajHackrx
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   Create a `.env` file in the root directory:
   ```env
   # Basic Configuration
   GOOGLE_API_KEY=your_google_ai_api_key_here
   API_TOKEN=your_bearer_token_here
   PINECONE_API_KEY=your_pinecone_api_key_here
   PINECONE_ENVIRONMENT=us-east-1
   
   # Hybrid Search Configuration (NEW!)
   ENABLE_HYBRID_SEARCH=True                    # Enable hybrid search
   PINECONE_INDEX_BASE_NAME=bajaj-hackrx       # Base name for indexes
   HYBRID_TOP_K_PER_INDEX=40                   # Results per index
   HYBRID_FINAL_TOP_K=10                       # Final results after reranking
   HYBRID_ENABLE_RERANKING=True                # Enable reranking
   
   # Optional: Multiple API Keys for better rate limiting
   HF_API_TOKENS=token1,token2,token3
   GEMINI_API_TOKENS=key1,key2,key3
   ```

## Hybrid Search Architecture

The system now supports **hybrid search** which combines:

### 🔍 Dense Search (Semantic)
- Uses vector embeddings to understand meaning and context
- Great for conceptual queries like "payment terms" → finds "premium due dates"
- Powered by Pinecone's `llama-text-embed-v2` model

### 🔤 Sparse Search (Lexical) 
- Uses keyword matching for exact term retrieval
- Perfect for specific terminology like "30 days" or "deductible"
- Powered by Pinecone's `pinecone-sparse-english-v0` model

### 🎯 Automatic Reranking
- Combines dense and sparse results using `bge-reranker-v2-m3`
- Provides unified relevance scoring
- Deduplicates and optimizes result ordering

### 🔄 Search Flow
1. Query sent to both dense and sparse indexes (40 results each)
2. Results merged and deduplicated 
3. Reranked by relevance using Pinecone's reranking model
4. Top 10 most relevant chunks returned to LLM

## Usage

### Testing Hybrid Search

Before starting the main application, you can test the hybrid search functionality:

```bash
python test_hybrid_search.py
```

This will demonstrate:
- Dense vs sparse search results
- Reranking effectiveness  
- Performance statistics
- Comparison with traditional search

### Starting the Server

```bash
python main.py
```

The API will be available at `http://localhost:8000`

### Search Mode Selection

The system automatically uses hybrid search if `ENABLE_HYBRID_SEARCH=True` in your `.env` file:

- **Hybrid Mode** (Recommended): Dense + Sparse + Reranking
- **Traditional Mode**: Dense search only (backward compatible)

### API Documentation

Interactive API documentation is available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### API Endpoints

#### Health Check
```http
GET /health
```

#### Main Processing Endpoint
```http
POST /hackrx/run
```

**Request Body:**
```json
{
  "documents": "https://example.com/policy.pdf",
  "questions": [
    "What is the grace period for premium payment?",
    "What is the waiting period for pre-existing diseases?"
  ]
}
```

**Note**: The system now supports both PDF (.pdf) and DOCX (.docx) documents. File type is automatically detected from the URL extension or file content. With hybrid search enabled, you'll get more comprehensive and accurate results.

**Headers:**
```http
Authorization: Bearer your_token_here
Content-Type: application/json
```

**Response:**
```json
{
  "answers": [
    "The grace period for premium payment is thirty (30) days after the due date.",
    "There is a waiting period of thirty-six (36) months for pre-existing diseases."
  ]
}
```

## Architecture

### Core Components

1. **DocumentProcessor** (`core/document.py`)
   - PDF and DOCX downloading and text extraction
   - Automatic file type detection
   - Text cleaning and preprocessing
   - Intelligent text chunking with overlap

2. **RAGService** (`core/rag.py`)
   - Embedding generation using SentenceTransformers
   - Vector store management with Pinecone
   - Semantic retrieval and answer generation
   - Response post-processing and cleaning

3. **API Layer** (`main.py`)
   - FastAPI application with CORS support
   - Bearer token authentication
   - Request/response validation with Pydantic

### Processing Pipeline

```
PDF/DOCX URL → Download → File Type Detection → Text Extraction → Cleaning → Chunking → Embeddings → Pinecone Vector Store
                                                                                                          ↓
Question → Query Embedding → Similarity Search → Context Retrieval → LLM → Answer Cleaning
```

## Configuration

### Document Processing Parameters

```python
# In main.py
document_processor = DocumentProcessor(
    chunk_size=500,      # Maximum characters per chunk
    chunk_overlap=80     # Overlap between consecutive chunks
)
```

### RAG Service Parameters

```python
# In core/rag.py
rag_service = RAGService(
    embedding_model="all-MiniLM-L6-v2",  # SentenceTransformer model
    top_k=5,                             # Number of chunks to retrieve
    gemma_model="gemma-3n-e4b-it"        # Google Gemma model
)
```

### Generation Parameters

```python
generation_config = genai.types.GenerationConfig(
    temperature=0.1,        # Low temperature for factual responses
    top_p=1.0,
    top_k=12,              # Token sampling parameter
    max_output_tokens=200,  # Maximum response length
    stop_sequences=None,
)
```

## Performance Considerations

- **Chunk Size**: Balanced at 500 characters for optimal context vs. precision
- **Chunk Overlap**: 80 characters ensures continuity between chunks
- **Top-K Retrieval**: 5 chunks provide sufficient context without noise
- **Temperature**: 0.1 ensures factual, consistent responses
- **Max Tokens**: 200 tokens balance completeness with efficiency

### cURL Usage Example

```bash
curl -X POST "http://localhost:8000/hackrx/run" \
  -H "Authorization: Bearer your_token_here" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": "https://example.com/policy.pdf",
    "questions": ["What is the deductible amount?"]
  }'
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request