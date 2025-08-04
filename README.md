# HackRX LLM Query-Retrieval System

An intelligent document processing and question answering system built with FastAPI, featuring PDF document processing, semantic search with embeddings, and LLM-powered question answering using Google's Gemma model.

## Features

- **PDF & DOCX Document Processing**: Download and extract text from PDF and DOCX documents via URL
- **Intelligent Text Chunking**: Smart text segmentation with overlapping chunks for better context preservation
- **Semantic Search**: Vector-based document retrieval using sentence transformers and FAISS
- **LLM Question Answering**: Powered by Google's Gemma 3n model for accurate, contextual responses
- **Answer Post-processing**: Intelligent cleaning and formatting of AI-generated responses
- **RESTful API**: Clean, documented API endpoints with authentication
- **Insurance Domain Optimized**: Specifically tuned for insurance policy document analysis

## Technology Stack

- **Backend**: FastAPI (Python 3.8+)
- **Document Processing**: PyMuPDF (fitz) for PDF text extraction, python-docx for DOCX processing
- **Embeddings**: SentenceTransformers with all-MiniLM-L6-v2 model
- **Vector Database**: FAISS for efficient similarity search
- **LLM**: Google Gemma 3n via Google AI API

## Prerequisites

- Python 3.8 or higher
- Google AI API key (for Gemma model access)
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
   GOOGLE_API_KEY=your_google_ai_api_key_here
   API_TOKEN=your_bearer_token_here
   ```

## Usage

### Starting the Server

```bash
python main.py
```

The API will be available at `http://localhost:8000`

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

**Note**: The system now supports both PDF (.pdf) and DOCX (.docx) documents. File type is automatically detected from the URL extension or file content.

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
   - Vector store management with FAISS
   - Semantic retrieval and answer generation
   - Response post-processing and cleaning

3. **API Layer** (`main.py`)
   - FastAPI application with CORS support
   - Bearer token authentication
   - Request/response validation with Pydantic

### Processing Pipeline

```
PDF/DOCX URL → Download → File Type Detection → Text Extraction → Cleaning → Chunking → Embeddings → Vector Store
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