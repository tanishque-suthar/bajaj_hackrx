# Local RAG System - Pure Local Inference 🚀

An intelligent document processing and question-answering system that runs **completely locally** using state-of-the-art models without any cloud API dependencies. Built with FastAPI, featuring GPU/CPU auto-detection, Qwen3 embeddings, and Llama-3.1-8B for powerful local inference.

## ✨ Features

- **🏠 100% Local Inference**: No cloud APIs, complete privacy and control
- **🧠 Advanced Models**: 
  - **Qwen/Qwen3-Embedding-0.6B** for high-quality embeddings
  - **meta-llama/Llama-3.1-8B-Instruct** for intelligent Q&A
- **⚡ GPU/CPU Auto-Detection**: Automatically uses best available hardware
- **🗜️ Memory Optimization**: 8-bit quantization for VRAM efficiency
- **📄 Document Processing**: PDF & DOCX support with intelligent chunking
- **🔍 Semantic Search**: FAISS vector database for fast similarity search
- **🔗 RESTful API**: Clean FastAPI endpoints with authentication
- **💾 Local Caching**: Models cached locally, download once and use forever

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Document      │    │   Qwen3-Embedding │    │  Llama-3.1-8B   │
│   Processing    │───▶│   (Local)        │───▶│  (Local)        │
│   (PDF/DOCX)    │    │   Vector Store   │    │  Answer Gen     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🔧 Technology Stack

- **Backend**: FastAPI (Python 3.8+)
- **Models**: 
  - Qwen3-Embedding-0.6B (Alibaba)
  - Llama-3.1-8B-Instruct (Meta)
- **ML Framework**: PyTorch, Transformers, Sentence-Transformers
- **Vector Database**: FAISS for similarity search
- **Document Processing**: PyMuPDF, python-docx
- **Quantization**: BitsAndBytesConfig for memory efficiency

## 📋 Prerequisites

### Hardware Requirements
- **Minimum**: 8GB RAM, any CPU
- **Recommended**: 16GB+ RAM, GPU with 4GB+ VRAM
- **Storage**: ~20GB for models (downloaded once)

### Software Requirements
- Python 3.8+ 
- Windows/Linux/macOS
- CUDA toolkit (optional, for GPU acceleration)

## 🚀 Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/your-username/bajaj_hackrx.git
cd bajaj_hackrx
```

### 2. Setup Virtual Environment
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment
```bash
# Setup Hugging Face token (needed for Llama access)
python setup_hf_token.py
```

### 5. Test Local System
```bash
# Test the local RAG system
python test_local_rag.py
```

### 6. Start API Server
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## ⚙️ Configuration

The system uses environment variables in `.env` file:

```env
# Hugging Face token (required for Llama-3.1-8B)
HUGGINGFACE_TOKEN=hf_your_token_here

# Model cache directory (optional)
HF_HOME=D:\huggingface-cache

# API authentication
API_TOKEN=your_secure_api_token

# Model configuration
LOCAL_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B
LOCAL_LLM_MODEL=meta-llama/Llama-3.1-8B-Instruct

# Hardware settings
DEVICE=auto  # auto, cuda, cpu
USE_QUANTIZATION=true
QUANTIZATION_BITS=8

# Generation parameters
MAX_NEW_TOKENS=200
TEMPERATURE=0.1
TOP_P=0.9
TOP_K=50

# RAG settings
TOP_K_CHUNKS=5
CHUNK_SIZE=500
CHUNK_OVERLAP=80
```

## 🧠 How It Works

### 1. **Document Processing**
```python
# Process PDF/DOCX documents
chunks = document_processor.extract_chunks(pdf_content)
# Creates 500-char chunks with 80-char overlap for better context
```

### 2. **Embedding Creation** 
```python
# Convert text to vectors using Qwen3-Embedding
embeddings = await embedding_service.create_embeddings(chunks)
# Uses query-specific prompts for better retrieval performance
```

### 3. **Vector Search**
```python
# Find most relevant chunks using FAISS
relevant_chunks = await rag_service.retrieve_relevant_chunks(query)
# Returns top-5 most similar document sections
```

### 4. **Answer Generation**
```python
# Generate answers using Llama-3.1-8B locally
answer = await llm_service.generate_answer(prompt)
# Uses optimized Llama-3.1 chat template for best results
```

## 📡 API Usage

### Start the Server
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### API Documentation
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

### Main Endpoint

**Process Document & Answer Questions**
```http
POST /hackrx/run
Content-Type: application/json
Authorization: Bearer your_api_token

{
  "documents": "https://example.com/document.pdf",
  "questions": [
    "What is the grace period for premium payment?",
    "What are the exclusions in the policy?"
  ]
}
```

**Response**
```json
{
  "results": [
    {
      "question": "What is the grace period for premium payment?",
      "answer": "The grace period for premium payment is 30 days from the due date.",
      "processing_time": 2.1
    }
  ],
  "total_processing_time": 2.1,
  "system_info": {
    "service": "Local RAG",
    "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
    "llm_model": "meta-llama/Llama-3.1-8B-Instruct",
    "device": "cuda"
  }
}
```

### Health Check
```http
GET /health
```

## 🛠️ Development

### Project Structure
```
bajaj_hackrx/
├── core/
│   ├── local_rag.py      # Main local RAG implementation
│   └── document.py       # Document processing utilities
├── main.py               # FastAPI application
├── test_local_rag.py     # Testing script
├── setup_hf_token.py     # HF token setup utility
├── requirements.txt      # Dependencies
├── .env                  # Configuration
└── README.md            # This file
```

### Running Tests
```bash
# Test local RAG system
python test_local_rag.py

# Test with sample PDF
python test_local_rag.py --file sample.pdf

# Test API endpoints
curl -X POST "http://localhost:8000/hackrx/run" \
  -H "Authorization: Bearer your_token" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": "https://example.com/test.pdf",
    "questions": ["What is this document about?"]
  }'
```

## 🔧 Hardware Optimization

### GPU Configuration
- **Automatic Detection**: System auto-detects CUDA availability
- **8-bit Quantization**: Reduces VRAM usage by ~50%
- **Memory Management**: Automatic cleanup and garbage collection

### Performance Tips
1. **For GPU Users**: Ensure CUDA toolkit is installed
2. **For CPU Users**: Increase thread count in config
3. **Memory Issues**: Reduce batch sizes or use 4-bit quantization
4. **Storage**: Use SSD for model cache for faster loading

## 🐛 Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```bash
# Solution: Enable quantization in .env
USE_QUANTIZATION=true
QUANTIZATION_BITS=8
```

**2. Model Download Fails**
```bash
# Check your HF token
python setup_hf_token.py --test

# Manually set cache directory
export HF_HOME=/path/to/cache
```

**3. Slow Performance**
```bash
# Check if GPU is being used
python -c "import torch; print(torch.cuda.is_available())"

# Monitor GPU usage
nvidia-smi
```

**4. Import Errors**
```bash
# Reinstall dependencies
pip install -r requirements.txt --upgrade
```

## 🚀 Deployment

### Local Development
```bash
uvicorn main:app --reload --port 8000
```

### Production Deployment
```bash
# With Gunicorn
gunicorn main:app -w 1 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000

# With Docker (create Dockerfile)
docker build -t local-rag .
docker run -p 8000:8000 -v /path/to/models:/models local-rag
```

### Cloudflare Tunnel (Coming Soon)
```bash
# Will be added in next iteration
cloudflared tunnel --url http://localhost:8000
```

## 📊 Performance Metrics

### Model Sizes
- **Qwen3-Embedding-0.6B**: ~1.2GB
- **Llama-3.1-8B-Instruct**: ~15GB
- **Total Storage**: ~20GB (including cache)

### Speed Benchmarks
- **GPU (RTX 3060Ti)**: ~2-3s per question
- **CPU (i5-10th gen)**: ~10-15s per question
- **Embedding Speed**: ~100ms per document chunk

### Memory Usage
- **With Quantization**: ~6-8GB VRAM
- **Without Quantization**: ~12-15GB VRAM
- **CPU Mode**: ~8-12GB RAM

## 🔒 Privacy & Security

- **🔐 Complete Privacy**: All processing happens locally
- **🚫 No Data Transmission**: Documents never leave your machine
- **🔑 Token Security**: HF token only used for model downloads
- **🛡️ API Authentication**: Bearer token for endpoint security

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙋‍♂️ Support

- **Issues**: [GitHub Issues](https://github.com/your-username/bajaj_hackrx/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/bajaj_hackrx/discussions)
- **Email**: support@yourproject.com

## 🎯 Future Roadmap

- [ ] **Reranker Integration**: BAAI/bge-reranker-v2-m3
- [ ] **Cloudflare Tunnel**: Easy public deployment
- [ ] **Model Switching**: Runtime model swapping
- [ ] **Batch Processing**: Multiple document support
- [ ] **Web Interface**: Simple UI for non-developers
- [ ] **Docker Support**: Containerized deployment
- [ ] **API Rate Limiting**: Production-ready features

---

**Built with ❤️ for local AI inference and privacy-first document processing**
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