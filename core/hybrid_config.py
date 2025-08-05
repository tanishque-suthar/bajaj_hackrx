"""
Configuration for Hybrid Search Implementation
Centralized configuration for dense and sparse search parameters
"""

import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class HybridSearchConfig:
    """Configuration class for hybrid search settings"""
    
    # Embedding models
    external_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    dense_model: str = "llama-text-embed-v2"  # Pinecone integrated model
    sparse_model: str = "pinecone-sparse-english-v0"  # Pinecone integrated model
    
    # Search parameters
    top_k_per_index: int = 40  # Number of results to retrieve from each index
    final_top_k: int = 10      # Final number of results after reranking
    
    # Pinecone settings
    pinecone_environment: str = "us-east-1"
    pinecone_index_base_name: str = "bajaj-hackrx"
    
    # Reranking
    rerank_model: str = "bge-reranker-v2-m3"
    enable_reranking: bool = True
    
    # Performance settings
    batch_size: int = 100
    max_wait_time: int = 300  # Seconds to wait for index creation
    
    # Session management
    auto_cleanup: bool = True
    
    # Embedding dimensions
    external_embedding_dim: int = 384  # For all-MiniLM-L6-v2
    
    @classmethod
    def from_env(cls) -> 'HybridSearchConfig':
        """Create configuration from environment variables"""
        return cls(
            external_embedding_model=os.getenv("HYBRID_EXTERNAL_EMBEDDING_MODEL", cls.external_embedding_model),
            dense_model=os.getenv("HYBRID_DENSE_MODEL", cls.dense_model),
            sparse_model=os.getenv("HYBRID_SPARSE_MODEL", cls.sparse_model),
            top_k_per_index=int(os.getenv("HYBRID_TOP_K_PER_INDEX", str(cls.top_k_per_index))),
            final_top_k=int(os.getenv("HYBRID_FINAL_TOP_K", str(cls.final_top_k))),
            pinecone_environment=os.getenv("PINECONE_ENVIRONMENT", cls.pinecone_environment),
            pinecone_index_base_name=os.getenv("PINECONE_INDEX_BASE_NAME", cls.pinecone_index_base_name),
            rerank_model=os.getenv("HYBRID_RERANK_MODEL", cls.rerank_model),
            enable_reranking=os.getenv("HYBRID_ENABLE_RERANKING", "True").lower() in ("true", "1", "yes", "on"),
            batch_size=int(os.getenv("HYBRID_BATCH_SIZE", str(cls.batch_size))),
            max_wait_time=int(os.getenv("HYBRID_MAX_WAIT_TIME", str(cls.max_wait_time))),
            auto_cleanup=os.getenv("AUTO_CLEANUP_VECTORS", "True").lower() in ("true", "1", "yes", "on"),
            external_embedding_dim=int(os.getenv("HYBRID_EXTERNAL_EMBEDDING_DIM", str(cls.external_embedding_dim)))
        )
    
    @property
    def dense_index_name(self) -> str:
        """Get the dense index name"""
        return f"{self.pinecone_index_base_name}-dense"
    
    @property
    def sparse_index_name(self) -> str:
        """Get the sparse index name"""
        return f"{self.pinecone_index_base_name}-sparse"
    
    @property
    def region(self) -> str:
        """Get the properly formatted region"""
        if not self.pinecone_environment.endswith("-aws"):
            return f"{self.pinecone_environment}-aws"
        return self.pinecone_environment
    
    def validate(self) -> None:
        """Validate configuration settings"""
        if self.top_k_per_index <= 0:
            raise ValueError("top_k_per_index must be positive")
        
        if self.final_top_k <= 0:
            raise ValueError("final_top_k must be positive")
        
        if self.final_top_k > self.top_k_per_index * 2:
            raise ValueError("final_top_k cannot exceed total possible results from both indexes")
        
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        
        if self.external_embedding_dim <= 0:
            raise ValueError("external_embedding_dim must be positive")
        
        required_env_vars = ["PINECONE_API_KEY"]
        for var in required_env_vars:
            if not os.getenv(var):
                raise ValueError(f"Required environment variable {var} is not set")


# Global configuration instance
DEFAULT_CONFIG = HybridSearchConfig.from_env()
