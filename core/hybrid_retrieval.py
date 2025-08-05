"""
Hybrid Search Implementation using Pinecone
Implements separate dense and sparse indexes for optimal search performance
"""

import logging
import os
import time
import uuid
import numpy as np
from typing import List, Dict, Optional, Tuple
from pinecone import Pinecone, ServerlessSpec
from huggingface_hub import InferenceClient
from core.api_key_manager import HuggingFaceAPIKeyManager, PineconeAPIKeyManager

logger = logging.getLogger(__name__)

class HybridSearchService:
    """
    Hybrid search service using separate dense and sparse Pinecone indexes
    Follows Pinecone's recommended approach for hybrid search
    """
    
    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        dense_model: str = "llama-text-embed-v2",
        sparse_model: str = "pinecone-sparse-english-v0",
        top_k_per_index: int = 40,
        final_top_k: int = 10,
        auto_cleanup: bool = True
    ):
        self.embedding_model = embedding_model
        self.dense_model = dense_model
        self.sparse_model = sparse_model
        self.top_k_per_index = top_k_per_index
        self.final_top_k = final_top_k
        self.auto_cleanup = auto_cleanup
        self.embedding_dim = 384  # Dimension for all-MiniLM-L6-v2
        
        # Initialize API key managers
        self._initialize_api_managers()
        
        # Initialize Pinecone client
        self._initialize_pinecone()
        
        # Initialize indexes
        self._initialize_indexes()
        
        # Session management
        self.current_session_id = None
        self.chunks = []
        
        logger.info("✅ Hybrid search service initialized successfully")

    def _initialize_api_managers(self):
        """Initialize API key managers"""
        try:
            # Hugging Face API Key Manager
            self.hf_key_manager = HuggingFaceAPIKeyManager()
            logger.info(f"Initialized HF API Key Manager with {len(self.hf_key_manager.api_keys)} keys")
        except Exception as e:
            logger.error(f"Failed to initialize HF API Key Manager: {str(e)}")
            # Fallback to single key approach
            hf_api_token = os.getenv("HF_API_TOKEN")
            if not hf_api_token:
                raise ValueError("Neither HF_API_TOKENS nor HF_API_TOKEN found in environment variables")
            self.hf_key_manager = None
            self.hf_client = InferenceClient(api_key=hf_api_token)
            logger.warning("Using fallback single HF API token")

    def _initialize_pinecone(self):
        """Initialize Pinecone client"""
        try:
            pinecone_api_key = os.getenv("PINECONE_API_KEY")
            if not pinecone_api_key:
                raise ValueError("PINECONE_API_KEY not found in environment variables")
            
            logger.info("Initializing Pinecone client for hybrid search...")
            self.pinecone_client = Pinecone(api_key=pinecone_api_key)
            self.pinecone_environment = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
            
            # Parse region correctly - handle both "us-east-1" and "us-east-1-aws" formats
            self.region = self.pinecone_environment
            if not self.region.endswith("-aws"):
                self.region = f"{self.region}-aws"
            
            # Index names
            base_name = os.getenv("PINECONE_INDEX_BASE_NAME", "bajaj-hackrx")
            self.dense_index_name = f"{base_name}-dense"
            self.sparse_index_name = f"{base_name}-sparse"
            
            logger.info(f"Pinecone client created for region: {self.region}")
            logger.info(f"Dense index name: {self.dense_index_name}")
            logger.info(f"Sparse index name: {self.sparse_index_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {str(e)}")
            raise ValueError(f"Pinecone initialization failed: {str(e)}")

    def _initialize_indexes(self):
        """Initialize or connect to dense and sparse indexes"""
        try:
            logger.info("Initializing dense and sparse indexes...")
            
            # Get existing indexes
            existing_indexes = self.pinecone_client.list_indexes()
            index_names = [index.name for index in existing_indexes]
            logger.info(f"Existing indexes: {index_names}")
            
            # Initialize dense index
            self._create_or_connect_dense_index(index_names)
            
            # Initialize sparse index
            self._create_or_connect_sparse_index(index_names)
            
            logger.info("✅ Both dense and sparse indexes initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize indexes: {str(e)}")
            raise Exception(f"Index initialization failed: {str(e)}")

    def _create_or_connect_dense_index(self, existing_indexes: List[str]):
        """Create or connect to dense index"""
        if self.dense_index_name not in existing_indexes:
            logger.info(f"Creating dense index: {self.dense_index_name}")
            
            # Check if integrated embedding models are available
            try:
                # Try to create with integrated embedding model
                self.pinecone_client.create_index_for_model(
                    name=self.dense_index_name,
                    cloud="aws",
                    region=self.pinecone_environment,
                    embed={
                        "model": self.dense_model,
                        "field_map": {"text": "chunk_text"}
                    }
                )
                self.use_integrated_dense = True
                logger.info(f"Created dense index with integrated model: {self.dense_model}")
            except Exception as e:
                logger.warning(f"Failed to create index with integrated model: {str(e)}")
                logger.info("Falling back to external embedding approach for dense index")
                
                # Fallback to external embeddings
                self.pinecone_client.create_index(
                    name=self.dense_index_name,
                    dimension=self.embedding_dim,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region=self.region
                    )
                )
                self.use_integrated_dense = False
                logger.info("Created dense index with external embeddings")
            
            # Wait for index to be ready
            self._wait_for_index_ready(self.dense_index_name)
        else:
            logger.info(f"Dense index already exists: {self.dense_index_name}")
            # Determine if it uses integrated embeddings (heuristic approach)
            try:
                index_desc = self.pinecone_client.describe_index(self.dense_index_name)
                self.use_integrated_dense = hasattr(index_desc, 'spec') and hasattr(index_desc.spec, 'embed')
            except:
                self.use_integrated_dense = False
        
        # Connect to dense index
        self.dense_index = self.pinecone_client.Index(self.dense_index_name)
        logger.info(f"Connected to dense index: {self.dense_index_name}")

    def _create_or_connect_sparse_index(self, existing_indexes: List[str]):
        """Create or connect to sparse index"""
        if self.sparse_index_name not in existing_indexes:
            logger.info(f"Creating sparse index: {self.sparse_index_name}")
            
            try:
                # Create with integrated sparse embedding model
                self.pinecone_client.create_index_for_model(
                    name=self.sparse_index_name,
                    cloud="aws",
                    region=self.pinecone_environment,
                    embed={
                        "model": self.sparse_model,
                        "field_map": {"text": "chunk_text"}
                    }
                )
                self.use_integrated_sparse = True
                logger.info(f"Created sparse index with integrated model: {self.sparse_model}")
            except Exception as e:
                logger.error(f"Failed to create sparse index with integrated model: {str(e)}")
                raise Exception(f"Sparse index creation failed. Integrated sparse models are required: {str(e)}")
            
            # Wait for index to be ready
            self._wait_for_index_ready(self.sparse_index_name)
        else:
            logger.info(f"Sparse index already exists: {self.sparse_index_name}")
            self.use_integrated_sparse = True
        
        # Connect to sparse index
        self.sparse_index = self.pinecone_client.Index(self.sparse_index_name)
        logger.info(f"Connected to sparse index: {self.sparse_index_name}")

    def _wait_for_index_ready(self, index_name: str, max_wait_time: int = 300):
        """Wait for index to be ready"""
        start_time = time.time()
        while time.time() - start_time < max_wait_time:
            try:
                status = self.pinecone_client.describe_index(index_name).status
                if status.get('ready', False):
                    logger.info(f"Index {index_name} is ready")
                    return
            except Exception as e:
                logger.warning(f"Error checking index status: {str(e)}")
            
            logger.info(f"Waiting for index {index_name} to be ready...")
            time.sleep(5)
        
        raise Exception(f"Index {index_name} did not become ready within {max_wait_time} seconds")

    def _generate_session_id(self) -> str:
        """Generate a unique session ID for document processing"""
        return f"hybrid_session_{int(time.time())}_{uuid.uuid4().hex[:8]}"

    async def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """Create embeddings for external dense vectors (fallback)"""
        if not self.hf_key_manager:
            return await self._create_embeddings_single_client(texts)
        
        logger.info(f"Creating embeddings for {len(texts)} texts via Hugging Face API")
        
        max_attempts = len(self.hf_key_manager.api_keys)
        last_error = None
        
        for attempt in range(max_attempts):
            api_key = self.hf_key_manager.get_next_available_key()
            
            if not api_key:
                logger.error("No available Hugging Face API keys")
                break
            
            try:
                client = InferenceClient(api_key=api_key)
                embeddings = client.feature_extraction(
                    text=texts,
                    model=self.embedding_model
                )
                
                self.hf_key_manager.mark_key_successful(api_key)
                embeddings_array = np.array(embeddings, dtype=np.float32)
                logger.info(f"Created embeddings with shape: {embeddings_array.shape}")
                return embeddings_array
                
            except Exception as e:
                error_msg = str(e)
                last_error = e
                
                logger.warning(f"HF API attempt {attempt + 1} failed: {error_msg}")
                
                if self.hf_key_manager.is_rate_limit_error(error_msg):
                    self.hf_key_manager.mark_key_rate_limited(api_key)
                    continue
                elif self.hf_key_manager.is_auth_error(error_msg):
                    self.hf_key_manager.mark_key_failed(api_key)
                    continue
                else:
                    continue
        
        if last_error:
            raise Exception(f"All Hugging Face API keys failed. Last error: {str(last_error)}")
        else:
            raise Exception("All Hugging Face API keys are unavailable")

    async def _create_embeddings_single_client(self, texts: List[str]) -> np.ndarray:
        """Fallback method for single client"""
        try:
            logger.info(f"Creating embeddings for {len(texts)} texts via single HF client")
            embeddings = self.hf_client.feature_extraction(
                text=texts,
                model=self.embedding_model
            )
            embeddings_array = np.array(embeddings, dtype=np.float32)
            return embeddings_array
        except Exception as e:
            logger.error(f"Error creating Hugging Face embeddings: {str(e)}")
            raise Exception(f"Error creating Hugging Face embeddings: {str(e)}")

    async def build_hybrid_vector_store(self, chunks: List[str]) -> None:
        """Build both dense and sparse vector stores from document chunks"""
        try:
            logger.info(f"Building hybrid vector store from {len(chunks)} chunks")
            if len(chunks) == 0:
                raise ValueError("No chunks provided to build vector store")
            
            self.chunks = chunks
            
            # Generate a new session ID
            self.current_session_id = self._generate_session_id()
            logger.info(f"Generated session ID: {self.current_session_id}")
            
            # Prepare records with session metadata
            records = []
            for i, chunk in enumerate(chunks):
                if not chunk or len(chunk.strip()) == 0:
                    logger.warning(f"Skipping empty chunk at index {i}")
                    continue
                    
                record = {
                    "_id": f"{self.current_session_id}_chunk_{i}",
                    "chunk_text": chunk,
                    "session_id": self.current_session_id,
                    "chunk_index": i,
                    "chunk_length": len(chunk),
                    "timestamp": int(time.time())
                }
                records.append(record)
            
            if len(records) == 0:
                raise ValueError("No valid chunks found after filtering empty chunks")
            
            logger.info(f"Prepared {len(records)} records for upserting")
            
            # Upsert to dense index
            await self._upsert_to_dense_index(records)
            
            # Upsert to sparse index
            await self._upsert_to_sparse_index(records)
            
            # Verify the data was uploaded by checking index stats
            try:
                dense_stats = self.dense_index.describe_index_stats()
                logger.info(f"Dense index after upsert: {dense_stats.total_vector_count} total vectors")
            except Exception as e:
                logger.warning(f"Could not get dense index stats: {str(e)}")
            
            try:
                sparse_stats = self.sparse_index.describe_index_stats()
                logger.info(f"Sparse index after upsert: {sparse_stats.total_vector_count} total vectors")
            except Exception as e:
                logger.warning(f"Could not get sparse index stats: {str(e)}")
            
            logger.info(f"✅ Hybrid vector store built with {len(chunks)} chunks for session: {self.current_session_id}")
            
        except Exception as e:
            logger.error(f"Error building hybrid vector store: {str(e)}")
            raise Exception(f"Error building hybrid vector store: {str(e)}")

    async def _upsert_to_dense_index(self, records: List[Dict]):
        """Upsert records to dense index"""
        try:
            logger.info("Upserting to dense index...")
            
            if self.use_integrated_dense:
                # Use integrated embedding - Pinecone handles the embedding
                namespace = f"session_{self.current_session_id}"
                self.dense_index.upsert_records(namespace, records)
                logger.info(f"Upserted {len(records)} records to dense index using integrated embeddings")
            else:
                # Use external embeddings
                texts = [record["chunk_text"] for record in records]
                embeddings = await self.create_embeddings(texts)
                
                # Prepare vectors for Pinecone upsert
                vectors_to_upsert = []
                for record, embedding in zip(records, embeddings):
                    metadata = {k: v for k, v in record.items() if k != "chunk_text"}
                    metadata["text"] = record["chunk_text"][:1000]  # Limit text size for metadata
                    
                    vectors_to_upsert.append({
                        "id": record["_id"],
                        "values": embedding.tolist(),
                        "metadata": metadata
                    })
                
                # Upsert in batches
                batch_size = 100
                for i in range(0, len(vectors_to_upsert), batch_size):
                    batch = vectors_to_upsert[i:i + batch_size]
                    self.dense_index.upsert(vectors=batch)
                    logger.info(f"Dense: Upserted batch {i // batch_size + 1}/{(len(vectors_to_upsert) + batch_size - 1) // batch_size}")
                
                logger.info(f"Upserted {len(records)} records to dense index using external embeddings")
                
        except Exception as e:
            logger.error(f"Error upserting to dense index: {str(e)}")
            raise

    async def _upsert_to_sparse_index(self, records: List[Dict]):
        """Upsert records to sparse index"""
        try:
            logger.info("Upserting to sparse index...")
            
            if self.use_integrated_sparse:
                # Use integrated sparse embedding - Pinecone handles the sparse vectorization
                namespace = f"session_{self.current_session_id}"
                self.sparse_index.upsert_records(namespace, records)
                logger.info(f"Upserted {len(records)} records to sparse index using integrated embeddings")
            else:
                raise Exception("Sparse index requires integrated sparse embedding models")
                
        except Exception as e:
            logger.error(f"Error upserting to sparse index: {str(e)}")
            raise

    async def hybrid_search(self, query: str) -> List[Dict]:
        """Perform hybrid search combining dense and sparse results"""
        try:
            if not self.current_session_id:
                raise ValueError("Vector store not initialized. Call build_hybrid_vector_store first.")
            
            logger.info(f"Performing hybrid search for query: {query[:50]}...")
            logger.info(f"Session ID: {self.current_session_id}, Available chunks: {len(self.chunks)}")
            
            # Search dense index (semantic search)
            dense_results = await self._search_dense_index(query)
            dense_count = len(dense_results.get('result', {}).get('hits', []))
            logger.info(f"Dense search returned {dense_count} results")
            
            # Search sparse index (lexical search)
            sparse_results = await self._search_sparse_index(query)
            sparse_count = len(sparse_results.get('result', {}).get('hits', []))
            logger.info(f"Sparse search returned {sparse_count} results")
            
            # Merge and deduplicate results
            merged_results = self._merge_and_deduplicate(dense_results, sparse_results)
            logger.info(f"Merged results: {len(merged_results)} unique results")
            
            # If no results from indexes, fall back to simple chunk search
            if len(merged_results) == 0:
                logger.warning("No results from hybrid search, falling back to simple chunk matching")
                fallback_results = self._fallback_chunk_search(query)
                return fallback_results
            
            # Rerank results
            reranked_results = await self._rerank_results(query, merged_results)
            
            logger.info(f"✅ Hybrid search completed: {len(reranked_results)} final results")
            return reranked_results
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {str(e)}")
            # Fallback to simple chunk search in case of errors
            logger.info("Falling back to simple chunk search due to error")
            try:
                return self._fallback_chunk_search(query)
            except Exception as fallback_error:
                logger.error(f"Fallback search also failed: {str(fallback_error)}")
                raise Exception(f"Error in hybrid search: {str(e)}")

    async def _search_dense_index(self, query: str) -> Dict:
        """Search dense index for semantic similarity"""
        try:
            logger.info(f"Searching dense index with integrated: {self.use_integrated_dense}")
            
            if self.use_integrated_dense:
                # Use integrated embedding
                namespace = f"session_{self.current_session_id}"
                logger.info(f"Searching namespace: {namespace}")
                
                results = self.dense_index.search(
                    namespace=namespace,
                    query={
                        "top_k": self.top_k_per_index,
                        "inputs": {
                            "text": query
                        }
                    }
                )
                hits_count = len(results.get('result', {}).get('hits', []))
                logger.info(f"Dense search (integrated) completed: {hits_count} results")
                
                # Log first few results for debugging
                if hits_count > 0:
                    first_hit = results['result']['hits'][0]
                    logger.info(f"Top dense result: {first_hit.get('_score', 0):.3f} - {first_hit.get('fields', {}).get('chunk_text', '')[:100]}...")
                
                return results
            else:
                # Use external embeddings
                logger.info("Using external embeddings for dense search")
                query_embedding = await self.create_embeddings([query])
                
                results = self.dense_index.query(
                    vector=query_embedding[0].tolist(),
                    top_k=self.top_k_per_index,
                    include_metadata=True,
                    filter={"session_id": self.current_session_id}
                )
                
                # Convert to same format as integrated search
                formatted_results = {
                    "result": {
                        "hits": [
                            {
                                "_id": match.id,
                                "_score": float(match.score),
                                "fields": {
                                    "chunk_text": match.metadata.get("text", "")
                                }
                            }
                            for match in results.matches
                        ]
                    }
                }
                hits_count = len(formatted_results['result']['hits'])
                logger.info(f"Dense search (external) completed: {hits_count} results")
                
                # Log first few results for debugging
                if hits_count > 0:
                    first_hit = formatted_results['result']['hits'][0]
                    logger.info(f"Top dense result: {first_hit.get('_score', 0):.3f} - {first_hit.get('fields', {}).get('chunk_text', '')[:100]}...")
                
                return formatted_results
                
        except Exception as e:
            logger.error(f"Error searching dense index: {str(e)}")
            # Return empty results instead of raising
            return {"result": {"hits": []}}

    async def _search_sparse_index(self, query: str) -> Dict:
        """Search sparse index for lexical similarity"""
        try:
            namespace = f"session_{self.current_session_id}"
            logger.info(f"Searching sparse index in namespace: {namespace}")
            
            results = self.sparse_index.search(
                namespace=namespace,
                query={
                    "top_k": self.top_k_per_index,
                    "inputs": {
                        "text": query
                    }
                }
            )
            hits_count = len(results.get('result', {}).get('hits', []))
            logger.info(f"Sparse search completed: {hits_count} results")
            
            # Log first few results for debugging
            if hits_count > 0:
                first_hit = results['result']['hits'][0]
                logger.info(f"Top sparse result: {first_hit.get('_score', 0):.3f} - {first_hit.get('fields', {}).get('chunk_text', '')[:100]}...")
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching sparse index: {str(e)}")
            # Return empty results instead of raising
            return {"result": {"hits": []}}

    def _merge_and_deduplicate(self, dense_results: Dict, sparse_results: Dict) -> List[Dict]:
        """Merge and deduplicate dense and sparse search results"""
        try:
            dense_hits = dense_results.get('result', {}).get('hits', [])
            sparse_hits = sparse_results.get('result', {}).get('hits', [])
            
            # Deduplicate by _id
            deduped_hits = {hit['_id']: hit for hit in dense_hits + sparse_hits}.values()
            
            # Sort by _score descending
            sorted_hits = sorted(deduped_hits, key=lambda x: x['_score'], reverse=True)
            
            # Transform to format for reranking
            result = [
                {
                    '_id': hit['_id'], 
                    'chunk_text': hit['fields']['chunk_text']
                } 
                for hit in sorted_hits
            ]
            
            logger.info(f"Merged {len(dense_hits)} dense + {len(sparse_hits)} sparse = {len(result)} unique results")
            return result
            
        except Exception as e:
            logger.error(f"Error merging results: {str(e)}")
            raise

    async def _rerank_results(self, query: str, merged_results: List[Dict]) -> List[Dict]:
        """Rerank merged results using Pinecone's reranking model"""
        try:
            if len(merged_results) == 0:
                return []
            
            logger.info(f"Reranking {len(merged_results)} results...")
            
            # Use Pinecone's reranking service
            rerank_response = self.pinecone_client.inference.rerank(
                model="bge-reranker-v2-m3",
                query=query,
                documents=merged_results,
                rank_fields=["chunk_text"],
                top_n=self.final_top_k,
                return_documents=True,
                parameters={
                    "truncate": "END"
                }
            )
            
            # Format results
            reranked_results = []
            for item in rerank_response.data:
                # Get the full chunk text from our stored chunks
                chunk_id = item['document']['_id']
                chunk_index = int(chunk_id.split('_chunk_')[-1])
                
                full_chunk_text = ""
                if chunk_index < len(self.chunks):
                    full_chunk_text = self.chunks[chunk_index]
                else:
                    full_chunk_text = item['document']['chunk_text']
                
                reranked_results.append({
                    "chunk": full_chunk_text,
                    "score": float(item['score']),
                    "index": chunk_index,
                    "vector_id": chunk_id,
                    "rerank_score": float(item['score'])
                })
            
            logger.info(f"Reranking completed: {len(reranked_results)} final results")
            return reranked_results
            
        except Exception as e:
            logger.warning(f"Reranking failed, returning top merged results: {str(e)}")
            # Fallback to top merged results without reranking
            fallback_results = []
            for i, item in enumerate(merged_results[:self.final_top_k]):
                chunk_id = item['_id']
                chunk_index = int(chunk_id.split('_chunk_')[-1])
                
                full_chunk_text = ""
                if chunk_index < len(self.chunks):
                    full_chunk_text = self.chunks[chunk_index]
                else:
                    full_chunk_text = item['chunk_text']
                
                fallback_results.append({
                    "chunk": full_chunk_text,
                    "score": 1.0 - (i * 0.1),  # Simple decreasing score
                    "index": chunk_index,
                    "vector_id": chunk_id,
                    "rerank_score": 1.0 - (i * 0.1)
                })
            
            return fallback_results

    def _fallback_chunk_search(self, query: str) -> List[Dict]:
        """Fallback search using simple text matching when vector search fails"""
        try:
            if not self.chunks:
                logger.warning("No chunks available for fallback search")
                return []
            
            logger.info(f"Performing fallback chunk search for query: {query[:50]}...")
            
            # Simple text similarity scoring
            query_lower = query.lower()
            query_words = set(query_lower.split())
            
            scored_chunks = []
            for i, chunk in enumerate(self.chunks):
                chunk_lower = chunk.lower()
                chunk_words = set(chunk_lower.split())
                
                # Calculate simple overlap score
                overlap = len(query_words.intersection(chunk_words))
                total_words = len(query_words)
                
                # Calculate keyword presence score
                keyword_score = overlap / max(total_words, 1)
                
                # Calculate substring match score
                substring_score = 1.0 if query_lower in chunk_lower else 0.0
                
                # Combined score
                final_score = (keyword_score * 0.7) + (substring_score * 0.3)
                
                if final_score > 0:  # Only include chunks with some relevance
                    scored_chunks.append({
                        "chunk": chunk,
                        "score": final_score,
                        "index": i,
                        "vector_id": f"{self.current_session_id}_chunk_{i}",
                        "rerank_score": final_score
                    })
            
            # Sort by score and return top results
            scored_chunks.sort(key=lambda x: x['score'], reverse=True)
            top_chunks = scored_chunks[:self.final_top_k]
            
            logger.info(f"Fallback search found {len(top_chunks)} relevant chunks")
            return top_chunks
            
        except Exception as e:
            logger.error(f"Error in fallback chunk search: {str(e)}")
            return []

    def debug_index_contents(self) -> Dict:
        """Debug method to check if data exists in indexes"""
        debug_info = {
            "session_id": self.current_session_id,
            "chunks_count": len(self.chunks),
            "dense_index": {},
            "sparse_index": {}
        }
        
        try:
            # Check dense index stats
            dense_stats = self.dense_index.describe_index_stats()
            debug_info["dense_index"] = {
                "total_vectors": dense_stats.total_vector_count,
                "dimension": getattr(dense_stats, 'dimension', 'unknown'),
                "namespaces": dict(getattr(dense_stats, 'namespaces', {}))
            }
        except Exception as e:
            debug_info["dense_index"]["error"] = str(e)
        
        try:
            # Check sparse index stats
            sparse_stats = self.sparse_index.describe_index_stats()
            debug_info["sparse_index"] = {
                "total_vectors": sparse_stats.total_vector_count,
                "namespaces": dict(getattr(sparse_stats, 'namespaces', {}))
            }
        except Exception as e:
            debug_info["sparse_index"]["error"] = str(e)
        
        logger.info(f"Debug index contents: {debug_info}")
        return debug_info

    def cleanup_session(self, session_id: Optional[str] = None):
        """Clean up vectors from a specific session in both indexes"""
        try:
            target_session = session_id or self.current_session_id
            if not target_session:
                logger.warning("No session ID provided for cleanup")
                return
            
            logger.info(f"Cleaning up hybrid search session: {target_session}")
            
            if self.use_integrated_dense or self.use_integrated_sparse:
                # For integrated indexes, delete by namespace
                namespace = f"session_{target_session}"
                
                try:
                    self.dense_index.delete(namespace=namespace, delete_all=True)
                    logger.info(f"Cleaned up dense index namespace: {namespace}")
                except Exception as e:
                    logger.warning(f"Failed to cleanup dense index: {str(e)}")
                
                try:
                    self.sparse_index.delete(namespace=namespace, delete_all=True)
                    logger.info(f"Cleaned up sparse index namespace: {namespace}")
                except Exception as e:
                    logger.warning(f"Failed to cleanup sparse index: {str(e)}")
            else:
                # For external embeddings, delete by filter
                try:
                    self.dense_index.delete(filter={"session_id": target_session})
                    logger.info(f"Cleaned up dense index vectors for session: {target_session}")
                except Exception as e:
                    logger.warning(f"Failed to cleanup dense index: {str(e)}")
            
            if target_session == self.current_session_id:
                self.current_session_id = None
                self.chunks = []
            
            logger.info(f"✅ Hybrid search session cleanup completed: {target_session}")
            
        except Exception as e:
            logger.error(f"Error cleaning up hybrid search session: {str(e)}")

    def get_statistics(self) -> Dict:
        """Get statistics about the hybrid search service"""
        stats = {
            "service": "HybridSearchService",
            "dense_index": {
                "name": self.dense_index_name,
                "integrated_embeddings": self.use_integrated_dense,
                "model": self.dense_model if self.use_integrated_dense else self.embedding_model
            },
            "sparse_index": {
                "name": self.sparse_index_name,
                "integrated_embeddings": self.use_integrated_sparse,
                "model": self.sparse_model if self.use_integrated_sparse else "external"
            },
            "current_session": self.current_session_id,
            "top_k_per_index": self.top_k_per_index,
            "final_top_k": self.final_top_k,
            "auto_cleanup": self.auto_cleanup
        }
        
        # Get index statistics
        try:
            dense_stats = self.dense_index.describe_index_stats()
            stats["dense_index"]["total_vectors"] = dense_stats.total_vector_count
            stats["dense_index"]["dimension"] = dense_stats.dimension
        except Exception as e:
            stats["dense_index"]["error"] = f"Unable to get stats: {str(e)}"
        
        try:
            sparse_stats = self.sparse_index.describe_index_stats()
            stats["sparse_index"]["total_vectors"] = sparse_stats.total_vector_count
        except Exception as e:
            stats["sparse_index"]["error"] = f"Unable to get stats: {str(e)}"
        
        # API key statistics
        if self.hf_key_manager:
            stats["huggingface"] = self.hf_key_manager.get_statistics()
        else:
            stats["huggingface"] = {
                "mode": "single_key_fallback",
                "message": "Using single HF_API_TOKEN"
            }
        
        return stats


def create_hybrid_search_service() -> HybridSearchService:
    """Factory function to create hybrid search service"""
    auto_cleanup = os.getenv("AUTO_CLEANUP_VECTORS", "True").lower() in ("true", "1", "yes", "on")
    top_k_per_index = int(os.getenv("HYBRID_TOP_K_PER_INDEX", "40"))
    final_top_k = int(os.getenv("HYBRID_FINAL_TOP_K", "10"))
    
    return HybridSearchService(
        top_k_per_index=top_k_per_index,
        final_top_k=final_top_k,
        auto_cleanup=auto_cleanup
    )
