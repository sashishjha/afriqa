from src.retrieval.bm25_retriever import BM25Retriever

# Lazy imports — DenseRetriever needs faiss which may not be installed locally
try:
    from src.retrieval.dense_retriever import DenseRetriever
    from src.retrieval.hybrid_retriever import HybridRetriever
except ImportError:
    pass
