"""Dense retriever backed by SentenceTransformers + FAISS.
Modified for AMD ROCm (RBCCPS cluster, IISc - AMD MI210 GPUs).
Uses faiss-cpu (no faiss-gpu). SentenceTransformer runs on ROCm GPU via HIP.
"""

import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def _auto_device() -> str:
    """Return 'cuda' if a GPU is available (works with ROCm HIP), else 'cpu'."""
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


class DenseRetriever:
    def __init__(
        self,
        model_name: str = "sentence-transformers/LaBSE",
        device: str = None,
    ):
        if device is None:
            device = _auto_device()
        self.model = SentenceTransformer(model_name, device=device)
        self.index = None
        self.documents: list = []
        self.answers: list = []
        self.dim: int = 0

    # -- index building --

    def build_index(self, questions: list, answers: list, batch_size: int = 64):
        self.documents = list(questions)
        self.answers = list(answers)
        embs = self.model.encode(
            self.documents,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        ).astype(np.float32)
        self.dim = embs.shape[1]
        # FAISS CPU index - works identically on ROCm and CUDA systems
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(embs)

    # -- retrieval --

    def retrieve(self, query: str, top_k: int = 5) -> list:
        q_emb = self.model.encode(
            [query], normalize_embeddings=True,
        ).astype(np.float32)
        scores, idxs = self.index.search(q_emb, top_k)
        return [
            {
                "question": self.documents[i],
                "answer": self.answers[i],
                "score": float(s),
                "index": int(i),
            }
            for s, i in zip(scores[0], idxs[0])
            if i >= 0
        ]

    def batch_retrieve(self, queries: list, top_k: int = 5, batch_size: int = 64) -> list:
        all_results = []
        for start in tqdm(range(0, len(queries), batch_size), desc="Dense retrieval"):
            batch = queries[start : start + batch_size]
            q_embs = self.model.encode(
                batch, normalize_embeddings=True,
            ).astype(np.float32)
            scores, idxs = self.index.search(q_embs, top_k)
            for sc_row, id_row in zip(scores, idxs):
                all_results.append([
                    {
                        "question": self.documents[i],
                        "answer": self.answers[i],
                        "score": float(s),
                        "index": int(i),
                    }
                    for s, i in zip(sc_row, id_row)
                    if i >= 0
                ])
        return all_results

    # -- persistence --

    def save_index(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        faiss.write_index(self.index, path)

    def load_index(self, path: str, questions: list = None, answers: list = None):
        self.index = faiss.read_index(path)
        if questions:
            self.documents = list(questions)
        if answers:
            self.answers = list(answers)
        self.dim = self.index.d
