import faiss
import numpy as np
from langchain_core.documents import Document


class FaissDocumentStore:
    def __init__(self, chunks: list[Document], embeddings: np.ndarray):
        self.chunks = chunks
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)

    def search(self, query_embedding: np.ndarray, k: int = 5) -> list[Document]:
        scores, indexes = self.index.search(query_embedding, k)
        results: list[Document] = []
        for score, idx in zip(scores[0], indexes[0]):
            if idx == -1:
                continue
            doc = self.chunks[idx]
            doc.metadata = {**doc.metadata, "score": float(score)}
            results.append(doc)
        return results
