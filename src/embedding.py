from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
from langchain_core.documents import Document

class EmbeddingPipeline:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", chunk_size: int = 220, chunk_overlap: int = 40):
        self.model = SentenceTransformer(model_name)
        tokenizer = self.model.tokenizer
        self.splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            tokenizer=tokenizer,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def chunk_documents(self, documents: list[Document]) -> list[Document]:
        return self.splitter.split_documents(documents)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, normalize_embeddings=True).astype("float32")

    def embed_chunks(self, chunks: list[Document]) -> np.ndarray:
        texts = [chunk.page_content for chunk in chunks]
        return self.embed_texts(texts)
