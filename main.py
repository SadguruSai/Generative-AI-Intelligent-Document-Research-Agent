from src.data_loader import load_all_documents
from src.embedding import EmbeddingPipeline
from src.search import ResearchAgent


def main():
    docs = load_all_documents("data/samples")
    pipeline = EmbeddingPipeline()
    chunks = pipeline.chunk_documents(docs)
    embeddings = pipeline.embed_chunks(chunks)
    agent = ResearchAgent(chunks, embeddings)
    result = agent.ask("What are the main topics in these documents?")
    print(result["answer"])


if __name__ == "__main__":
    main()
