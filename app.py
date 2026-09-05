from pathlib import Path
from tempfile import TemporaryDirectory

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from src.data_loader import load_all_documents
from src.embedding import EmbeddingPipeline
from src.search import ResearchAgent


st.set_page_config(page_title="Document Research Agent", layout="wide")
st.title("Intelligent Document Research Agent")
st.caption("LangGraph + LangChain documents + Sentence-Transformers + FAISS + optional Tavily")

uploaded_files = st.file_uploader("Upload PDF, TXT, CSV, or XLSX files", accept_multiple_files=True)

if "agent" not in st.session_state:
    st.session_state.agent = None
if "messages" not in st.session_state:
    st.session_state.messages = []

if st.button("Build research index", type="primary"):
    if not uploaded_files:
        st.warning("Upload at least one document first.")
    else:
        with st.spinner("Reading documents, chunking text, and building FAISS index..."):
            with TemporaryDirectory() as temp_dir:
                for file in uploaded_files:
                    Path(temp_dir, file.name).write_bytes(file.getvalue())
                docs = load_all_documents(temp_dir)
            pipeline = EmbeddingPipeline()
            chunks = pipeline.chunk_documents(docs)
            embeddings = pipeline.embed_chunks(chunks)
            st.session_state.agent = ResearchAgent(chunks, embeddings)
            st.session_state.messages = []
            st.success(f"Indexed {len(chunks)} chunks from {len(docs)} loaded document pages/rows.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("trace"):
            st.caption(str(message["trace"]))
        if message.get("metrics"):
            with st.expander("Evaluation Metrics"):
                cols = st.columns(4)
                m = message["metrics"]
                cols[0].metric("Retrieval Precision", f"{m['retrieval_precision']:.0%}")
                cols[1].metric("Faithfulness", f"{m['faithfulness']:.0%}")
                cols[2].metric("Answer Relevance", f"{m['answer_relevance']:.0%}")
                cols[3].metric("Latency", f"{m['latency_seconds']}s")
                st.caption(
                    f"Chunks retrieved: {m['total_chunks_retrieved']} | "
                    f"Relevant chunks used: {m['relevant_chunks_used']} | "
                    f"Retrieval attempts: {m['retrieval_attempts']}"
                )

question = st.chat_input("Ask a research question")
if question and st.session_state.agent:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.spinner("Planning, retrieving, answering, and verifying evidence..."):
        result = st.session_state.agent.ask(question)
    trace = {"plan": result["plan"], "supported": result["supported"], "retrieval_attempts": result["attempts"]}
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "trace": trace,
        "metrics": result.get("metrics"),
    })
    with st.chat_message("assistant"):
        st.markdown(result["answer"])
        st.caption(str(trace))
        if result.get("metrics"):
            with st.expander("Evaluation Metrics"):
                cols = st.columns(4)
                m = result["metrics"]
                cols[0].metric("Retrieval Precision", f"{m['retrieval_precision']:.0%}")
                cols[1].metric("Faithfulness", f"{m['faithfulness']:.0%}")
                cols[2].metric("Answer Relevance", f"{m['answer_relevance']:.0%}")
                cols[3].metric("Latency", f"{m['latency_seconds']}s")
                st.caption(
                    f"Chunks retrieved: {m['total_chunks_retrieved']} | "
                    f"Relevant chunks used: {m['relevant_chunks_used']} | "
                    f"Retrieval attempts: {m['retrieval_attempts']}"
                )
elif question:
    st.info("Build the research index before asking questions.")
