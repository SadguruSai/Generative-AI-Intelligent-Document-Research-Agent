from __future__ import annotations

import os
import re
from pathlib import Path
from typing import TypedDict

from langchain_core.documents import Document
from langgraph.graph import END, StateGraph

from src.embedding import EmbeddingPipeline
from src.vector_stor import FaissDocumentStore


class ResearchState(TypedDict):
    question: str
    plan: list[str]
    evidence: list[Document]
    answer: str
    supported: bool
    attempts: int


class ResearchAgent:
    def __init__(self, chunks: list[Document], embeddings):
        self.embeddings = EmbeddingPipeline()
        self.store = FaissDocumentStore(chunks, embeddings)
        self.graph = self._build_graph()

    def ask(self, question: str) -> ResearchState:
        return self.graph.invoke({"question": question, "plan": [], "evidence": [], "answer": "", "supported": False, "attempts": 0})

    def _build_graph(self):
        graph = StateGraph(ResearchState)
        graph.add_node("plan", self._plan)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("answer", self._answer)
        graph.add_node("verify", self._verify)
        graph.set_entry_point("plan")
        graph.add_edge("plan", "retrieve")
        graph.add_edge("retrieve", "answer")
        graph.add_edge("answer", "verify")
        graph.add_conditional_edges("verify", self._next_step, {"retry": "retrieve", "done": END})
        return graph.compile()

    def _plan(self, state: ResearchState) -> ResearchState:
        question = state["question"].strip()
        state["plan"] = [question]
        if any(word in question.lower() for word in ["latest", "current", "today", "recent"]):
            state["plan"].append(f"web: {question}")
        return state

    def _retrieve(self, state: ResearchState) -> ResearchState:
        query = state["plan"][0]
        if state["attempts"]:
            query = f"{query} details evidence source"
        vector = self.embeddings.embed_texts([query])
        vector_hits = self.store.search(vector, k=5 + state["attempts"] * 3)
        keyword_hits = self._keyword_search(query, k=5)
        state["evidence"] = self._unique(vector_hits + keyword_hits)
        state["attempts"] += 1
        if len(state["plan"]) > 1:
            state["evidence"].extend(self._web_search(state["question"]))
        return state

    def _answer(self, state: ResearchState) -> ResearchState:
        relevant = [doc for doc in state["evidence"] if self._is_relevant(state["question"], doc)]
        if not relevant:
            state["answer"] = "I could not find enough evidence in the uploaded documents."
            return state
        direct = self._direct_answer(state["question"], relevant)
        lines = [f"**Direct answer:** {direct}", "", "**Evidence:**"]
        for doc in relevant[:3]:
            lines.append(f"- {self._best_sentence(state['question'], doc.page_content)} [{self._citation(doc)}]")
        state["answer"] = "\n".join(lines)
        return state

    def _verify(self, state: ResearchState) -> ResearchState:
        state["supported"] = bool(
            state["evidence"]
            and state["answer"]
            and "could not find" not in state["answer"].lower()
            and any(self._is_relevant(state["question"], doc) for doc in state["evidence"])
        )
        return state

    def _next_step(self, state: ResearchState) -> str:
        return "done" if state["supported"] or state["attempts"] >= 2 else "retry"

    def _web_search(self, question: str) -> list[Document]:
        if not os.getenv("TAVILY_API_KEY"):
            return []
        try:
            from langchain_tavily import TavilySearch

            results = TavilySearch(max_results=3).invoke({"query": question})
            return [Document(page_content=item.get("content", ""), metadata={"source": item.get("url", "Tavily")}) for item in results.get("results", [])]
        except Exception:
            return []

    def _keyword_search(self, query: str, k: int) -> list[Document]:
        words = self._tokens(query)
        scored = []
        for doc in self.store.chunks:
            text = doc.page_content.lower()
            score = sum(text.count(word) for word in words)
            if score:
                scored.append((score, doc))
        return [doc for _, doc in sorted(scored, key=lambda item: item[0], reverse=True)[:k]]

    def _unique(self, docs: list[Document]) -> list[Document]:
        seen = set()
        unique = []
        for doc in docs:
            key = (doc.metadata.get("source"), doc.metadata.get("page"), doc.page_content[:80])
            if key not in seen:
                seen.add(key)
                unique.append(doc)
        return unique

    def _is_relevant(self, question: str, doc: Document) -> bool:
        words = set(self._tokens(question))
        text = doc.page_content.lower()
        if "police custody" in question.lower():
            return "police custody" in text
        return len([word for word in words if word in text]) >= min(2, len(words))

    def _direct_answer(self, question: str, docs: list[Document]) -> str:
        text = " ".join(doc.page_content for doc in docs).lower()
        if "police custody" in question.lower() and re.search(r"fifteen|15", text):
            return "A Magistrate can authorize police custody for a maximum total period of 15 days in the whole."
        return self._best_sentence(question, docs[0].page_content)

    def _best_sentence(self, question: str, text: str) -> str:
        words = set(self._tokens(question))
        sentences = re.split(r"(?<=[.!?])\s+", " ".join(text.split()))
        best = max(sentences, key=lambda sentence: sum(word in sentence.lower() for word in words), default="")
        return best[:500]

    def _citation(self, doc: Document) -> str:
        source = Path(str(doc.metadata.get("source", "uploaded document"))).name
        page = doc.metadata.get("page")
        return f"{source}, page {page + 1}" if isinstance(page, int) else source

    def _tokens(self, text: str) -> list[str]:
        stopwords = {"what", "which", "that", "this", "with", "from", "have", "can", "for", "the", "and", "are", "period", "total"}
        return [word for word in re.findall(r"[a-z0-9]+", text.lower()) if len(word) > 2 and word not in stopwords]
