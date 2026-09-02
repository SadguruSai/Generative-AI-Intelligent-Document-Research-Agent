from pathlib import Path

from langchain_community.document_loaders import CSVLoader, PyPDFLoader, TextLoader
from langchain_community.document_loaders.excel import UnstructuredExcelLoader
from langchain_core.documents import Document


LOADERS = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".csv": CSVLoader,
    ".xlsx": UnstructuredExcelLoader,
}


def load_all_documents(data_dir: str) -> list[Document]:
    """Load supported files from a directory into LangChain Documents."""
    documents: list[Document] = []
    for path in Path(data_dir).rglob("*"):
        loader_cls = LOADERS.get(path.suffix.lower())
        if not loader_cls or "vector_store" in path.parts:
            continue
        try:
            documents.extend(loader_cls(str(path)).load())
        except Exception as exc:
            documents.append(Document(page_content=f"Could not load {path.name}: {exc}", metadata={"source": str(path)}))
    return documents
