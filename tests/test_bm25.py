from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document


def test_bm25_retrieves_founded_fact():
    docs = [
        Document(page_content="Promtior was founded in May 2023.", metadata={"source": "facts"}),
        Document(page_content="Promtior focuses on RAG implementations and generative AI services.", metadata={"source": "facts"}),
    ]
    retriever = BM25Retriever.from_documents(docs)
    hits = retriever.get_relevant_documents("When was the company founded?")
    assert any("May 2023" in d.page_content for d in hits)

