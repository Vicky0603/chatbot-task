from __future__ import annotations

from typing import List, Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
import openai


from src.config import settings
from src.vectorstore.loader import get_vectorstore

Document = Any


def _format_docs(docs: List[Document]) -> str:
    """
    Join the retrieved documents' content into a single string.
    """
    return "\n\n".join(doc.page_content for doc in docs)


# Load vector store and create retriever
_vectorstore = get_vectorstore()
# Use MMR to improve recall and set a higher k
retriever = _vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 8, "fetch_k": 24})

# Define prompt for RAG
prompt = ChatPromptTemplate.from_template(
    """
You are a helpful assistant that answers questions ONLY using the context provided.
If the answer is not in the context, say that you don't know and suggest rephrasing the question.

Context:
{context}

Question:
{question}

Answer in a clear and concise way.
"""
)

# LLM
llm = ChatOpenAI(
    model=settings.model_name,
    api_key=settings.openai_api_key,
    temperature=0,
)


# RAG chain: input = question (string)
rag_chain = (
    {"context": retriever | _format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
)
