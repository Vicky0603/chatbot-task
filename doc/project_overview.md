# Promtior RAG Chatbot – Project Overview

## 1. Introduction
This project implements a Retrieval-Augmented Generation (**RAG**) assistant designed to answer questions about **Promtior** using information extracted from its public website and optional PDF material. The solution uses **LangChain**, **Chroma**, **OpenAI embeddings**, and **LangServe**, and includes a minimal frontend to interact with the chatbot.

---

## 2. Architecture Summary
The system is composed of four main parts:

### **2.1 Data Ingestion**
- Website content is fetched using `WebBaseLoader` for selected Promtior URLs.
- Optional PDFs are ingested using `PyPDFLoader`.
- All documents are unified into a single corpus.

### **2.2 Embeddings & Vector Store**
- Text is chunked using `RecursiveCharacterTextSplitter`.
- Embeddings generated with OpenAI’s `text-embedding-3-small`.
- Data is indexed and persisted using **Chroma** (SQLite backend).

### **2.3 RAG Chain**
- A retriever (`k=4`) fetches relevant context.
- A structured prompt ensures grounded answers.
- `ChatOpenAI` generates the final response.
- The chain is exposed as a LangServe endpoint:
  - `POST /promtior-rag/invoke`

### **2.4 Frontend**
- Simple HTML/CSS/JS chat interface.
- Sends user messages to the API and renders responses.
- Served statically via FastAPI.

---

## 3. Folder Structure
promtior-rag-chatbot/
├── src/
│ ├── main.py
│ ├── config.py
│ ├── chains/rag_chain.py
│ ├── ingestion/
│ │ ├── load_promtior_site.py
│ │ └── build_vector_store.py
│ └── vectorstore/loader.py
├── frontend/
│ ├── index.html
│ ├── styles.css
│ └── app.js
└── data/vectorstore/

---

## 4. Environment Variables
OPENAI_API_KEY=your_key
MODEL_NAME=gpt-4o-mini(or the model you want)
VECTORSTORE_DIR=./data/vectorstore
FRONTEND_DIR=./frontend

---

## 5. Running the Project
pip install -r requirements.txt
python -m src.ingestion.build_vector_store
uvicorn src.main:app --reload
Open the chatbot at:
http://localhost:8000/

--- 

## 6. Conclusion

This project delivers a full end-to-end RAG assistant with:

- Website + PDF ingestion

- Persistent vector store

- Retrieval pipeline with LangChain

- API served with LangServe

- Lightweight browser-based chatbot

It fulfills the technical requirements while keeping the architecture modular and deployment-ready. 