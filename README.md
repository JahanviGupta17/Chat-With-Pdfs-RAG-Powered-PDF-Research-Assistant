# 📚 ResearchBot – AI-Powered Research Assistant

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.26.0-orange)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Issues](https://img.shields.io/github/issues/JahanviGupta17/Chat-With-Pdfs-RAG-Powered-PDF-Research-Assistant)](https://github.com/JahanviGupta17/Chat-With-Pdfs-RAG-Powered-PDF-Research-Assistant/issues)

**ResearchBot** is a **Retrieval-Augmented Generation (RAG) platform** built for researchers, students, and professionals who want to quickly digest and interact with large collections of PDFs and research papers. Using **state-of-the-art Hugging Face LLMs**, **Pinecone vector database**, and **smart text embeddings**, ResearchBot extracts, indexes, and summarizes key insights—delivering **fast, context-aware answers with source citations**.  

---
## 🌟 Features

- **📄 Multi-PDF Ingestion:** Upload multiple PDFs at once, automatically extracted.  
- **🧩 Smart Text Chunking:** Efficiently splits documents for accurate retrieval.  
- **🔍 RAG Pipeline:** Embeddings + vector database = highly relevant answers.  
- **📝 Summarization & Sources:** Summarizes chunks with source citations.  
- **💻 Interactive UI:** Notebook-style **dark theme** interface with smooth animations.  
- **⚡ Hugging Face LLMs:** Fast, high-quality answers using modern transformers.  
- **📈 Scalable:** Handles **large PDFs** using Pinecone for vector search.  
## 🏗 Installation & Setup
### Clone the repository
- git clone https://github.com/JahanviGupta17/Chat-With-Pdfs-RAG-Powered-PDF-Research-Assistant

- cd Chat-With-Pdfs-RAG-Powered-PDF-Research-Assistant

### Install dependencies
pip install -r requirements.txt

### Set up environment variables / secrets
- Hugging Face API token
export HF_TOKEN="your_huggingface_api_token"
### Pinecone API token and environment
- export PINECONE_TOKEN="your_pinecone_api_token"
- export PINECONE_ENV="your_pinecone_env_region"

### Run Streamlit
- streamlit run app.py

---

## 🏗 Architecture Overview

```text
+-----------------+       +-------------------+       +------------------+
|  PDF Documents  | --->  |  Text Extraction  | --->  |   Chunking       |
|  (Multiple)     |       |   (pdfplumber)    |       | (RecursiveChar)  |
+-----------------+       +-------------------+       +------------------+
         |                        |                           |
         v                        v                           v
  +-----------------+      +-------------------+       +------------------+
  | Embedding Model | ---> | Vector Database    | --->  |  Similarity       |
  | (SentenceTrans) |      |   (Pinecone)       |       |  Search (Top-k)  |
  +-----------------+      +-------------------+       +------------------+
                                           |
                                           v
                                    +------------------+
                                    |   LLM Querying   |
                                    | (HuggingFace)    |
                                    +------------------+
                                           |
                                           v
                                    +------------------+
                                    |  Streamlit UI    |
                                    | Answer + Sources |
                                    +------------------+

