
# Import libraries
import os
import streamlit as st
import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
import pinecone
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch

import os
import streamlit as st
import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import Pinecone as PineconeVectorStore
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import pinecone

!pip install --upgrade "pinecone-client>=5.2.0"

# ⚠️ Only for testing, don't share this publicly
PINECONE_API_KEY = "pcsk_6Sbvwk_AAz6zejmVbJ8wfXixysVjt1jDPiQ2cL3JAmHND4PjD7R9YGkDogpZPbHB4GNrRz"
PINECONE_ENV = "us-east1-gcp"

print("✅ Pinecone key loaded successfully")

# Using fast embeddings for large PDFs
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Local LLM using FLAN-T5-small (fast for short summaries)
llm_model_name = "google/flan-t5-small"
device = 0 if torch.cuda.is_available() else -1

tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(llm_model_name)

llm_pipeline = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    device=device,
    max_length=512
)

def extract_pdf_text(pdf_files):
    text = ""
    for pdf in pdf_files:
        with pdfplumber.open(pdf) as pdf_doc:
            for page in pdf_doc.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    return text

def split_text(text, chunk_size=1000, overlap=200):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.split_text(text)

def index_documents(chunks):
    # Remove empty chunks
    chunks = [c for c in chunks if c.strip()]

    # Convert to embeddings and upsert
    vectors = [(str(i), embeddings.embed_documents([chunk])[0], {"text": chunk}) for i, chunk in enumerate(chunks)]
    index.upsert(vectors)
    st.success(f"✅ {len(chunks)} chunks indexed into Pinecone")

def retrieve_context(query, top_k=5):
    # Retrieve top_k relevant chunks from Pinecone
    results = index.query(queries=[embeddings.embed_query(query)], top_k=top_k, include_metadata=True)
    docs = [match['metadata']['text'] for match in results['results'][0]['matches']]
    return docs

def summarize_chunks(chunks):
    summaries = []
    for i, chunk in enumerate(chunks):
        prompt = f"Summarize the following text in 1-2 sentences. Cite as [Source {i+1}].\n\n{chunk}\nSummary:"
        response = llm_pipeline(prompt)
        summaries.append(response[0]['generated_text'].strip())
    return " ".join(summaries)

def generate_answer(question):
    chunks = retrieve_context(question, top_k=5)
    if not chunks:
        return "Not enough information"

    context = summarize_chunks(chunks)
    prompt = f"""
You are a research assistant AI.
Answer the question concisely using only the provided context.
Cite sources like [Source X].
If information is missing, say 'Not enough information'.

Context:
{context}

Question:
{question}

Answer:
"""
    response = llm_pipeline(prompt)
    return response[0]['generated_text'].strip()

def main():
    st.set_page_config(page_title="ResearchBot", page_icon="📚", layout="wide")

    st.markdown(
        """
        <style>
        body {background-color:#0d1117; color:#c9d1d9;}
        .stButton>button {background-color:#238636; color:white; border-radius:10px;}
        .stTextInput>div>input {background-color:#161b22; color:white;}
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("📚 ResearchBot – Research Paper Assistant")

    question = st.text_input("Enter your research question:")
    if question:
        answer = generate_answer(question)
        st.subheader("🤖 Answer")
        st.write(answer)

    with st.sidebar:
        st.header("📁 Upload PDFs")
        pdfs = st.file_uploader("Upload your research papers", type="pdf", accept_multiple_files=True)
        if st.button("Index PDFs"):
            if not pdfs:
                st.warning("Please upload at least one PDF.")
            else:
                text = extract_pdf_text(pdfs)
                chunks = split_text(text)
                index_documents(chunks)

if __name__ == "__main__":
    main()
