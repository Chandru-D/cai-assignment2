import streamlit as st
import os
import re
import numpy as np
import PyPDF2
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from sentence_transformers.cross_encoder import CrossEncoder
import spacy
import subprocess

st.set_page_config(page_title="Financial Statement Analyzer", layout="wide")

st.title("📊 Financial Statement of Apple inc. for financial year of 2024 & 2025")
st.markdown("Ask questions about the financial statements of Apple Inc.")

# Load NLP model for entity recognition
subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
nlp = spacy.load("en_core_web_sm")

def load_pdf(file_path):
    """Loads text from a PDF file."""
    try:
        with open(file_path, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = "\n".join(page.extract_text() for page in pdf_reader.pages if page.extract_text())
        return text
    except Exception as e:
        st.error(f"Error loading PDF: {e}")
        return None

def extract_financial_data(text):
    """Extracts key financial data from structured text."""
    financial_data = {}
    pattern = re.compile(r"(Total current assets)\s+(\$?\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?)")
    matches = pattern.findall(text)
    for match in matches:
        financial_data[match[0]] = match[1]
    return financial_data

def chunk_text(text, chunk_size=500, overlap=100):
    """Chunks text into smaller, overlapping pieces while preserving financial sections."""
    sections = text.split("\n\n")
    chunks = []
    buffer = ""
    for section in sections:
        if len(buffer) + len(section) < chunk_size:
            buffer += section + "\n"
        else:
            chunks.append(buffer)
            buffer = section + "\n"
    if buffer:
        chunks.append(buffer)
    return chunks

def embed_chunks(chunks, embedding_model_name="all-MiniLM-L6-v2"):
    """Embeds text chunks using a specified Sentence Transformer model."""
    model = SentenceTransformer(embedding_model_name)
    embeddings = model.encode(chunks)
    return embeddings

def create_vector_db(embeddings):
    """Creates a FAISS vector database."""
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))
    return index

def create_bm25_index(chunks):
    """Creates a BM25 index from the text chunks."""
    tokenized_chunks = [chunk.split() for chunk in chunks]
    return BM25Okapi(tokenized_chunks)

def retrieve_chunks_vector_db(query, vector_db, chunks, embedding_model_name="all-MiniLM-L6-v2", top_k=3):
    """Retrieves top_k most relevant chunks using FAISS vector search."""
    query_embedding = embed_chunks([query], embedding_model_name)[0]
    D, I = vector_db.search(np.array([query_embedding]).astype('float32'), top_k)
    return [(chunks[i], D[0][j]) for j, i in enumerate(I[0])]

def retrieve_chunks_bm25(query, bm25_index, chunks, top_k=3):
    """Retrieves top_k most relevant chunks using BM25."""
    tokenized_query = query.split()
    scores = bm25_index.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(chunks[i], scores[i]) for i in top_indices]

# Load and process documents on page load
document_paths = ["FY24_Q1_Consolidated_Financial_Statements.pdf", "FY25_Q1_Consolidated_Financial_Statements.pdf"]
document_texts = []
financial_data_map = {}

for path in document_paths:
    if os.path.exists(path):
        text = load_pdf(path)
        if text:
            document_texts.append(text)
            financial_data_map.update(extract_financial_data(text))
    else:
        st.error(f"File not found: {path}")

if document_texts:
    combined_text = "\n".join(document_texts)
    chunks = chunk_text(combined_text)
    embeddings = embed_chunks(chunks)
    vector_db_index = create_vector_db(embeddings)
    bm25_index_obj = create_bm25_index(chunks)

    st.session_state.chunks = chunks
    st.session_state.vector_db_index = vector_db_index
    st.session_state.bm25_index_obj = bm25_index_obj
    st.session_state.financial_data_map = financial_data_map
    st.success(f"Loaded and processed {len(chunks)} chunks from {len(document_texts)} documents.")
else:
    st.error("No documents loaded.")

# Query section
st.header("Ask questions about the financial documents")
retrieval_method = st.radio("Choose retrieval method:", ("1.Basic RAG Search based", "2.Advanced RAG Search BM25 based"))
query = st.text_input("Enter your financial question:")

if st.button("Submit Query"):
    if not query:
        st.warning("Please enter a question.")
    else:
        with st.spinner("Searching for answers..."):
            if query.lower() in st.session_state.financial_data_map:
                answer = st.session_state.financial_data_map[query.lower()]
                st.subheader("Exact Financial Data Found:")
                st.markdown(f"**{query}:** {answer}")
            else:
                if retrieval_method == "1.Basic RAG Search based":
                    retrieved_chunks = retrieve_chunks_vector_db(query, st.session_state.vector_db_index, st.session_state.chunks)
                    method_name = "Basic Search"
                else:
                    retrieved_chunks = retrieve_chunks_bm25(query, st.session_state.bm25_index_obj, st.session_state.chunks)
                    method_name = "Advanced BM25 Search"

                st.subheader(f"Results from {method_name}")
                for i, (chunk, confidence) in enumerate(retrieved_chunks):
                    confidence_percentage = round(confidence * 100, 2)
                    with st.expander(f"Result {i+1} (Confidence: {confidence_percentage}%)"):
                        st.markdown(chunk)
