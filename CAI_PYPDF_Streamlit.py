import streamlit as st
import os
import numpy as np
import PyPDF2
import faiss
import re
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from sentence_transformers.cross_encoder import CrossEncoder

st.set_page_config(page_title="Financial Statement Analyzer", layout="wide")

st.title("📊 Financial Statement Analyzer for 2023 & 2024")
st.markdown("Automatically loaded financial statements for retrieval-based analysis.")

# File paths for the preloaded documents
PRELOADED_FILES = ["2023_financial_statement.pdf", "2024_financial_statement.pdf"]

def load_pdf(file_path):
    """Loads text from a PDF file."""
    try:
        with open(file_path, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = "".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
        return text
    except Exception as e:
        st.error(f"Error loading PDF: {e}")
        return None

def chunk_text(text, chunk_size=500, overlap=100):
    """Chunks text into smaller, overlapping pieces."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

def embed_chunks(chunks, embedding_model_name="all-MiniLM-L6-v2"):
    """Embeds text chunks using a specified Sentence Transformer model."""
    with st.spinner("Embedding document chunks..."):
        model = SentenceTransformer(embedding_model_name)
        embeddings = model.encode(chunks)
    return embeddings

def create_vector_db(embeddings):
    """Creates a FAISS vector database and adds embeddings."""
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))
    return index

def retrieve_chunks_vector_db(query, vector_db, chunks, embedding_model_name="all-MiniLM-L6-v2", top_k=3):
    """Retrieves top_k most relevant chunks from the vector database for a given query."""
    query_embedding = embed_chunks([query], embedding_model_name)[0]
    D, I = vector_db.search(np.array([query_embedding]).astype('float32'), top_k)
    retrieved_chunks = [(chunks[i], D[0][j]) for j, i in enumerate(I[0])]
    return retrieved_chunks

def create_bm25_index(chunks):
    """Creates a BM25 index from the text chunks."""
    tokenized_chunks = [chunk.split() for chunk in chunks]
    bm25_index = BM25Okapi(tokenized_chunks)
    return bm25_index

def retrieve_chunks_bm25(query, bm25_index, chunks, reranking_model_name="cross-encoder/ms-marco-MiniLM-L-12-v2", top_k_bm25=10, top_k_rerank=3):
    """Retrieves and re-ranks top chunks using BM25 and a Cross-Encoder."""
    tokenized_query = query.split()
    bm25_scores = bm25_index.get_scores(tokenized_query)
    top_chunk_indices_bm25 = np.argsort(bm25_scores)[::-1][:top_k_bm25]
    retrieved_chunks_bm25 = [chunks[i] for i in top_chunk_indices_bm25]

    # Re-ranking with Cross-Encoder
    rerank_inputs = [[query, chunk] for chunk in retrieved_chunks_bm25]
    cross_encoder = CrossEncoder(reranking_model_name)
    rerank_scores = cross_encoder.predict(rerank_inputs)

    chunk_score_pairs = list(zip(retrieved_chunks_bm25, rerank_scores))
    sorted_chunk_score_pairs = sorted(chunk_score_pairs, key=lambda x: x[1], reverse=True)

    reranked_chunks = [(chunk, score) for chunk, score in sorted_chunk_score_pairs[:top_k_rerank]]
    return reranked_chunks

def validate_input(query: str) -> bool:
    """Ensure input is financial-related."""
    # Expanded list of restricted terms (non-finance topics and inappropriate content)
    restricted_terms = [
        "weather", "movies", "sports", "travel", "health", "technology",
        "entertainment", "politics", "history", "geography", "science", "food",
        "France", "music", "games", "celebrity", "art", "fashion", "fitness",
        "sex", "violence", "drugs", "gambling", "adult", "explicit", "crime",
        "religion", "spirituality", "astrology", "conspiracy", "myth", "occult"
    ]

    query_lower = query.lower()

    # Check if query contains any restricted terms
    if any(re.search(rf"\b{term}\b", query_lower) for term in restricted_terms):
        return False
    else:
        return True

# Load and process predefined documents
st.subheader("Processing Preloaded Financial Documents...")

all_texts = []
for file in PRELOADED_FILES:
    if os.path.exists(file):
        st.success(f"Loading {file}...")
        document_text = load_pdf(file)
        if document_text:
            all_texts.append(document_text)
    else:
        st.error(f"File {file} not found. Ensure it's in the working directory.")

if all_texts:
    combined_text = " ".join(all_texts)

    # Process and store in session state
    chunks = chunk_text(combined_text)
    st.session_state.chunks = chunks

    embeddings = embed_chunks(chunks)
    vector_db_index = create_vector_db(embeddings)
    st.session_state.vector_db_index = vector_db_index

    bm25_index_obj = create_bm25_index(chunks)
    st.session_state.bm25_index_obj = bm25_index_obj

    st.success(f"Processed {len(chunks)} chunks from financial documents.")

# Query section
st.header("Ask questions about the financial documents")

retrieval_method = st.radio(
    "Choose retrieval method:",
    ("1.Basic RAG Search based", "2.Advanced RAG Search BM25 based")
)

query = st.text_input("Enter your financial question:")

if st.button("Submit Query"):
    if not query:
        st.warning("Please enter a question.")
    elif not validate_input(query):
        st.error("❌ Invalid question. Please ask a financial-related question.")
    else:
        with st.spinner("Searching for answers..."):
            if retrieval_method == "1.Basic RAG Search based":
                retrieved_chunks = retrieve_chunks_vector_db(
                    query,
                    st.session_state.vector_db_index,
                    st.session_state.chunks
                )
                method_name = "Basic Search"
            else:  # BM25 Search
                retrieved_chunks = retrieve_chunks_bm25(
                    query,
                    st.session_state.bm25_index_obj,
                    st.session_state.chunks
                )
                method_name = "Advanced BM25 Search"

            # Display results
            st.subheader(f"Results from {method_name}")

            for i, (chunk, confidence) in enumerate(retrieved_chunks):
                confidence_percentage = round(confidence * 100, 2)
                expander_state = i == 0  # Expand first result, collapse others

                with st.expander(f"Result {i+1} (Confidence: {confidence_percentage}%)", expanded=expander_state):
                    st.markdown(chunk)
