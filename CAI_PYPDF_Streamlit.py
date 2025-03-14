# import streamlit as st
# import os
# import re
# import numpy as np
# import PyPDF2
# import faiss
# from sentence_transformers import SentenceTransformer
# from rank_bm25 import BM25Okapi
# from sentence_transformers.cross_encoder import CrossEncoder
# from huggingface_hub import login

# st.set_page_config(page_title="Financial Statement Analyzer", layout="wide")

# st.title("📊 Financial Statement Analyzer")
# st.markdown("Upload your financial document and ask questions about it")

# # File upload section
# uploaded_file = st.file_uploader("Upload your financial statement (PDF or TXT)", type=["pdf", "txt"])

# def load_document(uploaded_file):
#     """Loads the document from the uploaded file and extracts text content."""
#     if uploaded_file is None:
#         return None

#     if uploaded_file.name.lower().endswith('.pdf'):
#         text = load_pdf(uploaded_file)
#     elif uploaded_file.name.lower().endswith('.txt'):
#         text = load_txt(uploaded_file)
#     else:
#         st.error("Unsupported file format. Please provide a PDF or TXT file.")
#         return None
#     return text

# def load_pdf(file):
#     """Loads text from a PDF file."""
#     try:
#         pdf_reader = PyPDF2.PdfReader(file)
#         text = ""
#         for page in pdf_reader.pages:
#             text += page.extract_text()
#         return text
#     except Exception as e:
#         st.error(f"Error loading PDF: {e}")
#         return None

# def load_txt(file):
#     """Loads text from a TXT file."""
#     try:
#         text = file.getvalue().decode("utf-8")
#         return text
#     except Exception as e:
#         st.error(f"Error loading TXT: {e}")
#         return None

# def chunk_text(text, chunk_size=500, overlap=100):
#     """Chunks text into smaller, overlapping pieces."""
#     chunks = []
#     start = 0
#     while start < len(text):
#         end = start + chunk_size
#         chunks.append(text[start:end])
#         start = end - overlap
#     return chunks

# def embed_chunks(chunks, embedding_model_name="all-MiniLM-L6-v2"):
#     """Embeds text chunks using a specified Sentence Transformer model."""
#     with st.spinner("Embedding document chunks..."):
#         model = SentenceTransformer(embedding_model_name)
#         embeddings = model.encode(chunks)
#     return embeddings

# def create_vector_db(embeddings, chunk_ids):
#     """Creates a FAISS vector database and adds embeddings."""
#     dimension = embeddings.shape[1]
#     index = faiss.IndexFlatL2(dimension)
#     index.add(np.array(embeddings).astype('float32'))
#     return index

# def retrieve_chunks_vector_db(query, vector_db, chunks, embedding_model_name="all-MiniLM-L6-v2", top_k=3):
#     """Retrieves top_k most relevant chunks from the vector database for a given query."""
#     query_embedding = embed_chunks([query], embedding_model_name)[0]
#     D, I = vector_db.search(np.array([query_embedding]).astype('float32'), top_k)
#     retrieved_chunks = [(chunks[i], D[0][j]) for j, i in enumerate(I[0])]
#     return retrieved_chunks

# def create_bm25_index(chunks):
#     """Creates a BM25 index from the text chunks."""
#     tokenized_chunks = [chunk.split() for chunk in chunks]
#     bm25_index = BM25Okapi(tokenized_chunks)
#     return bm25_index

# def retrieve_chunks_bm25(query, bm25_index, chunks, reranking_model_name="cross-encoder/ms-marco-MiniLM-L-12-v2", top_k_bm25=10, top_k_rerank=3):
#     """Retrieves and re-ranks top chunks using BM25 and a Cross-Encoder."""
#     tokenized_query = query.split()
#     bm25_scores = bm25_index.get_scores(tokenized_query)
#     top_chunk_indices_bm25 = np.argsort(bm25_scores)[::-1][:top_k_bm25]
#     retrieved_chunks_bm25 = [chunks[i] for i in top_chunk_indices_bm25]

#     # Re-ranking with Cross-Encoder
#     rerank_inputs = [[query, chunk] for chunk in retrieved_chunks_bm25]
#     cross_encoder = CrossEncoder(reranking_model_name)
#     rerank_scores = cross_encoder.predict(rerank_inputs)

#     chunk_score_pairs = list(zip(retrieved_chunks_bm25, rerank_scores))
#     sorted_chunk_score_pairs = sorted(chunk_score_pairs, key=lambda x: x[1], reverse=True)

#     reranked_chunks = [(chunk, score) for chunk, score in sorted_chunk_score_pairs[:top_k_rerank]]
#     return reranked_chunks


# def validate_input(query: str) -> bool:
#     """Ensure input is financial-related."""
#     # Expanded list of restricted terms (non-finance topics and inappropriate content)
#     restricted_terms = [
#         "weather", "movies", "sports", "travel", "health", "technology",
#         "entertainment", "politics", "history", "geography", "science", "food",
#         "France", "music", "games", "celebrity", "art", "fashion", "fitness",
#         "sex", "violence", "drugs", "gambling", "adult", "explicit", "crime",
#         "religion", "spirituality", "astrology", "conspiracy", "myth", "occult"
#     ]

#     # Finance-related keywords derived from the financial statements document
#     finance_keywords = [
#         "net sales", "cost of sales", "gross margin", "operating income", "revenue",
#         "income", "assets", "liabilities", "equity", "cash flow", "stock", "bond",
#         "investment", "market", "banking", "loan", "interest", "credit", "capital",
#         "tax", "budget", "dividend", "share", "earnings", "debt", "amortization",
#         "accounts payable", "accounts receivable", "retained earnings", "expenses",
#         "depreciation", "balance sheet", "income statement", "term debt", "liquidity",
#         "commercial paper", "operating activities", "financing activities", "investing activities"
#     ]

#     query_lower = query.lower()

#     # Check if query contains any restricted terms
#     if any(re.search(rf"\b{term}\b", query_lower) for term in restricted_terms):
#         return False

#     # Ensure at least one finance-related term is present
#     return any(re.search(rf"\b{term}\b", query_lower) for term in finance_keywords)

# # Sidebar for Hugging Face token
# with st.sidebar:
#     st.header("Settings")
#     hf_token = st.text_input("Hugging Face Token (optional)", type="password")
#     st.info("Enter your Hugging Face token if you want to use gated models")

#     if st.button("Login to Hugging Face"):
#         if hf_token:
#             try:
#                 login(hf_token)
#                 st.success("Successfully logged in to Hugging Face!")
#             except Exception as e:
#                 st.error(f"Login failed: {e}")
#         else:
#             st.warning("Please enter a token first.")

# # Main app flow
# if uploaded_file is not None:
#     with st.spinner("Processing document..."):
#         document_text = load_document(uploaded_file)

#         if document_text:
#             st.success(f"Document loaded: {uploaded_file.name}")

#             # Show document preview
#             with st.expander("Document Preview"):
#                 st.text(document_text[:1000] + "..." if len(document_text) > 1000 else document_text)

#             # Chunk and embed document
#             chunks = chunk_text(document_text)
#             st.session_state.chunks = chunks

#             embeddings = embed_chunks(chunks)
#             vector_db_index = create_vector_db(embeddings, list(range(len(chunks))))
#             st.session_state.vector_db_index = vector_db_index

#             bm25_index_obj = create_bm25_index(chunks)
#             st.session_state.bm25_index_obj = bm25_index_obj

#             st.success(f"Document processed into {len(chunks)} chunks")
#         else:
#             st.error("Failed to load document content.")

#     # Query section
#     st.header("Ask questions about your financial document")

#     retrieval_method = st.radio(
#         "Choose retrieval method:",
#         ("1.Basic RAG Search based", "2.Advanced RAG Search BM25 based")
#     )

#     query = st.text_input("Enter your financial question:")

#     if st.button("Submit Query"):
#         if not query:
#             st.warning("Please enter a question.")
#         elif not validate_input(query):
#             st.error("❌ Invalid question. Please ask a financial-related question.")
#         else:
#             with st.spinner("Searching for answers..."):
#                 if retrieval_method == "1.Basic RAG Search based":
#                     retrieved_chunks = retrieve_chunks_vector_db(
#                         query,
#                         st.session_state.vector_db_index,
#                         st.session_state.chunks
#                     )
#                     method_name = "Basic Search"
#                 else:  # BM25 Search
#                     retrieved_chunks = retrieve_chunks_bm25(
#                         query,
#                         st.session_state.bm25_index_obj,
#                         st.session_state.chunks
#                     )
#                     method_name = "Advanced BM25 Search"

#                 # Display results
#                 st.subheader(f"Results from {method_name}")

#                 for i, (chunk, confidence) in enumerate(retrieved_chunks):
#                     confidence_percentage = round(confidence * 100, 2)
#                     expander_state = i == 0  # Expand first result, collapse others

#                     with st.expander(f"Result {i+1} (Confidence: {confidence_percentage}%)", expanded=expander_state):
#                         st.markdown(chunk)

# else:
#     st.info("Please upload a financial document to get started.")
    
# # Footer
# st.markdown("---")
# st.caption("Financial Statement Analyzer using RAG (Retrieval-Augmented Generation)")
import os
import json
import numpy as np
import streamlit as st
import fitz  # PyMuPDF for PDF text extraction
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity
import faiss
from typing import List, Tuple

# Load models
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # Open-source embedding model
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")  # Re-ranking model

# Function to extract text from PDFs
def extract_text_from_pdf(pdf_path):
    """Extract text from a given PDF file."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"
    return text

# Load financial documents from PDFs
pdf_files = ["Document.pdf"]
documents = []

for pdf_file in pdf_files:
    if os.path.exists(pdf_file):
        pdf_text = extract_text_from_pdf(pdf_file)
        documents.append(pdf_text)  # Store extracted text
    else:
        st.warning(f"File {pdf_file} not found.")

# Convert documents to tokens for BM25
tokenized_corpus = [doc.split() for doc in documents]
bm25 = BM25Okapi(tokenized_corpus)

# Create embeddings
embeddings = np.array([embedding_model.encode(doc) for doc in documents])

# FAISS Vector Store
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# Function to retrieve relevant docs
def retrieve_documents(query: str, top_k: int = 3) -> List[Tuple[str, float]]:
    """Retrieve documents using BM25 and embeddings."""
    bm25_scores = bm25.get_scores(query.split())
    top_bm25_idx = np.argsort(bm25_scores)[::-1][:top_k]
    
    query_embedding = embedding_model.encode(query).reshape(1, -1)
    _, top_embedding_idx = index.search(query_embedding, top_k)
    
    # Combine BM25 and embeddings results
    retrieved_docs = list(set(top_bm25_idx) | set(top_embedding_idx[0]))
    rerank_inputs = [[query, documents[idx]] for idx in retrieved_docs]
    
    # Re-rank using Cross-Encoder
    rerank_scores = cross_encoder.predict(rerank_inputs)
    sorted_indices = np.argsort(rerank_scores)[::-1]
    
    return [(documents[retrieved_docs[i]], rerank_scores[i]) for i in sorted_indices]

# Guardrail (Input Validation)
def validate_input(query: str) -> bool:
    """Ensure input is financial-related."""
    restricted_terms = ["France", "weather", "movies", "sports"]
    return not any(term in query.lower() for term in restricted_terms)

# Streamlit UI
st.title("RAG Financial Chatbot")
st.write("Ask me financial questions based on company earnings statements!")

query = st.text_input("Enter your question:")
if query:
    if validate_input(query):
        results = retrieve_documents(query)
        st.subheader("Top Answers:")
        for doc, score in results:
            st.write(f"**Answer:** {doc[:500]}...")  # Display first 500 chars
            st.write(f"Confidence Score: {score:.2f}")
    else:
        st.write("❌ Invalid question. Please ask a financial-related question.")
