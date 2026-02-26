import os
import faiss
import pickle
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# Paths
PDF_FOLDER = "pdf"  # Your PDFs folder
VECTOR_FOLDER = "vectorstore"
os.makedirs(VECTOR_FOLDER, exist_ok=True)

# Use a multilingual embedding model for better Bengali support
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    documents = []
    for page_number, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            text = text.strip()
            if text:
                documents.append({
                    "content": text,
                    "pdf_name": os.path.basename(pdf_path),
                    "page_number": page_number + 1
                })
    return documents


def chunk_text(text, chunk_size=500, overlap=100):
    """Split text into overlapping chunks"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        start += chunk_size - overlap
    return chunks


all_chunks = []

# Process each PDF
for file in os.listdir(PDF_FOLDER):
    if file.lower().endswith(".pdf"):
        pdf_path = os.path.join(PDF_FOLDER, file)
        documents = extract_text_from_pdf(pdf_path)
        print(f"Processing {file}, {len(documents)} pages found.")
        for doc in documents:
            text_chunks = chunk_text(doc["content"])
            for chunk in text_chunks:
                all_chunks.append({
                    "content": chunk,
                    "pdf_name": doc["pdf_name"],
                    "page_number": doc["page_number"]
                })

print(f"Total chunks: {len(all_chunks)}")
print("Generating embeddings...")

# Generate embeddings
texts = [chunk["content"] for chunk in all_chunks]
embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

# Ensure embeddings are float32 for FAISS
embeddings = np.array(embeddings, dtype=np.float32)

# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)
faiss.write_index(index, os.path.join(VECTOR_FOLDER, "faiss_index"))

# Save chunks
with open(os.path.join(VECTOR_FOLDER, "chunks.pkl"), "wb") as f:
    pickle.dump(all_chunks, f)

print("All PDFs ingested and embeddings stored successfully!")