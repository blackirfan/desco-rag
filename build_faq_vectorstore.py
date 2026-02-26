import json
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import os

JSON_PATH = "desco_faq_full.json"
VECTOR_DIR = "vectorstore"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

with open(JSON_PATH, "r", encoding="utf-8") as f:
    faqs = json.load(f)

embed_model = SentenceTransformer(MODEL_NAME)

documents = []
texts = []

print("Preparing question vectors...")

for item in faqs:
    for question in item["question_variants"]:
        texts.append(question)

        documents.append({
            "id": item["id"],
            "intent": item["intent"],
            "category": item["category"],
            "question": question,
            "answer": item["answer"]
        })

print("Generating embeddings...")

embeddings = embed_model.encode(
    texts,
    batch_size=32,
    show_progress_bar=True,
    convert_to_numpy=True
).astype("float32")

faiss.normalize_L2(embeddings)

index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

os.makedirs(VECTOR_DIR, exist_ok=True)

faiss.write_index(index, f"{VECTOR_DIR}/faiss_index")

with open(f"{VECTOR_DIR}/chunks.pkl", "wb") as f:
    pickle.dump(documents, f)

print(f"Built {len(documents)} question vectors.")