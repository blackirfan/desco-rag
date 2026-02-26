import os
import faiss
import pickle
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# ==============================
# CONFIG
# ==============================

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
SIMILARITY_THRESHOLD = 0.5
TOP_K = 1

# Bangla generic stopwords (expand if needed)
STOPWORDS = {
    "কি", "কিভাবে", "কেন", "কত", "ডেসকো", "এর", "ও", "হবে", "চাই",
    "কোন", "কোনো", "গুলো", "সম্পর্কে"
}

app = FastAPI(title="DESCO FAQ + Bangla LLM API")

# ==============================
# LOAD EMBEDDING MODEL
# ==============================

embed_model = SentenceTransformer(MODEL_NAME)

# ==============================
# LOAD VECTORSTORE
# ==============================

index = faiss.read_index("vectorstore/faiss_index")

with open("vectorstore/chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

print(f"Loaded {index.ntotal} FAQ vectors")

# ==============================
# LOAD LLM CLIENT
# ==============================

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ==============================
# REQUEST MODEL
# ==============================

class QueryRequest(BaseModel):
    question: str


# ==============================
# RETRIEVE FAQ
# ==============================

def retrieve_faq(question: str):
    q_emb = embed_model.encode([question]).astype("float32")
    faiss.normalize_L2(q_emb)

    distances, indices = index.search(q_emb, 2)  # top 2

    top1_score = float(distances[0][0])
    top2_score = float(distances[0][1]) if len(distances[0]) > 1 else 0.0

    top1_index = int(indices[0][0])

    if top1_index == -1:
        return None, 0.0, 0.0

    best_match = chunks[top1_index]

    return best_match, top1_score, top2_score


# ==============================
# STRONG KEYWORD GUARD
# ==============================

def keyword_guard(user_question: str, faq_question: str):

    user_words = {
        w.strip("?,।.!")
        for w in user_question.split()
        if w not in STOPWORDS
    }

    faq_words = {
        w.strip("?,।.!")
        for w in faq_question.split()
        if w not in STOPWORDS
    }

    common_words = user_words.intersection(faq_words)

    # Require minimum TWO meaningful overlaps
    return len(common_words) >= 2


# ==============================
# LLM FALLBACK (Bangla Only)
# ==============================

def generate_llm_response(question: str):
    prompt = f"""
আপনি একজন পেশাদার ও সহায়ক DESCO বিদ্যুৎ সেবা সহকারী।

নিয়ম:
- শুধুমাত্র বাংলা ভাষায় উত্তর দিন।
- ইংরেজি ব্যবহার করবেন না।
- কোনো সতর্কতা বা অপ্রয়োজনীয় ভূমিকা যোগ করবেন না।
- সংক্ষিপ্ত ও প্রাসঙ্গিক উত্তর দিন।

প্রশ্ন: {question}

উত্তর:
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=400,
    )

    return response.choices[0].message.content.strip()


# ==============================
# CHAT ENDPOINT
# ==============================

@app.post("/chat")
async def chat(request: QueryRequest):
    question = request.question.strip()

    faq_match, score1, score2 = retrieve_faq(question)

    score_gap = score1 - score2

    # FAQ match condition
    if (
        faq_match
        and score1 >= SIMILARITY_THRESHOLD
        and score_gap >= 0.15
    ):
        return {
            "answer": faq_match["answer"],
            "intent": faq_match["intent"],
            "category": faq_match["category"],
            "confidence": round(score1, 3),
            "mode": "FAQ-Direct"
        }

    # LLM fallback
    try:
        answer = generate_llm_response(question)

        return {
            "answer": answer,
            "confidence": round(score1, 3),
            "mode": "LLM-Fallback"
        }

    except Exception:
        return {
            "answer": "দুঃখিত, বর্তমানে উত্তর প্রদান করা সম্ভব হচ্ছে না।",
            "confidence": round(score1, 3),
            "mode": "Error"
        }

# ==============================
# HEALTH CHECK
# ==============================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "total_vectors": index.ntotal,
        "embedding_model": MODEL_NAME
    }