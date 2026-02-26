⚡ DESCO Hybrid RAG Support System

A cost-efficient AI-powered support assistant using:

🔎 FAQ Semantic Retrieval (FAISS + Embeddings)

🧠 Intelligent Confidence + Margin Filtering

🖥️ Optional Local LLM

☁️ Cloud LLM Fallback (Bangla Only)

Designed for scalable public-facing utility or government support systems.

🏗️ Architecture Overview
User Query
   ↓
Semantic FAQ Retrieval (FAISS)
   ↓
High Confidence? → Direct FAQ Response
   ↓
Low Confidence? → LLM Fallback (Bangla Only)
Key Design Features

Hybrid RAG Architecture

Multilingual Embeddings

Score Threshold + Margin Validation

Bangla-only Response Enforcement

Cost-Aware LLM Routing

False Positive Protection

🚀 Getting Started
1️⃣ Clone the Repository
git clone <your-repo-url>
cd desco-rag
2️⃣ Create a Fresh Virtual Environment

Inside the project folder:

py -m venv venv

If py does not work:

python -m venv venv
3️⃣ Activate the Virtual Environment
venv\Scripts\activate

You should now see something like:

(venv) D:\work\contact-center\DESCO\desco-rag>

Verify Python version:

python --version
4️⃣ Install Dependencies
pip install -r requirements.txt
5️⃣ Build the Vector Store

Before running the API, generate embeddings:

python build_faq_vectorstore.py

This creates:

vectorstore/
 ├── faiss_index
 └── chunks.pkl
6️⃣ Set Environment Variables

Create a .env file in the project root:

GROQ_API_KEY=your_api_key_here

⚠️ Never commit .env to GitHub.

7️⃣ Run the FastAPI Server
uvicorn app:app --reload

The API will run at:

http://127.0.0.1:8000

Swagger Docs available at:

http://127.0.0.1:8000/docs
📌 API Example
POST /chat

Request:

{
  "question": "ডেসকো কি?"
}

Response:

{
  "answer": "...",
  "confidence": 0.87,
  "mode": "FAQ-Direct"
}

Modes:

FAQ-Direct

LLM-Fallback

Error
