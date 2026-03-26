from sentence_transformers import SentenceTransformer
import faiss
import pandas as pd

# Load model & FAISS index
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
index = faiss.read_index("data/embeddings/faiss.index")
essays = pd.read_csv("data/processed/ielts_clean.csv")

# Get query
query = input("Enter your essay topic or question: ")
query_emb = model.encode([query]).astype('float32')

# Retrieve top 5 essays
D, I = index.search(query_emb, k=5)
retrieved_essays = "\n\n".join([essays.iloc[idx]['essay'] for idx in I[0]])

# Prepare prompt for LLM
prompt = f"""
You are an IELTS expert. Analyze the following essays and provide:

1. Feedback (grammar, vocabulary, coherence, structure)
2. Suggestions to improve the essay
3. Band score with reasoning

Essays:
{retrieved_essays}
"""

# --- Example using OpenAI GPT ---
from openai import OpenAI
client = OpenAI(api_key="YOUR_API_KEY")

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
)

print("\n=== Feedback & Suggestions ===\n")
print(response.choices[0].message.content)