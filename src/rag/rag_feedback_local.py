# src/rag/rag_feedback.py
"""
RAG Feedback Module
-------------------
Retrieves similar essays from FAISS index and generates
IELTS feedback using Qwen2.5-1.5B 4-bit GPTQ locally.
"""

import pickle
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
from transformers import AutoTokenizer, pipeline
from auto_gptq import AutoGPTQForCausalLM

# ---------------------------
# 1️⃣ Load FAISS index & metadata
# ---------------------------

faiss_index_path = "../../data/embeddings/faiss.index"
metadata_path = "../../data/embeddings/metadata.pkl"
processed_csv_path = "../../data/processed/ielts_clean.csv"

# Load FAISS index
index = faiss.read_index(faiss_index_path)

# Load metadata (list of indices matching rows in CSV)
with open(metadata_path, "rb") as f:
    metadata = pickle.load(f)

# Load processed essays
essays_df = pd.read_csv(processed_csv_path)

# Load sentence transformer embeddings (must match build_index.py)
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ---------------------------
# 2️⃣ Load Qwen2.5-1.5B 4-bit
# ---------------------------

qwen_model_path = "Mohaaxa/qwen2.5-1.5b-gptq-4bit-v2"

print("Loading Qwen2.5-1.5B 4-bit model... (may take a minute)")
tokenizer = AutoTokenizer.from_pretrained(qwen_model_path)

model = AutoGPTQForCausalLM.from_pretrained(
    qwen_model_path,
    device_map="auto",       # GPU if available, else CPU
    use_safetensors=True,
    gptq_base_name=None
)

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    return_full_text=False
)

# ---------------------------
# 3️⃣ Helper: Retrieve top k essays
# ---------------------------

def retrieve_essays(query: str, top_k: int = 5):
    query_emb = embed_model.encode([query])
    D, I = index.search(query_emb, top_k)
    retrieved = []
    for idx in I[0]:
        row_idx = metadata[idx]
        essay_text = essays_df.iloc[row_idx]["essay"]
        band = essays_df.iloc[row_idx].get("band", "N/A")
        retrieved.append(f"--- Essay {row_idx+1} | Band: {band} ---\n{essay_text}")
    return retrieved

# ---------------------------
# 4️⃣ Main interactive loop
# ---------------------------

if __name__ == "__main__":
    print("📝 IELTS RAG Feedback System")
    while True:
        query = input("\nEnter your essay topic or question (or 'exit' to quit):\n> ")
        if query.lower() in ("exit", "quit"):
            break

        # Retrieve top essays
        top_essays = retrieve_essays(query, top_k=5)
        retrieved_text = "\n\n".join(top_essays)
        print("\nRetrieved Essays:")
        for e in top_essays:
            print(e[:500] + "...\n")  # print first 500 chars for brevity

        # Prepare prompt for Qwen
        prompt = f"""
You are an IELTS expert. Analyze the following essays and provide:
1) Feedback (grammar, vocabulary, coherence, structure)
2) Suggestions to improve
3) Band score with reasoning

Essays:
{retrieved_text}
"""

        # Generate feedback
        response = generator(prompt, max_new_tokens=400, temperature=0.7)
        print("\n🔹 RAG Feedback:\n")
        print(response[0]["generated_text"])