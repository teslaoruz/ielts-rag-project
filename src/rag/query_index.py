import faiss
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd

# Load model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Load FAISS index
index = faiss.read_index("data/embeddings/faiss.index")

# Load metadata (original essays)
essays = pd.read_csv("data/processed/ielts_clean.csv")  # now it's a DataFrame

# Input query
query = input("Enter your query: ")
query_emb = model.encode([query]).astype('float32')

# Search top 5
D, I = index.search(query_emb, k=5)

print("\nTop 5 similar essays:\n")
for idx in I[0]:
    row = essays.iloc[idx]
    print(f"--- Essay {idx+1} | Band: {row['band']} ---")
    print(row['essay'])
    print("\n")