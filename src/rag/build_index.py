import pandas as pd
import numpy as np
import os
import faiss
import pickle
from sentence_transformers import SentenceTransformer

DATA_PATH = "data/processed/ielts_clean.csv"
INDEX_PATH = "data/embeddings/faiss.index"
META_PATH = "data/embeddings/metadata.pkl"


def main():
    print("📥 Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    texts = df["essay"].tolist()

    print("🧠 Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("⚡ Creating embeddings...")
    embeddings = model.encode(texts, show_progress_bar=True)

    embeddings = np.array(embeddings).astype("float32")

    print("📦 Building FAISS index...")
    dim = embeddings.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    # Save index
    os.makedirs("data/embeddings", exist_ok=True)
    faiss.write_index(index, INDEX_PATH)

    # Save metadata as plain integer row mapping to avoid pandas pickle compatibility issues.
    row_map = list(range(len(df)))
    with open(META_PATH, "wb") as f:
        pickle.dump(row_map, f)

    print("✅ FAISS index built and saved!")


if __name__ == "__main__":
    main()
