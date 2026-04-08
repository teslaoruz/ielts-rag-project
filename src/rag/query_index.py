import pickle
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

DATA_PATH = Path("data/processed/ielts_clean.csv")
INDEX_PATH = Path("data/embeddings/faiss.index")
META_PATH = Path("data/embeddings/metadata.pkl")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5


def resolve_score_column(df: pd.DataFrame) -> str | None:
    lower_to_original = {column.lower(): column for column in df.columns}
    for candidate in ("band", "overall", "score"):
        if candidate in lower_to_original:
            return lower_to_original[candidate]

    for column in df.columns:
        lowered = column.lower()
        if "band" in lowered or "overall" in lowered or "score" in lowered:
            return column

    return None


def load_metadata(meta_path: Path):
    if not meta_path.exists():
        return None
    try:
        with meta_path.open("rb") as f:
            return pickle.load(f)
    except Exception as exc:
        print(f"Warning: metadata load failed ({exc}). Falling back to direct index mapping.")
        return None


def map_to_row_index(faiss_index: int, metadata) -> int:
    if metadata is None:
        return faiss_index
    if isinstance(metadata, list) and 0 <= faiss_index < len(metadata):
        mapped = metadata[faiss_index]
        if isinstance(mapped, (int, np.integer)):
            return int(mapped)
    return faiss_index


def main():
    model = SentenceTransformer(MODEL_NAME)
    index = faiss.read_index(str(INDEX_PATH))
    essays = pd.read_csv(DATA_PATH)
    metadata = load_metadata(META_PATH)
    score_column = resolve_score_column(essays)
    score_label = score_column if score_column else "score"

    query = input("Enter your query: ").strip()
    if not query:
        print("Query is empty. Exiting.")
        return

    query_emb = model.encode([query]).astype("float32")
    distances, indices = index.search(query_emb, k=TOP_K)

    print(f"\nTop {TOP_K} similar essays:\n")
    for rank, (distance, faiss_idx) in enumerate(zip(distances[0], indices[0]), start=1):
        if int(faiss_idx) < 0:
            continue

        row_idx = map_to_row_index(int(faiss_idx), metadata)
        if row_idx < 0 or row_idx >= len(essays):
            print(f"--- Rank {rank} | Skipped (row index {row_idx} is out of bounds) ---")
            continue

        row = essays.iloc[row_idx]
        score_value = row.get(score_column, "N/A") if score_column else "N/A"
        essay_text = str(row.get("essay", "[missing essay text]"))

        print(
            f"--- Rank {rank} | Row {row_idx + 1} | "
            f"{score_label}: {score_value} | Distance: {float(distance):.4f} ---"
        )
        print(essay_text)
        print("")


if __name__ == "__main__":
    main()
