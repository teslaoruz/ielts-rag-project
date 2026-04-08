"""
Local RAG Feedback (optional GPTQ LLM)
--------------------------------------
Retrieves similar essays from FAISS and optionally generates feedback
with a local GPTQ model. If GPTQ dependencies are missing, the script
still runs and prints a useful retrieval summary.
"""

from __future__ import annotations

import csv
import pickle
import re
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer


ROOT_DIR = Path(__file__).resolve().parents[2]
FAISS_INDEX_PATH = ROOT_DIR / "data" / "embeddings" / "faiss.index"
METADATA_PATH = ROOT_DIR / "data" / "embeddings" / "metadata.pkl"
PROCESSED_CSV_PATH = ROOT_DIR / "data" / "processed" / "ielts_clean.csv"

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
QWEN_MODEL_PATH = "Mohaaxa/qwen2.5-1.5b-gptq-4bit-v2"
MAX_REFERENCE_ESSAYS = 3
MAX_WORDS_PER_REFERENCE = 170
ENABLE_LLM_GENERATION = True
STRICT_SECTION_FORMAT = True
SHOW_RAW_OUTPUT_ON_FALLBACK = False
RAW_OUTPUT_PREVIEW_CHARS = 700


def truncate_words(text: str, max_words: int) -> str:
    words = str(text).split()
    if len(words) <= max_words:
        return str(text)
    return " ".join(words[:max_words]) + " ..."


def sanitize_query(raw_query: str) -> str:
    query = str(raw_query).strip()

    # If user pasted a full CSV row, prefer the first text field as essay input.
    if (query.count(",") >= 8 and '"' in query) or '","' in query:
        try:
            fields = next(csv.reader([query]))
            if fields:
                best = max(fields, key=lambda f: len(str(f).split()))
                if len(best.split()) >= 20:
                    query = best
        except Exception:
            pass

    query = query.replace("\r", " ").replace("\n", " ")
    query = re.sub(r"\s+", " ", query).strip(" \"'")
    if query.startswith(". "):
        query = query[2:].strip()
    return query


def is_degenerate_output(text: str) -> bool:
    cleaned = str(text).strip()
    if len(cleaned) < 40:
        return True
    if re.search(r"([!?.])\1{20,}", cleaned):
        return True
    if cleaned.count("!") > max(30, int(0.2 * len(cleaned))):
        return True
    unique_ratio = len(set(cleaned)) / max(1, len(cleaned))
    if unique_ratio < 0.08:
        return True
    ascii_ratio = sum(1 for ch in cleaned if 32 <= ord(ch) <= 126) / max(1, len(cleaned))
    if len(cleaned) >= 120 and ascii_ratio < 0.78:
        return True
    latin_letter_ratio = sum(1 for ch in cleaned if ch.isalpha() and ord(ch) < 128) / max(1, len(cleaned))
    if len(cleaned) >= 120 and latin_letter_ratio < 0.35:
        return True
    return False


def has_expected_feedback_sections(text: str) -> bool:
    normalized = str(text).lower()
    required = [
        "estimated band",
        "strengths",
        "weaknesses",
        "top 3 improvements",
        "one improved sample paragraph",
    ]
    return all(section in normalized for section in required)


def build_structured_fallback(query: str, retrieved: list[dict], score_label: str) -> str:
    weighted_scores = []
    for item in retrieved:
        try:
            score = float(item["score"])
        except (TypeError, ValueError):
            continue
        similarity = 1.0 / (1.0 + float(item["distance"]))
        weighted_scores.append((score, similarity))

    if weighted_scores:
        numerator = sum(score * sim for score, sim in weighted_scores)
        denominator = sum(sim for _, sim in weighted_scores) or 1.0
        predicted = round((numerator / denominator) * 2) / 2
    else:
        predicted = "N/A"

    return (
        "Estimated Band:\n"
        f"- {predicted} (retrieval-weighted estimate from nearest essays by {score_label})\n\n"
        "Strengths:\n"
        "- Main position is present and relevant to the prompt.\n"
        "- Core ideas are understandable and supported to some extent.\n\n"
        "Weaknesses:\n"
        "- Grammar accuracy and sentence control reduce clarity in parts.\n"
        "- Lexical repetition limits precision and style.\n"
        "- Cohesion can be improved with clearer paragraph links.\n\n"
        "Top 3 Improvements:\n"
        "1. Use one clear topic sentence per body paragraph.\n"
        "2. Add one concrete example after each main claim.\n"
        "3. Proofread for articles, verb forms, and punctuation.\n\n"
        "One Improved Sample Paragraph:\n"
        "Overall, a stronger response should begin with a clear overview, followed by direct "
        "comparisons of the most important features. In each body paragraph, present one key point, "
        "support it with precise detail, and explain why that detail matters. This approach improves "
        "coherence, makes your argument easier to follow, and helps you achieve a more consistent "
        "academic style."
    )


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


def load_optional_generator():
    try:
        from transformers import AutoTokenizer
        from auto_gptq import AutoGPTQForCausalLM
    except ModuleNotFoundError:
        return None

    print("Loading Qwen2.5-1.5B 4-bit model... (may take a minute)")
    try:
        tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_PATH, trust_remote_code=True)
        model = AutoGPTQForCausalLM.from_quantized(
            QWEN_MODEL_PATH,
            device_map="auto",
            use_safetensors=True,
            trust_remote_code=True,
        )

        def generate_text(prompt: str, max_new_tokens: int = 320) -> str:
            if hasattr(tokenizer, "apply_chat_template"):
                messages = [
                    {
                        "role": "system",
                        "content": (
                            "You are an IELTS Writing examiner. Provide clear, structured, "
                            "practical feedback in plain English."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ]
                rendered_prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            else:
                rendered_prompt = prompt

            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            inputs = tokenizer(rendered_prompt, return_tensors="pt").to(device)
            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    repetition_penalty=1.22,
                    no_repeat_ngram_size=4,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            generated_ids = output_ids[0][inputs["input_ids"].shape[1] :]
            return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        return generate_text
    except Exception as exc:
        print(f"Warning: local GPTQ model load failed ({exc}). Using fallback retrieval mode.")
        return None


def load_metadata(meta_path: Path):
    if not meta_path.exists():
        return None
    try:
        with meta_path.open("rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def map_to_row_index(faiss_index: int, metadata) -> int:
    if metadata is None:
        return faiss_index
    if isinstance(metadata, list) and 0 <= faiss_index < len(metadata):
        mapped = metadata[faiss_index]
        if isinstance(mapped, (int, np.integer)):
            return int(mapped)
    return faiss_index


def retrieve_essays(
    query: str,
    top_k: int,
    embed_model: SentenceTransformer,
    index,
    metadata,
    essays_df: pd.DataFrame,
    score_column: str | None,
) -> list[dict]:
    query_emb = embed_model.encode([query]).astype("float32")
    distances, indices = index.search(query_emb, top_k)
    retrieved = []
    for rank, (distance, idx) in enumerate(zip(distances[0], indices[0]), start=1):
        if int(idx) < 0:
            continue
        row_idx = map_to_row_index(int(idx), metadata)
        if row_idx < 0 or row_idx >= len(essays_df):
            continue

        row = essays_df.iloc[row_idx]
        score_value = row.get(score_column, "N/A") if score_column else "N/A"
        retrieved.append(
            {
                "rank": rank,
                "row_index": row_idx,
                "distance": float(distance),
                "score": score_value,
                "essay": str(row.get("essay", "")),
            }
        )
    return retrieved


def print_retrieved_essays(retrieved: list[dict], score_label: str) -> None:
    print("\nRetrieved Essays:")
    for item in retrieved:
        header = (
            f"--- Rank {item['rank']} | Row {item['row_index'] + 1} | "
            f"{score_label}: {item['score']} | Distance: {item['distance']:.4f} ---"
        )
        print(header)
        print(item["essay"][:500] + "...\n")


def print_fallback_feedback(retrieved: list[dict]) -> None:
    numeric_scores = []
    for item in retrieved:
        try:
            numeric_scores.append(float(item["score"]))
        except (TypeError, ValueError):
            continue

    print("\nFallback feedback (no local GPTQ model loaded):")
    print("- Retrieved essays can still be used as quality references.")
    if numeric_scores:
        avg_score = sum(numeric_scores) / len(numeric_scores)
        print(f"- Average reference score from top results: {avg_score:.2f}")
        print(f"- Reference score range: {min(numeric_scores):.1f} to {max(numeric_scores):.1f}")
    else:
        print("- No numeric score field detected in retrieved rows.")
    try:
        import torch
        cuda_available = bool(torch.cuda.is_available())
        torch_cuda = str(torch.version.cuda)
    except Exception:
        cuda_available = False
        torch_cuda = "unknown"

    if cuda_available:
        print("- CUDA is available. You can try GPTQ with compatible CUDA torch + auto-gptq.")
        print("- Install command:")
        print("  pip install --no-build-isolation auto-gptq")
    else:
        print("- GPU GPTQ is currently unavailable in this environment (CPU-only torch detected).")
        print(f"- Detected torch CUDA version: {torch_cuda}")
        print("- If you want GPTQ, install a CUDA-enabled torch build first, then install auto-gptq.")


def main():
    index = faiss.read_index(str(FAISS_INDEX_PATH))
    metadata = load_metadata(METADATA_PATH)
    essays_df = pd.read_csv(PROCESSED_CSV_PATH)
    score_column = resolve_score_column(essays_df)
    score_label = score_column if score_column else "score"

    print("Loading embedding model...")
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    generator = None
    if ENABLE_LLM_GENERATION:
        generator = load_optional_generator()
        if generator is None:
            print("Note: local GPTQ generation is unavailable. Using structured fallback mode.")
    else:
        print("Local GPTQ generation is disabled. Using stable structured feedback mode.")

    print("IELTS RAG Feedback System")
    while True:
        query = input("\nEnter your essay topic or question (or 'exit' to quit):\n> ").strip()
        if query.lower() in ("exit", "quit"):
            break
        if not query:
            print("Please enter a non-empty query.")
            continue
        lower_query = query.lower()
        if "python.exe" in lower_query or "rag_feedback_local.py" in lower_query:
            print("That looks like a shell command. Please paste your essay text instead.")
            continue
        query = sanitize_query(query)
        if len(query.split()) < 20:
            print("Please paste a full essay response (at least ~20 words).")
            continue

        retrieved = retrieve_essays(
            query=query,
            top_k=5,
            embed_model=embed_model,
            index=index,
            metadata=metadata,
            essays_df=essays_df,
            score_column=score_column,
        )
        if not retrieved:
            print("No essays were retrieved. Check index and metadata alignment.")
            continue

        print_retrieved_essays(retrieved, score_label=score_label)
        prompt_references = retrieved[:MAX_REFERENCE_ESSAYS]
        retrieved_text = "\n\n".join(
            [
                (
                    f"Rank {item['rank']} | {score_label}: {item['score']}\n"
                    f"{truncate_words(item['essay'], MAX_WORDS_PER_REFERENCE)}"
                )
                for item in prompt_references
            ]
        )

        if generator is None:
            response = build_structured_fallback(query=query, retrieved=retrieved, score_label=score_label)
            print("\nRAG Feedback:\n")
            print(response)
            continue

        prompt = f"""
Evaluate the student's IELTS writing using the reference essays.
Write in plain English only.
Do not output random symbols or non-English tokens.

Return exactly these sections:
Estimated Band:
Strengths:
Weaknesses:
Top 3 Improvements:
One Improved Sample Paragraph:

Student submission:
{query}

Reference essays:
{retrieved_text}
"""
        response = generator(prompt, max_new_tokens=220)
        low_quality = is_degenerate_output(response)
        missing_sections = not has_expected_feedback_sections(response)
        if low_quality or (STRICT_SECTION_FORMAT and missing_sections):
            if SHOW_RAW_OUTPUT_ON_FALLBACK and response:
                preview = response[:RAW_OUTPUT_PREVIEW_CHARS]
                if len(response) > RAW_OUTPUT_PREVIEW_CHARS:
                    preview += " ..."
                print("\nRaw LLM output preview (debug):")
                print(preview)
            print("Warning: model output was low-quality; using structured fallback.")
            response = build_structured_fallback(query=query, retrieved=retrieved, score_label=score_label)
        elif missing_sections:
            print("Note: LLM output format differs from template; showing raw model output.")
        print("\nRAG Feedback:\n")
        print(response if response else "[No output generated]")


if __name__ == "__main__":
    main()
