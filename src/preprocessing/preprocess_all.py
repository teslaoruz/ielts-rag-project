import pandas as pd
import os
import re

RAW_DIR = "data/raw"
OUTPUT_PATH = "data/processed/ielts_clean.csv"


# -----------------------------
# Text Cleaning
# -----------------------------
def clean_text(text):
    text = str(text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)  # remove weird unicode
    return text.strip()


# -----------------------------
# Detect and Process Files
# -----------------------------
def process_file(filepath):
    print(f"Processing: {filepath}")
    
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"Skipping (not csv): {filepath}")
        return None

    # Normalize column names
    cols = [c.lower() for c in df.columns]
    df.columns = cols

    essay_col = None
    band_col = None

    # Detect columns dynamically
    for c in cols:
        if "essay" in c:
            essay_col = c
        if "score" in c or "band" in c or "overall" in c:
            band_col = c

    if essay_col is None or band_col is None:
        print(f"Skipping (missing columns): {filepath}")
        return None

    # Standardize columns
    df = df.rename(columns={
        essay_col: "essay",
        band_col: "band"
    })

    df = df[["essay", "band"]]

    return df


# -----------------------------
# Band Categorization
# -----------------------------
def band_category(b):
    if b < 5:
        return "low"
    elif b < 7:
        return "mid"
    else:
        return "high"


# -----------------------------
# Main Pipeline
# -----------------------------
def main():
    all_dfs = []

    # Walk through all raw data
    for root, dirs, files in os.walk(RAW_DIR):
        for file in files:
            if file.endswith(".csv"):
                path = os.path.join(root, file)
                df = process_file(path)

                if df is not None:
                    all_dfs.append(df)

    if not all_dfs:
        print("❌ No valid data found!")
        return

    # Merge all datasets
    df = pd.concat(all_dfs, ignore_index=True)

    print("📊 Before cleaning:", len(df))

    # -----------------------------
    # Cleaning
    # -----------------------------
    df = df.dropna(subset=["essay", "band"])

    df["essay"] = df["essay"].apply(clean_text)

    # Convert band to numeric
    df["band"] = pd.to_numeric(df["band"], errors="coerce")
    df = df.dropna(subset=["band"])

    # Keep valid IELTS range
    df = df[(df["band"] >= 0) & (df["band"] <= 9)]

    # -----------------------------
    # Quality Filtering
    # -----------------------------
    # Remove duplicates
    df = df.drop_duplicates(subset=["essay"])

    # Add essay length
    df["length"] = df["essay"].apply(lambda x: len(x.split()))

    # IELTS realistic length filter
    df = df[(df["length"] > 80) & (df["length"] < 500)]

    # Add band category
    df["band_category"] = df["band"].apply(band_category)

    # Add unique ID
    df["id"] = range(len(df))

    print("📊 After cleaning:", len(df))

    # -----------------------------
    # Save
    # -----------------------------
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"✅ Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()