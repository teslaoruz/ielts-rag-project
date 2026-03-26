# IELTS RAG Project

An intelligent, multilingual IELTS writing assistant based on **Retrieval-Augmented Generation (RAG)**.  
This project uses publicly available IELTS essay datasets from Kaggle, processes them, and builds a FAISS-based semantic search index for essay retrieval.

---

## 🚀 Setup Instructions

### 1. Prerequisites

Make sure you have installed:

- Python 3.10+  
- `pip`  
- Git  
- Kaggle API key

---

### 2. Kaggle API Key

1. Go to [Kaggle API](https://www.kaggle.com/settings/api)  
2. Click **Create New API Token** → `kaggle.json` will download  
3. Move the file to:

```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
````

---

### 3. Clone the Project

```bash
git clone https://github.com/teslaoruz/ielts-rag-project.git
cd ielts-rag-project
```

---

### 4. Setup Python Environment

```bash
python -m venv venv
source venv/bin/activate   # Linux/macOS
# venv\Scripts\activate    # Windows
pip install --upgrade pip
pip install -r requirements.txt
```

---

### 5. Create Project Directories

The project will automatically create the following structure if not present:

```text
ielts-rag-project/
├── data/
│   ├── raw/                # Kaggle datasets downloaded here
│   ├── processed/          # Cleaned and merged dataset
│   └── embeddings/         # FAISS index and metadata
├── src/                    # Source code for preprocessing, RAG, and LLM
├── notebooks/              # Notebooks for experiments
├── configs/                # Config files
└── tests/                  # Unit tests
```

---

### 6. Download Kaggle Datasets

Run these commands inside `data/raw/`:

```bash
mkdir -p data/raw
cd data/raw

# Download datasets
kaggle datasets download -d mazlumi/ielts-writing-scored-essays-dataset
kaggle datasets download -d xntrng15/ielts-writing-dataset

# Unzip the datasets
unzip -o *.zip
cd ../../
```

> All files will be saved under `data/raw/` automatically.

---

### 7. Preprocess the Data

Run the preprocessing script to clean, merge, and structure datasets:

```bash
python src/preprocessing/preprocess_all.py
```

✅ Output: `data/processed/ielts_clean.csv`

---

### 8. Build Embeddings + FAISS Index

```bash
python src/rag/build_index.py
```

✅ Outputs:

```text
data/embeddings/faiss.index
data/embeddings/metadata.pkl
```

---

### 9. Test Semantic Search

```bash
python src/rag/query_index.py
```

Example input:

```
advantages of technology in education
```

---

### 10. Ready for RAG + LLM

After preprocessing and FAISS index creation, you can build the **RAG pipeline** to generate:

* Band predictions
* Structured feedback
* Essay improvement suggestions

---

## 🧠 Notes

* Do **not upload Kaggle datasets or your API key** to GitHub
* All preprocessing and embedding steps are **reproducible**
* Ensure Python environment is activated (`source venv/bin/activate`) before running scripts

---

## 🔗 References

* [Mazlumi IELTS Scored Essays Dataset](https://www.kaggle.com/datasets/mazlumi/ielts-writing-scored-essays-dataset)
* [Xntrng15 IELTS Writing Dataset](https://www.kaggle.com/datasets/xntrng15/ielts-writing-dataset)
* [SentenceTransformers](https://www.sbert.net/)
* [FAISS](https://github.com/facebookresearch/faiss)
