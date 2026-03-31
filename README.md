# IELTS RAG Project

This project is an IELTS writing evaluation system built with a Retrieval-Augmented Generation (RAG) architecture.

Current implementation status:
- Dataset collection and preprocessing: complete
- Embedding model and FAISS index: complete
- Retrieval-based scoring and feedback module (lightweight LLM placeholder): complete
- Streamlit frontend demo: complete
- Local quantized LLM inference: optional/future

## System Pipeline

```text
User Essay
   -> Embedding Model
   -> FAISS Retrieval (Top-k Similar Essays)
   -> Similarity-based Band Prediction
   -> Feedback Generator (Lightweight Placeholder)
   -> Frontend UI
```

## Project Structure

```text
ielts-rag-project/
├── data/
│   ├── raw/
│   ├── processed/
│   └── embeddings/
├── src/
│   ├── preprocessing/
│   └── rag/
│       ├── build_index.py
│       ├── query_index.py
│       ├── lightweight_inference.py
│       └── demo_lightweight.py
├── streamlit_app.py
├── requirements.txt
└── README.md
```

## Setup

1. Clone and enter project:

```bash
git clone https://github.com/teslaoruz/ielts-rag-project.git
cd ielts-rag-project
```

2. Create and activate environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

3. Download Kaggle datasets into `data/raw/`:

```bash
mkdir -p data/raw
cd data/raw
kaggle datasets download -d mazlumi/ielts-writing-scored-essays-dataset
kaggle datasets download -d xntrng15/ielts-writing-dataset
unzip -o "*.zip"
cd ../../
```

4. Preprocess data:

```bash
python src/preprocessing/preprocess_all.py
```

5. Build embeddings and FAISS index:

```bash
python src/rag/build_index.py
```

## Demo Options

### 1) Streamlit Frontend (recommended for presentation)

```bash
streamlit run streamlit_app.py
```

UI includes:
- Essay input box
- Top-k similar essay retrieval from FAISS
- Similarity-weighted predicted IELTS band
- Band descriptor summary
- Strength and improvement feedback

### 2) Terminal Demo

```bash
python src/rag/demo_lightweight.py
```

## Lightweight Inference Module

Because local quantized LLM inference may be blocked by hardware constraints, the project includes a fallback module:
- Retrieves nearest essays from FAISS
- Computes similarity-weighted band estimate
- Generates structured feedback via rule-based templates

This is implemented in:
- `src/rag/lightweight_inference.py`

Suggested academic framing:
> Due to hardware constraints, quantized LLM deployment is simulated. The full RAG retrieval and similarity-based evaluation pipeline is fully implemented and validated.

## Outputs

Core generated artifacts:
- `data/processed/ielts_clean.csv`
- `data/embeddings/faiss.index`
- `data/embeddings/metadata.pkl`

## Notes

- Do not commit Kaggle credentials (`~/.kaggle/kaggle.json`) to git.
- The current system is demo-ready without local LLM quantization.
- Optional next step: connect an external LLM API for richer natural-language feedback while keeping retrieval local.
- If you see a metadata pickle compatibility error, run `python src/rag/build_index.py` to regenerate `data/embeddings/metadata.pkl`.
