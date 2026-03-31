import pickle
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


DEFAULT_DATA_PATH = Path("data/processed/ielts_clean.csv")
DEFAULT_INDEX_PATH = Path("data/embeddings/faiss.index")
DEFAULT_META_PATH = Path("data/embeddings/metadata.pkl")
DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

BAND_DESCRIPTORS = {
    "task_response": {
        "high": "Addresses the task clearly with well-developed ideas.",
        "mid": "Covers the task but idea development is uneven.",
        "low": "Task coverage is limited or off-target.",
    },
    "coherence_cohesion": {
        "high": "Ideas are logically organized with smooth progression.",
        "mid": "Basic organization is visible but linking is repetitive.",
        "low": "Organization is weak and difficult to follow.",
    },
    "lexical_resource": {
        "high": "Good vocabulary range with generally precise word choice.",
        "mid": "Vocabulary is adequate but repetitive in places.",
        "low": "Limited vocabulary range with frequent misuse.",
    },
    "grammar_accuracy": {
        "high": "Uses varied structures with mostly accurate grammar.",
        "mid": "Mix of simple and complex forms with noticeable errors.",
        "low": "Frequent grammar errors reduce clarity.",
    },
}


@dataclass
class RetrievedEssay:
    row_index: int
    rank: int
    distance: float
    similarity: float
    band: float
    essay: str
    band_category: str


class LightweightRAGEvaluator:
    """Similarity-based IELTS evaluator used as an LLM placeholder."""

    def __init__(
        self,
        data_path: Path | str = DEFAULT_DATA_PATH,
        index_path: Path | str = DEFAULT_INDEX_PATH,
        meta_path: Path | str = DEFAULT_META_PATH,
        embed_model_name: str = DEFAULT_EMBED_MODEL,
        local_files_only: bool = False,
    ):
        self.data_path = Path(data_path)
        self.index_path = Path(index_path)
        self.meta_path = Path(meta_path)

        self.df = pd.read_csv(self.data_path)
        self.index = faiss.read_index(str(self.index_path))
        self.metadata = self._load_metadata(self.meta_path)
        try:
            self.embed_model = SentenceTransformer(
                embed_model_name, local_files_only=local_files_only
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to load embedding model. If running in an offline environment, "
                "download the model once online or set local_files_only=False in a networked setup."
            ) from exc

    @staticmethod
    def _load_metadata(meta_path: Path) -> Any:
        if not meta_path.exists():
            return None
        try:
            with meta_path.open("rb") as f:
                return pickle.load(f)
        except Exception as exc:
            warnings.warn(
                f"Could not deserialize metadata at {meta_path} ({exc}). "
                "Falling back to identity FAISS->row mapping. "
                "Rebuild index to regenerate metadata in a stable format.",
                RuntimeWarning,
            )
            return None

    def _map_to_row_index(self, faiss_index: int) -> int:
        if self.metadata is None:
            return faiss_index
        if isinstance(self.metadata, pd.DataFrame):
            return faiss_index
        if isinstance(self.metadata, list):
            candidate = self.metadata[faiss_index]
            if isinstance(candidate, (int, np.integer)):
                return int(candidate)
            return faiss_index
        return faiss_index

    @staticmethod
    def _band_bucket(band: float) -> str:
        if band >= 7.0:
            return "high"
        if band >= 5.0:
            return "mid"
        return "low"

    def retrieve_neighbors(self, essay_text: str, top_k: int = 5) -> list[RetrievedEssay]:
        query_emb = self.embed_model.encode([essay_text]).astype("float32")
        distances, indices = self.index.search(query_emb, top_k)
        neighbors = []
        for rank, (distance, idx) in enumerate(zip(distances[0], indices[0]), start=1):
            row_index = self._map_to_row_index(int(idx))
            row = self.df.iloc[row_index]
            band = float(row["band"])
            similarity = float(1.0 / (1.0 + distance))
            neighbors.append(
                RetrievedEssay(
                    row_index=row_index,
                    rank=rank,
                    distance=float(distance),
                    similarity=similarity,
                    band=band,
                    essay=str(row["essay"]),
                    band_category=self._band_bucket(band),
                )
            )
        return neighbors

    def predict_band(self, neighbors: list[RetrievedEssay]) -> float:
        similarities = np.array([n.similarity for n in neighbors], dtype=np.float32)
        bands = np.array([n.band for n in neighbors], dtype=np.float32)
        weighted_avg = float(np.average(bands, weights=similarities))
        return round(weighted_avg * 2) / 2

    @staticmethod
    def _feedback_tone(predicted_band: float) -> str:
        if predicted_band >= 7.0:
            return "strong"
        if predicted_band >= 5.5:
            return "developing"
        return "early"

    def generate_feedback(self, predicted_band: float, neighbors: list[RetrievedEssay]) -> dict[str, Any]:
        tone = self._feedback_tone(predicted_band)
        top_neighbor = neighbors[0]

        if tone == "strong":
            strengths = [
                "Your response likely maintains clear relevance to the prompt.",
                "Organization appears stable with generally logical progression.",
                "Language control is likely good enough for clear communication.",
            ]
            improvements = [
                "Push for more precise topic vocabulary and collocations.",
                "Use a wider mix of complex sentence structures with control.",
                "Add sharper examples or evidence to deepen idea development.",
            ]
            descriptor_level = "high"
        elif tone == "developing":
            strengths = [
                "Main points are understandable and connected to the topic.",
                "You likely show a usable structure with introduction/body/conclusion.",
                "Vocabulary appears functional for everyday argumentation.",
            ]
            improvements = [
                "Improve paragraph transitions and reduce repetitive connectors.",
                "Increase lexical variety and avoid repeating the same phrasing.",
                "Proofread for agreement, tense consistency, and punctuation.",
            ]
            descriptor_level = "mid"
        else:
            strengths = [
                "There is an attempt to answer the prompt directly.",
                "Some ideas are present and can be expanded.",
                "Basic vocabulary supports partial communication.",
            ]
            improvements = [
                "Focus first on clear paragraphing and sentence boundaries.",
                "Use simpler grammar accurately before adding complexity.",
                "Develop each main idea with one concrete supporting example.",
            ]
            descriptor_level = "low"

        descriptors = {
            criterion: BAND_DESCRIPTORS[criterion][descriptor_level]
            for criterion in BAND_DESCRIPTORS
        }
        return {
            "predicted_band": predicted_band,
            "feedback_tier": tone,
            "strengths": strengths,
            "improvements": improvements,
            "descriptor_level": descriptor_level,
            "descriptors": descriptors,
            "reference_neighbor_band": top_neighbor.band,
        }

    def evaluate(self, essay_text: str, top_k: int = 5) -> dict[str, Any]:
        neighbors = self.retrieve_neighbors(essay_text=essay_text, top_k=top_k)
        predicted_band = self.predict_band(neighbors)
        feedback = self.generate_feedback(predicted_band=predicted_band, neighbors=neighbors)
        return {
            "predicted_band": predicted_band,
            "neighbors": neighbors,
            "feedback": feedback,
        }
