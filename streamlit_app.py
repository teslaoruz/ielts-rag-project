import pandas as pd
import streamlit as st

from src.rag.lightweight_inference import LightweightRAGEvaluator

st.set_page_config(page_title="IELTS RAG Demo", layout="wide")


@st.cache_resource
def load_evaluator():
    return LightweightRAGEvaluator()


def main():
    st.title("IELTS Writing RAG Demo")
    st.caption("Embedding + FAISS retrieval + lightweight inference placeholder")

    with st.sidebar:
        st.header("Demo Controls")
        top_k = st.slider("Top-k retrieved essays", min_value=3, max_value=10, value=5)
        st.markdown(
            "This demo uses a similarity-weighted scoring and template feedback module "
            "as an LLM placeholder."
        )

    essay_input = st.text_area(
        "Paste IELTS essay text",
        height=250,
        placeholder="Write or paste an IELTS Task 2 essay here...",
    )

    if st.button("Evaluate"):
        if not essay_input.strip():
            st.warning("Please enter an essay before evaluation.")
            return

        evaluator = load_evaluator()
        result = evaluator.evaluate(essay_text=essay_input, top_k=top_k)
        predicted_band = result["predicted_band"]
        feedback = result["feedback"]
        neighbors = result["neighbors"]

        left, right = st.columns([1, 2])
        with left:
            st.metric("Predicted Band", f"{predicted_band:.1f}")
            st.markdown("**Inference Module**: Lightweight placeholder")
            st.markdown("**Descriptor Tier**: " + feedback["descriptor_level"].capitalize())

        with right:
            st.subheader("Band Descriptor Summary")
            descriptor_df = pd.DataFrame(
                [
                    {"Criterion": "Task Response", "Summary": feedback["descriptors"]["task_response"]},
                    {"Criterion": "Coherence & Cohesion", "Summary": feedback["descriptors"]["coherence_cohesion"]},
                    {"Criterion": "Lexical Resource", "Summary": feedback["descriptors"]["lexical_resource"]},
                    {"Criterion": "Grammar Accuracy", "Summary": feedback["descriptors"]["grammar_accuracy"]},
                ]
            )
            st.table(descriptor_df)

        st.subheader("Feedback")
        st.markdown("**Strengths**")
        for item in feedback["strengths"]:
            st.write(f"- {item}")
        st.markdown("**Improvements**")
        for item in feedback["improvements"]:
            st.write(f"- {item}")

        st.subheader(f"Top {len(neighbors)} Retrieved Essays")
        for n in neighbors:
            with st.expander(
                f"Rank {n.rank} | Band {n.band:.1f} | Similarity {n.similarity:.3f} | Row {n.row_index}"
            ):
                st.write(n.essay)


if __name__ == "__main__":
    main()
