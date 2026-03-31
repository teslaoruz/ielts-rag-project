try:
    from src.rag.lightweight_inference import LightweightRAGEvaluator
except ModuleNotFoundError as exc:
    if exc.name != "src":
        raise
    from lightweight_inference import LightweightRAGEvaluator


def main():
    evaluator = LightweightRAGEvaluator()
    print("IELTS RAG Lightweight Demo")
    print("Type 'exit' to quit.\n")

    while True:
        essay = input("Paste essay text:\n> ").strip()
        if essay.lower() in {"exit", "quit"}:
            break
        if not essay:
            print("Please enter essay text.\n")
            continue

        result = evaluator.evaluate(essay, top_k=5)
        print(f"\nPredicted band: {result['predicted_band']:.1f}")
        print("Top retrieved essays:")
        for n in result["neighbors"]:
            print(f"- Rank {n.rank} | Band {n.band:.1f} | Similarity {n.similarity:.3f}")
        print("\nFeedback:")
        for s in result["feedback"]["strengths"]:
            print(f"  + {s}")
        for i in result["feedback"]["improvements"]:
            print(f"  - {i}")
        print("")


if __name__ == "__main__":
    main()
