from rag.retriever import search_knowledge
from rag.rag_context import build_rag_query


def retrieve_for_prediction(
    predicted_class,
    confidence,
    user_question,
    top_k=5
):
    """
    Retrieves medical knowledge relevant to the
    current MRI prediction and user question.
    """

    # Build context-aware query
    query = build_rag_query(
        predicted_class=predicted_class,
        confidence=confidence,
        user_question=user_question
    )

    # Retrieve relevant medical knowledge
    results = search_knowledge(
        query,
        top_k=top_k
    )

    return {
        "query": query,
        "prediction": {
            "class": predicted_class,
            "confidence": confidence
        },
        "sources": results
    }


if __name__ == "__main__":

    result = retrieve_for_prediction(
        predicted_class="glioma",
        confidence=92.45,
        user_question="What does my result mean?",
        top_k=3
    )

    print()
    print("====================================")
    print("CONTEXT-AWARE RAG TEST")
    print("====================================")

    print("\nPrediction:")
    print(
        f"Class      : {result['prediction']['class']}"
    )
    print(
        f"Confidence : {result['prediction']['confidence']:.2f}%"
    )

    print("\nRetrieved Evidence:")

    for i, source in enumerate(
        result["sources"],
        start=1
    ):
        print()
        print(f"--- Evidence {i} ---")
        print(
            f"Similarity : {source['score']:.4f}"
        )
        print(
            f"Source     : {source['source']}"
        )
        print(
            f"Content    : {source['text'][:400]}"
        )

    print()
    print("====================================")