def build_rag_query(predicted_class, confidence, user_question):
    """
    Builds a context-aware RAG query using
    the current MRI prediction.
    """

    query = f"""
Current AI MRI classification:
Predicted class: {predicted_class}
Model confidence: {confidence:.2f}%

User question:
{user_question}

Provide information relevant to the predicted class,
while clearly explaining that the AI classification is
not a confirmed medical diagnosis.
"""

    return query.strip()


if __name__ == "__main__":

    query = build_rag_query(
        predicted_class="glioma",
        confidence=92.45,
        user_question="What does this result mean?"
    )

    print("====================================")
    print("RAG CONTEXT TEST")
    print("====================================")
    print(query)