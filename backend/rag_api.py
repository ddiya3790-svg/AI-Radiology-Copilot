from rag.rag_answer import generate_rag_answer


def ask_medical_assistant(
    predicted_class,
    confidence,
    question
):
    """
    Generate an evidence-grounded answer using
    the current MRI prediction and RAG knowledge base.
    """

    result = generate_rag_answer(
        predicted_class=predicted_class,
        confidence=confidence,
        user_question=question,
        top_k=5
    )

    return result