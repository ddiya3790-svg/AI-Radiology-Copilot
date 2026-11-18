import os
import pickle

import faiss
from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# PATHS
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

VECTORSTORE_DIR = os.path.join(
    BASE_DIR,
    "rag",
    "vectorstore"
)

INDEX_PATH = os.path.join(
    VECTORSTORE_DIR,
    "medical_knowledge.index"
)

METADATA_PATH = os.path.join(
    VECTORSTORE_DIR,
    "metadata.pkl"
)


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

print("Loading RAG embedding model...")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Embedding model loaded.")


# --------------------------------------------------
# LOAD FAISS INDEX
# --------------------------------------------------

print("Loading FAISS knowledge base...")

index = faiss.read_index(INDEX_PATH)

with open(METADATA_PATH, "rb") as file:
    metadata = pickle.load(file)

print(f"Loaded {len(metadata)} knowledge chunks.")


# --------------------------------------------------
# SEARCH FUNCTION
# --------------------------------------------------

def search_knowledge(query, top_k=5):

    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True
    )

    query_embedding = query_embedding.astype("float32")

    faiss.normalize_L2(query_embedding)

    scores, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    for score, idx in zip(scores[0], indices[0]):

        if idx == -1:
            continue

        results.append({
            "score": float(score),
            "text": metadata[idx]["text"],
            "source": metadata[idx]["source"]
        })

    return results


# --------------------------------------------------
# TEST
# --------------------------------------------------

if __name__ == "__main__":

    query = "What is a glioma?"

    results = search_knowledge(query, top_k=3)

    print()
    print("====================================")
    print("RAG SEARCH TEST")
    print("====================================")

    for i, result in enumerate(results, start=1):

        print(f"\nResult {i}")
        print(f"Score  : {result['score']:.4f}")
        print(f"Source : {result['source']}")
        print(f"Text   : {result['text'][:500]}")

    print()
    print("====================================")