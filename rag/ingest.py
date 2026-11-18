import os
import pickle
import re

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# PATHS
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DOCUMENTS_DIR = os.path.join(
    BASE_DIR,
    "rag",
    "documents"
)

VECTORSTORE_DIR = os.path.join(
    BASE_DIR,
    "rag",
    "vectorstore"
)

os.makedirs(VECTORSTORE_DIR, exist_ok=True)


# --------------------------------------------------
# EMBEDDING MODEL
# --------------------------------------------------

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Embedding model loaded.")


# --------------------------------------------------
# CHUNKING FUNCTION
# --------------------------------------------------

def create_chunks(text, source):
    """
    Creates cleaner chunks based on paragraphs and sections.
    Avoids cutting sentences in the middle whenever possible.
    """

    # Normalize line endings
    text = text.replace("\r\n", "\n")

    # Split around blank lines
    paragraphs = re.split(r"\n\s*\n", text)

    chunks = []

    current_chunk = ""

    for paragraph in paragraphs:

        paragraph = paragraph.strip()

        if not paragraph:
            continue

        # If adding the paragraph keeps the chunk reasonable
        if len(current_chunk) + len(paragraph) <= 1000:

            if current_chunk:
                current_chunk += "\n\n" + paragraph
            else:
                current_chunk = paragraph

        else:

            if current_chunk:
                chunks.append({
                    "text": current_chunk.strip(),
                    "source": source
                })

            current_chunk = paragraph

    # Add final chunk
    if current_chunk:
        chunks.append({
            "text": current_chunk.strip(),
            "source": source
        })

    return chunks


# --------------------------------------------------
# LOAD DOCUMENTS
# --------------------------------------------------

documents = []

for filename in sorted(os.listdir(DOCUMENTS_DIR)):

    if not filename.endswith(".txt"):
        continue

    filepath = os.path.join(
        DOCUMENTS_DIR,
        filename
    )

    with open(
        filepath,
        "r",
        encoding="utf-8"
    ) as file:

        text = file.read()

    documents.append({
        "source": filename,
        "text": text
    })


print(f"Loaded {len(documents)} documents.")


# --------------------------------------------------
# CREATE CLEAN CHUNKS
# --------------------------------------------------

chunks = []

for document in documents:

    document_chunks = create_chunks(
        document["text"],
        document["source"]
    )

    chunks.extend(document_chunks)


print(f"Created {len(chunks)} clean chunks.")


# --------------------------------------------------
# DISPLAY CHUNK INFORMATION
# --------------------------------------------------

print()
print("Chunk distribution:")

for source in sorted(
    set(chunk["source"] for chunk in chunks)
):

    count = sum(
        1
        for chunk in chunks
        if chunk["source"] == source
    )

    print(f"  {source}: {count} chunks")


# --------------------------------------------------
# CREATE EMBEDDINGS
# --------------------------------------------------

texts = [
    chunk["text"]
    for chunk in chunks
]

print()
print("Creating embeddings...")

embeddings = embedding_model.encode(
    texts,
    convert_to_numpy=True,
    show_progress_bar=True
)

embeddings = embeddings.astype(
    "float32"
)


# --------------------------------------------------
# NORMALIZE EMBEDDINGS
# --------------------------------------------------

faiss.normalize_L2(
    embeddings
)


# --------------------------------------------------
# CREATE FAISS INDEX
# --------------------------------------------------

dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(
    dimension
)

index.add(
    embeddings
)


# --------------------------------------------------
# SAVE VECTOR DATABASE
# --------------------------------------------------

index_path = os.path.join(
    VECTORSTORE_DIR,
    "medical_knowledge.index"
)

metadata_path = os.path.join(
    VECTORSTORE_DIR,
    "metadata.pkl"
)

faiss.write_index(
    index,
    index_path
)

with open(
    metadata_path,
    "wb"
) as file:

    pickle.dump(
        chunks,
        file
    )


# --------------------------------------------------
# FINAL STATUS
# --------------------------------------------------

print()
print("====================================")
print("RAG KNOWLEDGE BASE UPDATED")
print("====================================")

print(f"Documents : {len(documents)}")
print(f"Chunks    : {len(chunks)}")
print(f"Embeddings: {len(embeddings)}")
print(f"Dimension : {dimension}")

print()
print("FAISS index:")
print(index_path)

print()
print("Metadata:")
print(metadata_path)

print("====================================")