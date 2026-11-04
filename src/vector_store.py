
import faiss
import numpy as np


def create_vector_store(embeddings):

    # Convert embeddings into NumPy array
    embeddings = np.array(
        embeddings,
        dtype="float32"
    )

    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings)

    # Get embedding dimensions
    dimension = embeddings.shape[1]

    # Create FAISS index using inner product
    index = faiss.IndexFlatIP(dimension)

    # Add normalized embeddings
    index.add(embeddings)

    return index


def search_similar_chunks(index, query_embedding, k=5):

    # Convert query embedding into NumPy array
    query_embedding = np.array(
        [query_embedding],
        dtype="float32"
    )

    # Normalize the question embedding
    faiss.normalize_L2(query_embedding)

    # Avoid requesting more results than available chunks
    k = min(k, index.ntotal)

    # Search for the most similar chunks
    scores, indices = index.search(
        query_embedding,
        k
    )

    return scores[0], indices[0]
