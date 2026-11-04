
from src.embeddings import create_embeddings, model
from src.vector_store import (
    create_vector_store,
    search_similar_chunks
)


def build_rag_pipeline(chunks):
    """
    Create embeddings and build the FAISS vector database.
    """

    if not chunks:
        raise ValueError("No PDF chunks available.")

    embeddings = create_embeddings(chunks)
    index = create_vector_store(embeddings)

    return index


def retrieve_relevant_chunks(
    question,
    chunks,
    index,
    k=8,
    min_similarity=0.25,
    neighbor_window=0,
    max_results=8
):
    """
    Retrieve relevant PDF chunks.

    Features:
    1. Similarity-based retrieval.
    2. Relevance filtering.
    3. Optional neighbouring chunk expansion.
    4. Duplicate prevention.
    5. Page and similarity tracking.
    """

    if not isinstance(question, str) or not question.strip():
        return []

    if not chunks or index is None:
        return []

    # STEP 1: Create question embedding
    query_embedding = model.encode(
        [question]
    )[0]

    # STEP 2: Search FAISS
    scores, indices = search_similar_chunks(
        index,
        query_embedding,
        k
    )

    # STEP 3: Collect valid direct matches
    matched_chunks = []
    seen_indices = set()

    for score, chunk_index in zip(scores, indices):

        chunk_index = int(chunk_index)
        score = float(score)

        if chunk_index < 0 or chunk_index >= len(chunks):
            continue

        if chunk_index in seen_indices:
            continue

        seen_indices.add(chunk_index)

        matched_chunks.append({
            "index": chunk_index,
            "score": score
        })

    if not matched_chunks:
        return []

    # STEP 4: Filter by relevance
    best_score = max(
        item["score"] for item in matched_chunks
    )

    threshold = max(
        min_similarity,
        best_score * 0.70
    )

    relevant_matches = [
        item
        for item in matched_chunks
        if item["score"] >= threshold
    ]

    # If filtering removes everything, retain the best match.
    if not relevant_matches:
        relevant_matches = [matched_chunks[0]]

    # STEP 5: Select direct matches first
    selected = {}

    for item in relevant_matches:

        chunk_index = item["index"]

        selected[chunk_index] = {
            "score": item["score"],
            "retrieval_type": "direct_match"
        }

    # STEP 6: Optional neighbouring context
    # Disabled by default to reduce irrelevant passages.
    if neighbor_window > 0:

        for item in relevant_matches:

            chunk_index = item["index"]

            start = max(
                0,
                chunk_index - neighbor_window
            )

            end = min(
                len(chunks),
                chunk_index + neighbor_window + 1
            )

            for neighbour_index in range(start, end):

                if neighbour_index not in selected:

                    selected[neighbour_index] = {
                        "retrieval_type": "neighbour"
                    }

    # STEP 7: Limit results
    direct_indices = [
        item["index"] for item in relevant_matches
    ]

    selected_indices = sorted(
        selected.keys(),
        key=lambda idx: (
            0 if idx in direct_indices else 1,
            -selected[idx].get("score", -1)
        )
    )

    selected_indices = selected_indices[:max_results]

    # Restore document order for the final context.
    selected_indices.sort()

    # STEP 8: Prepare final results
    results = []

    for chunk_index in selected_indices:

        chunk = chunks[chunk_index].copy()
        metadata = selected[chunk_index]

        chunk["retrieval_type"] = metadata[
            "retrieval_type"
        ]

        if "score" in metadata:
            chunk["similarity_score"] = metadata["score"]

        results.append(chunk)

    return results
