import numpy as np

# This snippet demonstrates a custom retrieval function using an external document store.
# It reflects my ability to integrate external data sources with RAG by computing simple cosine similarities.
def custom_retriever(query, document_store):
    # In a real scenario, you would generate a query embedding here.
    query_embedding = np.random.rand(768)  # Placeholder for the actual query embedding.
    similarities = [
        np.dot(query_embedding, doc_emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(doc_emb))
        for doc_emb in document_store
    ]
    top_idx = np.argmax(similarities)
    return top_idx, document_store[top_idx]

# Example external document store with dummy embeddings.
document_store = [
    np.random.rand(768),  # Dummy embedding for document 1.
    np.random.rand(768),  # Dummy embedding for document 2.
]
query = "What is RAG?"
top_index, top_document = custom_retriever(query, document_store)
print("Most relevant document index:", top_index)
