# test_rag_pipeline.py
import time
import pytest
from your_rag_module import rag_pipeline  # Replace with the actual import

def test_rag_response_accuracy():
    query = "Explain retrieval-augmented generation benefits."
    result = rag_pipeline(query)
    # Basic assertion to check output is not empty
    assert result is not None and len(result) > 0, "RAG pipeline returned empty output."

def test_rag_performance():
    query = "What is RAG?"
    start_time = time.time()
    _ = rag_pipeline(query)
    end_time = time.time()
    # Assert the response is generated within 2 seconds (adjust as needed)
    assert (end_time - start_time) < 2, "RAG pipeline is too slow."
