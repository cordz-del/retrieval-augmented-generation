import os
import pickle

# This snippet demonstrates caching outputs from the RAG model to avoid redundant computation.
# It illustrates best practices for efficiency and production deployment learned during the certification.
cache_file = "rag_output_cache.pkl"

def cache_output(query, output):
    cache = {}
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            cache = pickle.load(f)
    cache[query] = output
    with open(cache_file, "wb") as f:
        pickle.dump(cache, f)

def get_cached_output(query):
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            cache = pickle.load(f)
        return cache.get(query, None)
    return None

# Example usage:
cached = get_cached_output("What is RAG?")
if cached:
    print("Using cached output:", cached)
else:
    # Use the previously generated output from snippet 3 as a placeholder.
    output = generated_text  
    cache_output("What is RAG?", output)
    print("Output cached for future use.")
