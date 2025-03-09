# This snippet demonstrates a simple evaluation of the generated answer by comparing it with a reference.
# It shows my understanding of evaluation metrics and qualitative assessment in RAG applications.
reference_answer = "Retrieval-augmented generation enhances text generation by incorporating relevant external information."
# Using the generated_text from snippet 3.
from difflib import SequenceMatcher
similarity = SequenceMatcher(None, reference_answer, generated_text).ratio()
print("Similarity score between generated and reference answer:", similarity)
