from difflib import SequenceMatcher

def evaluate_generation(reference, generated):
    similarity = SequenceMatcher(None, reference, generated).ratio()
    return similarity

# Example usage:
reference_answer = "Retrieval-augmented generation enhances text generation by incorporating relevant external information."
generated_answer = rag_pipeline("What is retrieval-augmented generation?")
score = evaluate_generation(reference_answer, generated_answer)
print("Similarity score:", score)
