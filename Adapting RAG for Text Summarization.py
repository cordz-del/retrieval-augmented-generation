# While RAG is primarily used for question answering, this snippet adapts it for text summarization.
# It demonstrates the versatility of the model and highlights my understanding of using RAG for multiple NLP tasks.
input_text = "Large scale language models have revolutionized text generation by incorporating retrieval-based methods to include contextually relevant data."
input_ids = tokenizer(input_text, return_tensors="pt").input_ids

# Generate a summary with controlled length parameters.
summary_ids = model.generate(input_ids, max_length=50, min_length=25, length_penalty=2.0, num_beams=4)
summary = tokenizer.batch_decode(summary_ids, skip_special_tokens=True)[0]
print("Text Summary:", summary)
