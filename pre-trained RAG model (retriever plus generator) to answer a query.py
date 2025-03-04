from transformers import RagTokenizer, RagRetriever, RagTokenForGeneration

# Load the RAG tokenizer, retriever, and model
tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-base")
retriever = RagRetriever.from_pretrained(
    "facebook/rag-token-base",
    index_name="exact",  # Use the "exact" index (or choose "compressed", etc.)
    use_dummy_dataset=True  # For demo purposes; replace with your actual dataset
)
model = RagTokenForGeneration.from_pretrained("facebook/rag-token-base", retriever=retriever)

# Define a query to ask the model
query = "What are the health benefits of green tea?"

# Prepare input and generate a response
input_dict = tokenizer.prepare_seq2seq_batch(query, return_tensors="pt")
generated_ids = model.generate(input_ids=input_dict["input_ids"])
generated_answer = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

print("Generated Answer:", generated_answer[0])
