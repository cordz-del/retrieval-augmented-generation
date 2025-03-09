# This snippet shows how to use the retriever component to fetch relevant documents for a given query.
# It illustrates the retrieval aspect of RAG, where the model identifies external information to enhance generation.
query = "What is retrieval-augmented generation?"
input_ids = tokenizer(query, return_tensors="pt").input_ids

# Retrieve documents from the internal index.
retrieved_outputs = retriever(input_ids.numpy(), return_tensors="pt")
print("Retrieved document IDs:", retrieved_outputs["doc_ids"])
