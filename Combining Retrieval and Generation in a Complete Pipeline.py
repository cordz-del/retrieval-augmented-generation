# This snippet combines retrieval and generation into a single pipeline function.
# It showcases a comprehensive understanding of RAG techniques, integrating all learned components.
def rag_pipeline(query):
    # Tokenize the input query.
    input_ids = tokenizer(query, return_tensors="pt").input_ids
    # Retrieve documents (even though we don't use retriever_outputs here, it illustrates the integration point).
    retriever_outputs = retriever(input_ids.numpy(), return_tensors="pt")
    # Generate the answer using the model.
    outputs = model.generate(input_ids)
    answer = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    return answer

sample_query = "Explain the benefits of retrieval-augmented generation."
final_answer = rag_pipeline(sample_query)
print("Final generated answer from RAG pipeline:", final_answer)
