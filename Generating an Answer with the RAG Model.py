# This snippet demonstrates generating an answer using the pre-trained RAG model.
# It shows how the model fuses the retrieved documents with the input query to produce a coherent answer.
outputs = model.generate(input_ids)
generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
print("Generated answer:", generated_text)
