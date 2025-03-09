from transformers import pipeline

# Leveraging Hugging Face's pipeline API, this snippet creates an end-to-end question answering pipeline.
# It underlines the practical integration of retrieval and generation within a streamlined framework.
qa_pipeline = pipeline("text2text-generation", model="facebook/rag-token-base", tokenizer="facebook/rag-token-base")
question = "How does retrieval-augmented generation improve text generation?"
result = qa_pipeline(question)
print("QA Pipeline Result:", result)
