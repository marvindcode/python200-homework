from pathlib import Path
from dotenv import load_dotenv
import os
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex

from pathlib import Path
docs_dir = Path("assignments_06/resources/groundwork_docs")
assert docs_dir.exists(), f"Document directory not found: {docs_dir}"



env_path = Path(__file__).resolve().parents[1] / ".env"

if load_dotenv(env_path):
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")

api_key = os.getenv("OPENAI_API_KEY")

if api_key:
    print("OPENAI_API_KEY found.")
else:
    raise ValueError("OPENAI_API_KEY was not found. Check your .env file location and spelling.")



documents = SimpleDirectoryReader(str(docs_dir)).load_data()

print("\nDocuments loaded:")
print(len(documents))

print("\nFile names:")
for document in documents:
    print(document.metadata.get("file_name"))



index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine(similarity_top_k=3)

print("\nIndex built successfully. Ready to answer questions.")


questions = [
    "What are Groundwork's hours on weekends?",
    "Do you offer any dairy-free milk options?",
    "How does the loyalty program work?",
    "How did Groundwork Coffee get started?",
    "Do you offer catering or wholesale orders?",
]

for question in questions:
    print("\n" + "=" * 50)
    print("QUESTION:")
    print(question)

    response = query_engine.query(question)

    print("\nANSWER:")
    print(response)

    top_node = response.source_nodes[0]

    print("\nTOP RETRIEVED SOURCE NODE:")
    print("Document name:", top_node.node.metadata.get("file_name"))
    print("Similarity score:", round(top_node.score, 4))
    print("Chunk preview:")
    print(top_node.node.get_content()[:200])


# Step 4 Reflection:
# The assistant sounded confident and specific because it used the
# Groundwork documents as retrieved context before answering.
# The answers were easier to trust when the top retrieved source node
# clearly matched the question. This is the main advantage of RAG:
# I can inspect where the answer came from instead of only trusting
# the model's general knowledge.
#
# Some results may still surprise me if the top retrieved chunk is only
# partly related to the question. That shows why checking source nodes
# is important in a RAG system.


# --- Step 5: Find a Failure ---

failure_question = "Why the owner stop selling coffee from Mexico?"

print("\n" + "=" * 50)
print("FAILURE TEST QUESTION:")
print(failure_question)

failure_response = query_engine.query(failure_question)

print("\nFULL RESPONSE:")
print(failure_response)

print("\nALL RETRIEVED SOURCE NODES:")

for i, node in enumerate(failure_response.source_nodes, start=1):
    print("\nSource node", i)
    print("Document name:", node.node.metadata.get("file_name"))
    print("Similarity score:", round(node.score, 4))
    print("Chunk preview:")
    print(node.node.get_content()[:200])


# Step 5 Reflection 

# I asked: "Why the owner stop selling coffee from Mexico?" because I wanted to test how the system handles a question that was not included in the documents.  The system gave an answer that sounded confident but was not supported by any of the documents.  This is a problem of RAG because it can still generate an answer even when the retrieved information is not relevant or sufficient.  



# Step 6 Reflection 

# 1.The lesson built semantic RAG manually — chunking, embedding, and indexing took many lines of code. How many lines did the equivalent LlamaIndex implementation take in your project? What does that tell you about the value of using a framework?

# A semantic RAG model requieres more steps than the keyword-based RAG model because it needs to process the documents into vector embeddings, build an index, and perform similarity search. However, the semantic RAG model can retrieve more relevant information even when the wording of the query is different from the documents, which is a big advantage over keyword-based retrieval.  The LlamaIndex implementation took few lines of code to build the index and to make the queries.  The LlamaIndex allow to focus on the applciation logical instead of the technical implementation of RAG.




# 2.You have now built a system that answers questions from real documents. Describe a different use case — not a coffee shop — where this approach would add genuine value to a business or organization.

# Another use case it could be an ordering sytem for ordering candle making supplies. The system could answer questions about the different types of wax, wicks, and fragrances available, as well as provide information about shipping times and return policies. There aew many questions when ordering supplies, that will be easeier for customers if the systems can answer them quickly and accurately. This would add value to the business by improving customer satisfaction and reducing the workload on customer service representatives.  It will be necessary to upload technical documents that can answer basic questions about the products.  This could be applied to many industries, where customers have questions avbout products an or services.

# 3.What is one failure mode that RAG cannot fully prevent, even when retrieval is working correctly?

# Even if the retrieval works, the RAG system cannot fully prevent the information retrieved is not correct or relevant to the question.  So the retrieval could work loading the right document, but the answer could be vague or incorrect because the model doesn't have the logic to answer correctly.  It oculd sound confident but the answer could be wrong. 







