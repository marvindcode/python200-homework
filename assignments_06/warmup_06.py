# Part 1: Warmup Exercises



from dotenv import load_dotenv
import os

if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")


# RAG CONCEPTS

## Concepts Q1

# Three teams at a software company are each building a different AI project. Add a comment block to your code that identifies the best approach — prompt engineering, fine-tuning, or RAG — for each scenario, and gives a 1-2 sentence explanation of your reasoning.
# Scenario A: A legal team wants an assistant that can answer questions about their internal policy library — hundreds of PDFs that are updated every quarter.
# Scenario B: A startup wants their model to write product copy in a very specific brand voice — a dry, minimalist style that does not appear much online. They have 3,000 examples their in-house writers produced over the years.
# Scenario C: A data analyst needs to ask an LLM questions about a single two-page report she just received. She does not need this to work for any other document.

# Scenario A: RAG (Retrieval-Augmented Generation) is the best approach because it can retrive updated information from a large collection of documents.  For large amounts of data needed to access quickly and accurately, RAG is more efficient than fine-tuning or prompt engineering. 
# Scenario B: Fine-tuning is the best approach because the startup wants their model to generate text in a specific style, and the fine-tuning process let the model learn the tone and writing style. 
# They have a sufficient amount of training data.
# Scenario C: Prompt engineering is the best approach because the data analyst only needs one report, and can put it directly into the prompt to ask questions about a single document.


## Concepts Q2
# AI hallucinations (responses that sound confident but are wrong) can be particularly difficult to detect. Add a comment to your code answering this:
# Why is a confidently wrong answer more harmful than one that says "I am not sure"? Give one example of a real situation where a confident hallucination could cause harm.
# Think about the tone of the response as well as its content — why does the way the model expresses an answer affect how much we trust it?

#A confidently wrong answer is more harmful because the user can assume it's correct, that's the problem with hallucinations. If the user doesn't know much about a theme, they might trust the model's response without question, leading to incorrect decisions or actions based on the false information.
#An example could apply to different areas, for instance, in education if a student receives a confident but incorrect answer from an AI tutor, the student will get confused and not understand the material.
#It affects because the tone of the response it could be very convincing, making it harder for the user to distinguish between correct and incorrect information. The problem is that the student will stop trusting the AI if they receive a lot of incorrect information.


## Concepts Q3
# The steps below make up a complete RAG pipeline, but they are out of order. Copy the list into your code as a comment, arrange them in the correct order, and add a one-sentence description of what happens at each step.
steps = [
    "Generate a response from the LLM",
    "Extract text from source documents",
    "Receive the user's query",
    "Retrieve the most relevant chunks",
    "Convert text chunks into embeddings",
    "Inject retrieved chunks into the prompt",
    "Split text into chunks",
    "Embed the user's query",
]
# 1. Receive the user's query: The user asks a question or provides a request.
# 2. Embed the user's query: The query is converted into a numerical representation that can be used to find similar queries.
# 3. Retrieve the most relevant chunks: the system searches for the most relevant text chunks based on the embedded query.
# 4. Extract text from source documents: The text is extracted from the source documents.
# 5. Split text into chunks: The text is split into smaller, manageable pieces.
# 6. Convert text chunks into embeddings: The text chunks are converted into numerical representations.
# 7. Inject retrieved chunks into the prompt: The relevant text chunks are added to the prompt to provide context for the LLM.
# 8. Generate a response from the LLM: The LLM generates a response based on the prompt that includes the retrieved text chunks.


#KEYWORD RAG

import string

def simple_keyword_retrieval(query, documents, verbose=True):
    """Keyword retrieval using token overlap scoring."""
    stopwords = {
        "a", "an", "the", "and", "or", "in", "on", "of", "for", "to", "is",
        "are", "was", "were", "by", "with", "at", "from", "that", "this",
        "as", "be", "it", "its", "their", "they", "we", "you", "our"
    }
    translator = str.maketrans("", "", string.punctuation)

    query_words = {
        w.translate(translator)
        for w in query.lower().split()
        if w not in stopwords
    }
    if verbose:
        print(f"\nQuery tokens (filtered): {sorted(query_words)}")

    scores = []
    for name, content in documents.items():
        content_words = {
            w.translate(translator)
            for w in content.lower().split()
            if w not in stopwords
        }
        overlap = query_words & content_words
        score = len(overlap)
        scores.append((score, name, content))
        if verbose:
            print(f"[{name}] overlap={score} -> {sorted(overlap)}")

    scores.sort(reverse=True)
    best = next(((name, content) for score, name, content in scores if score > 0), None)
    if best:
        if verbose:
            print(f"\nSelected best match: {best[0]}")
        return [best]
    else:
        if verbose:
            print("\nNo overlapping keywords found.")
        return [("None found", "No relevant content.")]


#KEYWORD Q1


query = "What are your hours on the weekend?"

documents = {
    "menu.txt": "We serve espresso, lattes, cappuccinos, and cold brew. Pastries include croissants and muffins baked fresh daily. Oat milk and almond milk are available.",

    "hours.txt": "We are open Monday through Friday from 7am to 7pm. On weekends we open at 8am and close at 5pm. We are closed on Thanksgiving and Christmas Day.",

    "hiring.txt": "We are currently hiring baristas and shift supervisors. Send your resume to jobs@groundworkcoffee.com.",

    "loyalty.txt": "Join our loyalty program to earn one point per dollar spent. Redeem 100 points for a free drink of your choice.",
}

results = simple_keyword_retrieval(query, documents, verbose=True)

print("\nSelected document:")
print(results[0][0])

# The selected document was loyalty.txt, is not the best answer because it only has one keyword that matches with the query, which is "your". The best answer should be hours.txt because it contains more relevant keywords like "hours" and "weekend". This shows the limitation of keyword based retrieval, as it can miss relevant documants if they don't have enough overlapping keywords, even if they are more relevant to the user's query. 


#KEYWORD Q2

query = "Do you have anything without caffeine?"

results = simple_keyword_retrieval(query, documents, verbose=True)

print("\nSelected document:")
print(results[0][0])


#WHICH DOCUMENT WAS SELECTED? No one because none of the documents mention "caffeine".

#WHETHER KEYWORD RAG GOT THIS RIGHT? No, it did not. The selected document may not be correct because none of the documents directly mention "caffeine."

#WHAT KIND OF RETRIEVAL WOULD DO BETTER HERE? Semantic search or contextual retrieval would do better here.



#KEYWORD Q3

# Before running any code, predict which document will be selected for the query below. Write your prediction and your reasoning as a comment first, then run the code to check.

query = "How do I sign up for rewards?"

# Was your prediction correct? If the result surprised you, add a comment explaining what happened.

results = simple_keyword_retrieval(query, documents, verbose=True)

print("\nSelected document:")
print(results[0][0])

# None of the documents were selected,  Any of the words was overlaped because didn't match any of the words.  The selected document could be loyalty.txt, from the stands pooint that theme is related with the keywords "sign up" and "rewards". This shows that keyword based retrieval struggles with queries that use different wording than the documents, even if they are semantically related.


#Semantic RAG Concepts

##Semantic Question 1

#1.A vector embeding represents text converted into numbers to convert into vectors.  

#2.The cosaine method compares two text chunks with values between -1 and 1, where a value of 1 indicates identical meaning and 0 indicates no similarity. The chunk with cosine similarity 0.85 is more relevant because it is much closer in meaning to the query than the chunk with 0.30.

#3. Semantic search find relevant chunks text even without exact words because embeddings have meaning, not just matching vocabulary.


##Semantic Question 2


# | Feature                 | Keyword RAG                    | Semantic RAG                     |
# |-------------------------|--------------------------------|--------------------------------- |
# | What is compared?       | Exact word overlap             | Vector embeddings                |
# | What is retrieved?      | Full document                  | Relrvant Chunks                           |
# | Can it handle synonyms? | No                             | Yes                              |
# | Storage format          | Plain text dictionary          | Vector database, index          |
# | Relevance score         | Overlapping keywords           | Cosine similarity                |



# LlamaIndex 

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex


from pathlib import Path

brightleaf_dir = Path("resources/brightleaf_pdfs")

docs = SimpleDirectoryReader(str(brightleaf_dir)).load_data()

index = VectorStoreIndex.from_documents(docs)

query_engine = index.as_query_engine(similarity_top_k=3)


##LlamaIndex Question 1

# Build an in-memory LlamaIndex pipeline using the Brightleaf Solar PDFs and run the two queries below. For each query, print:
# •	The question
# •	The answer from the model
# •	For each of the 3 retrieved source nodes: the similarity score and the first 150 characters of the chunk text
questions = [
    "What employee benefits does BrightLeaf offer?",
    "What are BrightLeaf's security policies?",
]
# Use similarity_top_k=3. After printing the results, add a comment for each query answering:
# •	Do the retrieved chunks look relevant to the question?
# •	Does the model's response sound confident and specific, or does it hedge with phrases like "based on the context" or "I'm not sure"? Note what you observe about the tone.
# •	Did anything unexpected get retrieved?


for q in questions:

    print("\nQUESTION:")
    print(q)

    response = query_engine.query(q)

    print("\nANSWER:")
    print(response)

    print("\nSOURCE NODES:")

    for node in response.source_nodes:
        print("\nSimilarity Score:")
        print(round(node.score, 4))

        print("Chunk Preview:")
        print(node.node.get_content()[:150])


# Do the retrieved chunks look relevant to the question? yes, the retrieved chunks look relevant to the question, one chunck is 0.7816 and the other 0.7801.  When retrieving chunks, the system found relevant information about the employee benefits and security procedures.
	
#Does the model's response sound confident and specific, or does it hedge with phrases like "based on the context" or "I'm not sure"? Note what you observe about the tone.  The answers were specific in the model which makes the model confident. The tone of the response is direct and informative.  The answer make sense, it mentioned the employee benefits.  The model didn't anser with different information.

#Did anything unexpected get retrieved? no, nothing unexpected was retrieved. The retrieved chunks were important to the questions asked. 


##LlamaIndex Question 2
query = "What employee benefits does BrightLeaf offer?"

print("\n--- similarity_top_k = 1 ---")

qe1 = index.as_query_engine(similarity_top_k=1)

response1 = qe1.query(query)

print(response1)

for node in response1.source_nodes:
    print(round(node.score, 4))


print("\n--- similarity_top_k = 5 ---")

qe5 = index.as_query_engine(similarity_top_k=5)

response5 = qe5.query(query)

print(response5)

for node in response5.source_nodes:
    print(round(node.score, 4))


# The response changed as increasing similarity_top_k because gives more context to the model. The more retrieved chunks, the more information the model has to generate a response but more context is not always better because unrelated chunks can be included. 


##LlamaIndex Question 3

query = "Does the BrightLeaf offer life insurance?"

response = query_engine.query(query)

print("\nQUESTION:")
print(query)

print("\nANSWER:")
print(response)

for node in response.source_nodes:
    print("\nScore:")
    print(round(node.score, 4))

    print(node.node.get_content()[:150])


# I expected the system to struggle because the question is vague and the documents may not directly mention "life insurance." The system might retrieve chunks about employee benefits in general, but it may not find specific information about life insurance.

# To improve this system, I would add more documents that mentions employee benefits in detail, including life insurance. 


##LlamaIndex Question 4

from llama_index.llms.openai import OpenAI
from llama_index.core.evaluation import (
    FaithfulnessEvaluator,
    RelevancyEvaluator,
)

judge_llm = OpenAI(
    model="gpt-4o-mini",
    temperature=0.2
)

faithfulness_evaluator = FaithfulnessEvaluator(
    llm=judge_llm
)

relevancy_evaluator = RelevancyEvaluator(
    llm=judge_llm
)

#Query1
q = "What employee benefits does BrightLeaf offer?"

response = query_engine.query(q)

faithfulness = faithfulness_evaluator.evaluate_response(
    query=q,
    response=response
)

relevancy = relevancy_evaluator.evaluate_response(
    query=q,
    response=response
)

print("\nFaithfulness Score:")
print(faithfulness.score)

print("\nRelevancy Score:")
print(relevancy.score)

#Query2
low_query = "Does Brightleaf has coffee?"

low_response = query_engine.query(low_query)

faithfulness2 = faithfulness_evaluator.evaluate_response(
    query=low_query,
    response=low_response
)

relevancy2 = relevancy_evaluator.evaluate_response(
    query=low_query,
    response=low_response
)

print("\nSecond Query Faithfulness:")
print(faithfulness2.score)

print("\nSecond Query Relevancy:")
print(relevancy2.score)

# What does a faithfulness score of 1.0 mean? What would a score of 0.0 indicate?  A faithfulness score of 1.0 means that the model's response is completely faithful to the source documents, meaning all information in the response can be directly supported by the retrieved chunks. A score of 0.0 would indicate that the response is not faithful at all, meaning it contains information that cannot be found in the retrieved chunks and may be hallucinated or fabricated by the model.
# What does a relevancy score measure, and how is it different from faithfulness?Relevancy measures wheter the answer is actually related to the user's question.
# Did the scores change between your two queries? If so, why do you think that happened? The faithfulness didn't change, was the same for both queries, but the relevancy score changed on the second query relevance to 1 because the question is not related to the documents, so the model's response is not relevant to the query.
# What is the "LLM-as-a-judge" approach, and why is it used for RAG evaluation instead of a simple accuracy metric? The "LLM-as-a-judge" approach involves using a large language model (LLM) to evaluate the quality of responses generated by another LLM in a RAG pipeline. This method is used for RAG evaluation because it can judge the quality of the response in more detailed way than an accuracy metric, which may not capture the faithfulness and relevancy. The LLM can provide a more comprehensive assessment of the response's quality, including whether it is factually correct (faithful) and whether it addresses the user's query (relevant).


