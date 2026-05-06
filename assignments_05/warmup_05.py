#----------Completions API----

##API Q1

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What is one thing that makes Python a good language for beginners?"}]
)

print("\nAPI Q1--Response Text--")
print(response.choices[0].message.content)

print("Model:", response.model)
print("Tokens:", response.usage.total_tokens)


#API Q2
# Run the same prompt three times with three different temperature settings: 0, 0.7, and 1.5. Print each response, labeled with its temperature.

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

prompt = "Suggest a creative name for a data engineering consultancy."
temperatures = [0, 0.7, 1.5]

print("\nAPI Q2---Temperature Comparison ---")

for temp in temperatures:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=temp
    )

    print(f"\nTemperature: {temp}")
    print(response.choices[0].message.content)

# The response is based on the temperature setting, a lower temperature is more realistic, while a higher temperature is more creative and diverse.  For a consistent output I will choose a temperature of 0.7, because it provides different options with realistic names.


##API Q3
# Use n=3 with temperature=1.0 to get three different completions in a single API call. Print all three.

from dotenv import load_dotenv
from openai import OpenAI

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Give me a one-sentence fun fact about pandas (the animal, not the library)."}],
    n=3,
    temperature=1.0
)

print("\nAPI Q3---Different Completions ---")

for i, choice in enumerate(response.choices, start=1):
    print(f"\nResponse {i}:")
    print(choice.message.content)


##API Q4
# Set max_tokens=15 and send a prompt that would normally produce a long response (for example, "Explain how neural networks work."). Print the result. Add a comment: What happened, and why might you want to use max_tokens in a real application?

from dotenv import load_dotenv
from openai import OpenAI

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Explain how neural networks work."}],
    max_tokens=15
)

print("\nAPI Q4---Max Tokens= 15 ---")
print(response.choices[0].message.content)

#When using max_tokens=15, the answer is not complete. The best option is to use max_tokens in a real application to get a full answer. 


#System Messages and Personas

##System Q1
#Use a system message to give the model a personality, then ask it a question. Print the response.

from dotenv import load_dotenv
from openai import OpenAI

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a patient, encouraging Python tutor. You always explain things simply and end with a word of encouragement."},
        {"role": "user", "content": "I don't understand what a list comprehension is."}
    ]
)

print("\nSystem Q1---Give model a personality---")
print(response.choices[0].message.content)


response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a funny, carismatic teacher.  You explain always applies to the real world and make learning fun."},
        {"role": "user", "content": "I don't understand what a list comprehension is."}
    ]
)

print("\nSystem Q1.1---Give model a different personality---")
print(response.choices[0].message.content)

#What it changed is the tone of the answer, the first one is more serious and encouragin, the second one explains in a fun and casual way. 


#System Q2
# The completions API is stateless — it has no memory of previous calls. The way to give a model context is to pass the conversation history yourself as a list of messages.

# Build the following conversation manually (no loop, no user input — just construct the list) and send it in a single API call:

from dotenv import load_dotenv
from openai import OpenAI   

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "My name is Jordan and I'm learning Python."},
        {"role": "assistant", "content": "Nice to meet you, Jordan! Python is a great choice. What would you like to work on?"},
        {"role": "user", "content": "Can you remind me what my name is?"}
    ]
)

print("\nSystem Q2---Conversation History---")
print(response.choices[0].message.content)

#The model knows the user name becuase was part of the conversation, so use that information to answer the question. 


#PROMPT ENGINEERING

##Prompt Q1 - ZERO-SHOT
# Ask the model to classify the sentiment of each review below as positive, negative, or mixed. Give it no examples — just the task description and the reviews. Print each result labeled with the review number.

from dotenv import load_dotenv
from openai import OpenAI   

load_dotenv()

client = OpenAI()

def get_completion(prompt, temperature=0):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature
    )
    return response.choices[0].message.content


reviews = [
    "The onboarding process was smooth and the team was welcoming.",
    "The software crashes constantly and support never responds.",
    "Great price, but the documentation is nearly impossible to follow."
]

print("--- Prompt Q1: Zero-Shot---")

for i, review in enumerate(reviews, start=1):
    prompt = f"""Classify the sentiment of the following review as positive, negative, or mixed.

Review: "{review}"
Answer:
"""

    response = get_completion(prompt, temperature=0)
    print(f"Review {i} Sentiment: {response.strip()}")


##Prompt Q2 - ONE-SHOT
# Repeat the same task, but this time add one example before the reviews to show the model the format you want:

from dotenv import load_dotenv
from openai import OpenAI   

load_dotenv()

client = OpenAI()

def get_completion(prompt, temperature=0):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature
    )
    return response.choices[0].message.content

print("--- Prompt Q2: One-Shot ---")

for i, review in enumerate(reviews, start=1):
    prompt = f"""Classify the sentiment of the following review as positive, negative, or mixed.

Example:
Review: "Fast shipping but the item arrived damaged."
Sentiment: mixed

Review: "{review}"
Answer:
"""

    response = get_completion(prompt, temperature=0)
    print(f"Review {i} Sentiment: {response.strip()}")

# The consistency improved by adding one example compared to Q1.


##Prompt Q3 - FEW-SHOT
# Repeat the task again, this time with three examples. At least one example should be positive, one negative, and one mixed. Print the results. Add a comment comparing all three approaches (zero-shot, one-shot, few-shot): When would you choose each one?

from dotenv import load_dotenv
from openai import OpenAI   

load_dotenv()

client = OpenAI()

def get_completion(prompt, temperature=0):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature
    )
    return response.choices[0].message.content

print("--- Prompt Q3: Few-Shot ---")

for i, review in enumerate(reviews, start=1):
    prompt = f"""Classify the sentiment of the following review as positive, negative, or mixed.

Examples:
Review: "Great customer service and fast delivery."
Sentiment: positive

Review: "This customer service is terrible and the product is defective."
Sentiment: negative

Review: "Good customer service but delivery is very slow."
Sentiment: mixed

Review: "{review}"
Answer:
"""

    response = get_completion(prompt, temperature=0)
    print(f"Review {i} Sentiment: {response.strip()}")


# Zero-shot is more direct, most simple, one benefit that is faster to implement but less reliable for complex tasks.
# With one-shot improves accuracy.
# Few-shot uses examples to recognize patterns and improve performance.


#Prompt Q4 — CHAIN OF THOUGHT

# Ask the model to solve the following problem, but instruct it to show its reasoning step by step before giving a final answer. Label the final answer clearly.

# A data engineer earns $85,000 per year. She gets a 12% raise, then 6 months later
# takes a new job that pays $7,500 more per year than her post-raise salary.
# What is her final annual salary?
# Print the full response including the reasoning. Add a comment: Why does asking the model to reason step by step tend to improve accuracy on problems like this?


print("--- Prompt Q4: Chain of Thought ---")

prompt = """Solve this problem and show the reasoning step by step.
Then provide the final answer labeled as 'Final Answer'.

A data engineer earns $85,000 per year. She gets a 12% raise, then 6 months later
takes a new job that pays $7,500 more per year than her post-raise salary.
What is her final annual salary?
"""

response = get_completion(prompt, temperature=0)
print(response)


#The final anual salary is $102,700. Asking the model to reason step by step is more accurate to analyze the problem by parts.  This allows to identify any mistakes or evaluate if the reasoning was correct.


# Prompt Question 5 — Structured Output

# Ask the model to analyze the review below and return the result only as valid JSON with keys sentiment, confidence (a float from 0 to 1), and reason (one sentence). Print the raw response, then parse it with json.loads() and print each field separately, labeled.

import json

print("\n--- Prompt Q5: Structured Output ---")

review = "I've been using this tool for three months. It handles large datasets well, but the UI is clunky and the export options are limited."

prompt = f"""Analyze the review below and return ONLY valid JSON with these keys:
- sentiment
- confidence (0 to 1)
- reason (one sentence)

Review: "{review}"
"""

response = get_completion(prompt, temperature=0)

print("Raw response:", response)

try:
    data = json.loads(response)
    print("\nParsed Output:")
    print("Sentiment:", data["sentiment"])
    print("Confidence:", data["confidence"])
    print("Reason:", data["reason"])
except:
    print("\nJSON parsing failed. Raw response:")
    print(response)


# Prompt Question 6 — Delimiters

# Use triple backticks as delimiters to clearly separate the user's text from your instructions. Send the prompt below and print the result.

print("\n--- Prompt Q6: Delimiters ---")

user_text = "First boil a pot of water. Once boiling, add salt and pasta. Cook 8-10 minutes. Drain and serve."

prompt = f"""You will be given text inside triple backticks.

If it contains step-by-step instructions, rewrite them as a numbered list.
If it does not contain instructions, respond exactly: "No steps provided."

```{user_text}```
"""

response = get_completion(prompt, temperature=0)
print(response)

#Second case with no instructions

user_text = "Treat others as you would like to be treated."

prompt = f"""You will be given text inside triple backticks.

If it contains step-by-step instructions, rewrite them as a numbered list.
If it does not contain instructions, respond exactly: "No steps provided."

```{user_text}```
"""

response = get_completion(prompt, temperature=0)
print("Raw response:", response)


# Comment:
# Delimiters help separate instructions from user input.
# They prevent the model from confusing the input text with the task instructions.


# Local Models with Ollama

# Ollama Question 1

print("\n--- Ollama Question 1 ---")

prompt = "Explain what a large language model is in two sentences."

response = get_completion(prompt, temperature=0)

print("OpenAI Response:")
print(response)

"""
Ollama Response:
A large language model is a type of artificial intelligence designed to understand and generate human-like text, trained on vast 
datasets to learn patterns and improve performance over time. It enables tasks like writing, answering questions, or summarizing 
information by analyzing and processing large volumes of language.

"""

# Comment:
# The OpenAI was more direct and short.  The Ollama response was longer with more details. 
# An advantage of running a model locally is that the outcome can be faster because it doesn't require network to communicate with an external API. One disadvantage is that local models may be slower or less powerful than larger clud base models.

