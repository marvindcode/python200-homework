# --- LLMs as Transform ---Q1

# 1. I would use deterministic code to parse "Jan 5th, 2024" into "2024-01-05"
# because date is a single expected output.

# 2. Yes, I would use an LLM to classify the three options of billing, technical or general.  LLMs work well with text classification.

# 3. I would use deterministic code to calculate the average of a list of numbers, the result can get it with code, is expected a single output.

# 4. I would use an LLM to extract "Acme Corp" from "Sr. Data Eng @ Acme Corp (contract)" because  LLMs are used for field extraction.

# 5. For word counting it will be better to use deterministic code, the count should be exact.


# --- LLMs as Transform ---Q2

# The prompt "Summarize this product review in a few sentences" .  This prompt is too general, it could be long and will cause the pipeline to parse.

System = “Tell if the product is acceptable or not”.

 
# --- LLMs as Transform ---Q3

# Your dataset has 50,000 records and you need to run a classification call for each one using gpt-4o-mini. In a comment block, answer:
# 1. If each call takes 1 second on average, how long would sequential processing take?
# 50,000 records * 1 second = 50,000 seconds.
# 50,000 seconds / 60 = 833.3 minutes.
# 833.3 minutes / 60 = 13.9 hours.

# 2. What is one practical strategy to handle this more efficiently at scale, without changing models?
# One starategy is to use concurrent processing, or other will bew in batches. Thia reduces the total processing time while still respecting the rate limits of the API. Additionally, implementing retry logic for failed requests can help ensure that all records are processed successfully without manual intervention.


# --- Azure OpenAI ---Q1
# In a comment block, name two reasons an organization might use Azure OpenAI instead of calling the OpenAI API directly. Be specific -- "it's better" is not an answer.
# 1.An organization might use Azure OpenAI because the data stays insides Azure's infrastructure, which can be a requirement for compliance and security reasons.
#2. Azure OpenAI through the Microsoft's enterprise agreemement, establishes that can't be used for data training, which is a requirement for some organizations.


# --- Azure OpenAI ---Q2
# When you switch from OpenAI to AzureOpenAI, the client initialization takes three Azure-specific parameters. In a comment block, name them and describe what each one is. (Do not include the standard api_key -- describe the Azure-specific ones.)

# The three Azure-specific parameters for client initialization when switching from OpenAI to AzureOpenAI are:
#1. azure_ad_token_provider: This is a function that provides the necessary authentication tokens for accessing the Azure OpenAI service. It ensures that the client can authenticate and authorize requests to the Azure OpenAI API.
#2. azure endpoint: This is the URL of the Azure OpenAI endpoint to which the requests will be sent. It specifies the location of the Azure OpenAI service that the client will interact with.
#3. deployment_id: This is the ID of the specific model deployment within Azure OpenAI that you want to use. It identifies which model deployment the client should use for generating responses, allowing you to specify different models or versions of models that you have deployed in Azure OpenAI.


# --- Azure OpenAI ---Q3
# In a comment block, answer: when using AzureOpenAI, the model parameter in chat.completions.create() does not take a value like "gpt-4o-mini". What does it take instead, and where do you find the right value to use?
# Instead of using the model name, AzureOpenAI uses the deployment_id as the model parameter. You can find the correct deployment_id in the Azure portal under the OpenAI resource, in the "Deployments" section.