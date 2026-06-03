#---Azure Authentication--- #Q1
# In a comment block, answer: when you run a Python script locally that uses DefaultAzureCredential, what does it rely on to authenticate? What command must you have run first, and how does DefaultAzureCredential know to use it?
## To authenticate a Python script uses the existing az login session.  The command that we must have is “az login”.   The first command to run is the az login, after that it can pick up the session automatically.


#---Azure Authentication--- #Q2
# In a comment block, answer: why can't a deployed pipeline (running on an Azure VM or container) use az login for authentication? What does it use instead, and why does the same Python code work without changes?
## When deploying a pipeline to run on an Azure VM or in a container, there is no human around to run az login, that is why it can’t use az login for authentication because it doesn’t work in automated environmnets.  Instead uses managed identities.  The same Python code that works with az login works in the cloud with managed identity.


#---Azure Authentication--- #Q3
# You run a script that creates a DefaultAzureCredential and immediately gets an AuthenticationError. In a comment block, describe the two most likely causes and how you would diagnose each.
## One cause is because az login has not run recently, sessions expire after a period of inactivity. To diagnose this, I would run az login again to refresh the session.  Another cause is that the script is running in an environment where there is no az login session available, such as a new terminal or a different user account. To diagnose this, I would check if az login has been run in the current environment and ensure that the correct user account is being used.


#---Blob Storage--- #Q1
# In a comment block, describe the three-level hierarchy of Azure Blob Storage in your own words. Give a concrete analogy that maps each level to something familiar (a filesystem, a filing cabinet, etc.).
# Storage account: top level resource
# Container: a group of blobs within the storage account
# Blob: Individual file
# An analogy could be the Storage account is the drive, the container the top directory and blob the file.


#---Blob Storage--- #Q2
# For each scenario below, write one sentence in a comment block saying whether you would use Blob Storage or a relational database (like Azure SQL), and why.
# A REST API returns a JSON payload each hour. You need to store the raw responses for reprocessing later.
# Your pipeline produces a table of 50 million customer transactions that your analytics team queries by date range and customer ID every day.
# A computer vision model produces image embeddings as NumPy arrays. You need to save them between pipeline runs.
## For the first scenario, I would use Blob Storage because is just to reprocessing the JSON file and storage not to relate with other file or table. For the second scenario, I would use a relational database like Azure SQL because it is optimized for structured data and allows for efficient querying by date range and customer ID. For the third scenario, I would use Blob Storage because it can handle large binary data like NumPy arrays and provides scalable storage for the embeddings between pipeline runs.


#---Blob Storage--- #Q3
# Write a function list_container(container_client) that prints the name and size (in bytes) of every blob in the container, one per line. The function should take a ContainerClient object as its only argument and return nothing.
def list_container(container_client):
    blob_client = container_client.get_blob_client()
    
    for blob in container_client.list_blobs():
        print(f"Name: {blob.name}, Size: {blob.size} bytes")    



#---Blob Storage--- #Q4
# Write a function upload_text(container_client, blob_name, text) that encodes a Python string as UTF-8 and uploads it as a blob, overwriting any existing blob with the same name. The function should take a ContainerClient, a blob name string, and a text string, and return nothing.
def upload_text(container_client, blob_name, text):
    blob_client = container_client.get_blob_client(blob_name)
    blob_client.upload_blob(text.encode('utf-8'), overwrite=True)

