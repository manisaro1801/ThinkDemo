#!/usr/bin/env python
# coding: utf-8

# # TechXchange Barcelona
# 
# ## Lab 2084 - Deploying Large Language Models (LLMs) on OpenShift for IBM Power
# 
# Author: Marvin Gießing (marving@de.ibm.com)

# ## Respond to natural language questions using RAG approach
# 
# This notebook contains the steps and code to demonstrate support of Retrieval Augumented Generation using a local model deployed on Power10. It introduces commands for data retrieval, knowledge base building & querying, and model testing.
# 
# Some familiarity with Python is helpful. This notebook uses Python 3.10.
# 
# #### About Retrieval Augmented Generation
# Retrieval Augmented Generation (RAG) is a versatile pattern that can unlock a number of use cases requiring factual recall of information, such as querying a knowledge base in natural language.
# 
# In its simplest form, RAG requires 3 steps:
# 
# - Phase 1: Index knowledge base passages (once)
# - Phase 2: Retrieve relevant passage(s) from knowledge base (for every user query)
# - Phase 3: Generate a response by feeding retrieved passage into a large language model (for every user query)

# <a id="setup"></a>
# ## Setup environment and import relevant libraries
# 
# As one of the main components will be a document file (we use a PDF) the main imports are pypdf to parse that and pymilvus to set up the knowledge base.

# In[1]:


MILVUS_HOST="milvus-service"
MILVUS_PORT="19530"

LLAMA_HOST="llama-service"
LLAMA_PORT="8080"


# In[2]:


##Clean up before starting :)

from pymilvus import connections, utility

# Connect to Milvus Database
connections.connect(host="milvus-service", port="19530")

colls = utility.list_collections()
print(colls)

for coll in colls:
    utility.drop_collection(coll)


# ## Phase 1: Ingesting data & build up knowledge base
# ![image](https://github.com/mgiessing/watsonx-rag/blob/main/images/Ingest_Data.png?raw=true)

# In[3]:


# Download the sample PDF file
import requests
import os
FNAME = "HarryPotter.pdf"

if not os.path.exists(FNAME):
    res = requests.get('https://ibm.box.com/shared/static/d5rfawbu2tvny6zkh1o96u8797qimwmv.pdf')
    with open(FNAME, 'wb') as file:
        file.write(res.content)


# In[4]:


from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Milvus
from langchain.document_loaders import WebBaseLoader, PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter

#loader = WebBaseLoader([
    #"https://www.redbooks.ibm.com/redpapers/pdfs/redp5612.pdf",
#])

loader = PyPDFLoader("HarryPotter.pdf")

docs = loader.load()

text_splitter = CharacterTextSplitter(separator="\n", chunk_size=768, chunk_overlap=0)
docs = text_splitter.split_documents(docs)
len(docs)


# In[5]:


#docs


# In[6]:


embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vector_store = Milvus.from_documents(
    docs,
    embedding=embeddings,
    collection_name="demo",
    connection_args={"host": MILVUS_HOST, "port": MILVUS_PORT}
)

utility.list_collections()


# ## Phase 2: Retrieve relevant passage(s) from Knowledge Base
# ![image](https://github.com/mgiessing/watsonx-rag/blob/main/images/Retrieve_Data.png?raw=true)

# In[23]:


questions = ["What was the job of Mr. Dursley?", "What does Mr. Dursley look like?", "Where does the Dursley family live?"]
question = questions[0] # "What was the job of Mr. Dursley?"

docs = vector_store.similarity_search_with_score(question, k=3)

len(docs)


# ## Phase 3: Build prompt, pass to LLM & generate Response
# ![image](https://github.com/mgiessing/watsonx-rag/blob/main/images/Generate_Response.png?raw=true)

# In[24]:


def build_prompt(question, topn_chunks: list[str]):
    prompt = "Instructions: Compose a single, short sentence that only answers the query, using the provided search results."\
             "If the search results do not mention anything say 'Found nothing.'\n\n"
  
    prompt += "Search results:\n"
    for chunk in topn_chunks:
        prompt += f"[Page {chunk[0].metadata['page']}]: " + chunk[0].page_content.replace("\n", " ") + "\n\n"

    prompt += f"Query: {question}\n\nAnswer: "

    return prompt


# In[25]:


prompt = build_prompt(question, docs)
print(prompt)


# ### 3a) Completion

# In[22]:


import requests

json_data = {
    'prompt': prompt,
    'temperature': 0.1,
    'n_predict': 100,
    'stream': False,
}

res = requests.post(f'http://{LLAMA_HOST}:{LLAMA_PORT}/completion', json=json_data)

res.json()['content']


# ### 3b) Streaming

# In[11]:


import httpx
import json

json_data = {
    'prompt': prompt,
    'temperature': 0.1,
    'n_predict': 100,
    'stream': True,
}

client = httpx.AsyncClient(timeout=30) #set higher timeout, because long prompt evaluation might take longer
lastChunks = ""
async with client.stream('POST', f'http://{LLAMA_HOST}:{LLAMA_PORT}/completion', json=json_data) as response:
    async for chunk in response.aiter_bytes():
        try:
            data = json.loads(chunk.decode('utf-8')[6:])
        except:
            pass
        if data['stop'] is False:
            print(data['content'], end="")
        else:
            print('\n\n')
            print(data['timings'])


# ## (Optional) exercises if you finish early :)
# 
# In order to get better results you have a few options to try out:
# - Experiment with the parameters (e.g. temperature, top-k, top-p, n_predict )
# - Experiment with the prompt/instruction
# - Try out a different model (make sure to use a pre-converted `.gguf` model)
# - Load your own PDF if you like something more domain/business specific than Harry Potter :)

# 
