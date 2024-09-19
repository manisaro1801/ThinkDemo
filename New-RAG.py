def ingestdata():

    MILVUS_HOST="milvus-service"
    MILVUS_PORT="19530"


    ##Clean up before starting :)

    from pymilvus import connections, utility

    # Connect to Milvus Database
    connections.connect(host="milvus-service", port="19530")

    colls = utility.list_collections()
    print(colls)

    for coll in colls:
        utility.drop_collection(coll)

    # Phase 1: Ingesting data & build up knowledge base
    # ![image](https://github.com/mgiessing/watsonx-rag/blob/main/images/Ingest_Data.png?raw=true)

    
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.vectorstores import Milvus
    from langchain.document_loaders import WebBaseLoader, PyPDFLoader
    from langchain.text_splitter import CharacterTextSplitter


    loader = PyPDFLoader("https://ibm.box.com/s/ohmsko4a2rhj5to7szje8uy55kldfdgb")   ## Put the URL of box 

    docs = loader.load()

    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=768, chunk_overlap=0)
    docs = text_splitter.split_documents(docs)
    print(len(docs))
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vector_store = Milvus.from_documents(
        docs,
        embedding=embeddings,
        collection_name="demo",
        connection_args={"host": MILVUS_HOST, "port": MILVUS_PORT}
    )

    print(utility.list_collections())


def retrieve_data_from_db(question,vector_store):

    LLAMA_HOST="llama-service"
    LLAMA_PORT="8080"
	
    docs = vector_store.similarity_search_with_score(question, k=3)

    print(len(docs))

    def build_prompt(question, topn_chunks: list[str]):
        prompt = "Instructions: Compose a single, short sentence that only answers the query, using the provided search results."\
                "If the search results do not mention anything say 'Found nothing.'\n\n"
  
        prompt += "Search results:\n"
        for chunk in topn_chunks:
            prompt += f"[Page {chunk[0].metadata['page']}]: " + chunk[0].page_content.replace("\n", " ") + "\n\n"

        prompt += f"Query: {question}\n\nAnswer: "

        return prompt

    prompt = build_prompt(question, docs)
    print(prompt)


# ### 3a) Completion

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

    import httpx
    import json

    json_data = {
        'prompt': prompt,
        'temperature': 0.1,
        'n_predict': 100,
        'stream': True,
    }




