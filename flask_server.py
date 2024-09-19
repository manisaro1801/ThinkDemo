import json
import copy
import os
import sys

from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS

from gevent.pywsgi import WSGIServer
from RAG import retrieve_data_from_db ##function defined in RAG python code
from RAG import ingestdata            ##function defined in RAG to ingest data

#from watsonx_integration_service import retrieve_data_from_db

app = Flask(__name__)
CORS(app)


resp_data = {
  	'type': 'agent',
  	'sections': [
    		{
      			'type': 'text',
      			'data': 'I have found the following transactions based on your request.'
    		},
    		{
      			'type': 'table',
      			'data': [
      			]
    		}
  	]
    }

# GET routines
@app.route('/query', methods=['GET'])
def fetchResponse():
    resp_data_local = copy.deepcopy(resp_data);
    query = request.args.get('query')
    print(query)
    data = retrieve_data_from_db(query,vectorstore_var) ## This function fetches data from vector database
    return jsonify(data)

# GET routines
@app.route('/ingestdata', methods=['GET'])
def ingest_data():
    vectorstore_var=ingestdata()


if __name__ == '__main__':
   print(f'Starting API server on port 5001')
   http_server = WSGIServer(('0.0.0.0', 5001), app)
   http_server.serve_forever() 
