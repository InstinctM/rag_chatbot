# Instantiate macros here
PATH = r'/Users/dev14/Documents/ai_chatbot/TestPDFs/'
CHUNK_SIZE = 500
INFERENCE_MODEL = "llama3:8b"
EMBEDDING_MODEL = "mxbai-embed-large"
COLLECTION_NAME = 'llama_test_collection'
TEMP = 0

INDEX = {
    'index_type': 'IVF_FLAT',
    'metric_type': 'L2',
    'params': {'nlist': 128},
}

SEARCH_PARAMS = {
    'metric_type': 'L2',
    'params': {'nprobe': 10},
}

QUERY_PROMPT_TEMPLATE = """
Below is a history of the conversation so far, and a new question asked by the user that needs to be answered by searching in a knowledge base on data provide.
Generate a search query based on the conversation and the new question.
Do not include cited source filenames and document names e.g sample.pdf:0:0 or sample.pdf:1:3 in the search query terms.
Do not include any text inside [] or <<>> in the search query terms.
Do not include any special characters like '+'.
If you cannot generate a search query, return just the number 0.
"""

PROMPT_TEMPLATE = '''
Answer ONLY with the facts listed in the list of sources below. 
If there isn't enough information below, say you don't know. Do not generate answers that don't use the sources below. 
If asking a clarifying question to the user would help, ask the question.
If the question is not in English, answer in the language used in the question.
Each source has a name followed by colon and the actual information, always include the source name for each fact you use in the response. 
Use square brackets to reference the source, for example [sample.pdf:0:0]. 
Don't combine sources, list each source separately, for example [sample.pdf:0:1][sample.pdf:0:2].
'''

WARM_UP_QUESTIONS = [
    "What is neural networks?", 
    "What is the purpose of the activation function in a neural network?",
    "What are the common activation functions?",
    "What is the most important component in a neural network?",
    "How do neurons in a neural network communicate with each other?"
]
