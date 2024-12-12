from pymilvus import Collection, connections
import time
import macros
import json
from get_embeddings import get_embeddings
from langchain_ollama import OllamaLLM, OllamaEmbeddings

past_message = []

def nonewlines(s: str) -> str:
    return s.replace("\n", " ").replace("\r", " ")

def query():
    model = OllamaLLM(model=macros.INFERENCE_MODEL, temperature=macros.TEMP)
    embedding_model = OllamaEmbeddings(model=macros.EMBEDDING_MODEL)

    start = time.time()
    for question in macros.WARM_UP_QUESTIONS:
        prompt = f"Instructions: {macros.PROMPT_TEMPLATE} \nQuestion: {question}"
        response = model.invoke(prompt)
    print(f"Warm up question asked in {time.time()-start} seconds.")

    try:
        connections.connect('default', host='localhost', port='19530')
        collection = Collection(macros.COLLECTION_NAME)
        collection.load()
    except Exception as e:
        print(e)

    while True:
        query = "What is a neural network?"
        # query = input("What is your question? ")
        inf_start = time.time()

        # Keyword search
        # keyword_search = f"Instructions: {macros.QUERY_PROMPT_TEMPLATE} \nQuestion: {query}"
        # refined_prompt = model.invoke(keyword_search)
        # print(refined_prompt)

        # query_embeddings = embedding_model.embed_query(refined_prompt)
        # results = collection.search(data=[query_embeddings], 
        #                             anns_field="embeddings", 
        #                             param=macros.SEARCH_PARAMS, 
        #                             limit=3,
        #                             output_fields=["content", "filename"])
        # results = [doc for docs in results for doc in docs] # Flatten the array by 1 dimension
        # results = [f"{doc.entity.get('filename')}: {doc.entity.get('content')}" for doc in results]
        # content = '\n'.join(results)
        # prompt = (
        #     f"Instructions: {macros.PROMPT_TEMPLATE}\n"
        #     f"Past Messages: {None if not past_message else past_message}\n"
        #     f"Sources: {nonewlines(content)}\n"
        #     f"Question: {query}"
        # )

        ans = model.invoke(query)
        print(f"Inference finish in {time.time()-inf_start} seconds.")

        past_message.append({"role": "user", "content": query})
        past_message.append({"role": macros.INFERENCE_MODEL, "content": ans})

        with open("log.txt", 'w+') as file:
            json.dump(past_message, file, indent=4)

        break

if __name__ == "__main__":
    query()