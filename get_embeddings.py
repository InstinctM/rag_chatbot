import os
import macros
from langchain.schema.document import Document
from langchain_ollama import OllamaEmbeddings
from transformers import GPT2TokenizerFast

def generateID():
    count = 1
    while True:
        yield count
        count += 1

def calculate_chunk_ids(chunks):
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id
        chunk.metadata["chunkname"] = chunk_id

    return chunks

def get_embeddings(chunks: list[Document]) -> dict:
    chunks = calculate_chunk_ids(chunks)
    model = OllamaEmbeddings(model=macros.EMBEDDING_MODEL)
    data = []
    getNextID = generateID()
    
    # Batch processing
    batch_size = 10  # Adjust based on memory limits
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        embeddings = model.embed_documents(texts=[chunk.page_content for chunk in batch])

        for idx, chunk in enumerate(batch):
            id = next(getNextID)
            filename = os.path.basename(chunk.metadata["chunkname"])
            content = chunk.page_content
            data.append({
                "id": id,
                "filename": filename,
                "content": content,
                "embeddings": embeddings[idx]
            })

    assert len(data) == len(chunks) 
    return data