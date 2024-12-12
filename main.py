import re
import macros
import asyncio
import time
from create_collection import create_collection
from get_embeddings import get_embeddings
from query import query
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from pymilvus import connections, utility

def clean_text(chunks: list[Document]) -> list[Document]:
    for text in chunks:
        text.page_content = re.sub(r'\s+', ' ', text.page_content.replace('\n', ' ')).strip()
    return chunks

async def load_documents():
    doc_loader = PyPDFDirectoryLoader(macros.PATH)
    return doc_loader.load()

async def split_documents(document: list[Document]) -> list[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=macros.CHUNK_SIZE,
        chunk_overlap=macros.CHUNK_SIZE*0.2,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(document)

async def main():
    connections.connect('default', host='localhost', port='19530')
    if utility.has_collection(macros.COLLECTION_NAME):
        utility.drop_collection(macros.COLLECTION_NAME)
    collection = create_collection()
    
    t1 = time.time()

    docs = await load_documents()
    chunks = await split_documents(document=docs)
    chunks = clean_text(chunks=chunks)
    data = get_embeddings(chunks=chunks)

    print(f"Time taken to vectorize documents: {time.time()-t1} seconds.")
    
    collection.insert(data)
    collection.load()
    await query()

if __name__ == "__main__":
    asyncio.run(main())