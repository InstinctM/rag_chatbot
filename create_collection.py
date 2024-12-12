from pymilvus import (connections, FieldSchema, CollectionSchema, DataType, Collection, utility)
import macros

def create_collection():
    if not connections.has_connection('default'):
        connections.connect('default', host='localhost', port='19530')

    fields = [
        FieldSchema(name='id', dtype=DataType.INT64, is_primary=True, auto_id=False), 
        FieldSchema(name='filename', dtype=DataType.VARCHAR, max_length=500),
        FieldSchema(name='content', dtype=DataType.VARCHAR, max_length=5000),
        FieldSchema(name='embeddings', dtype=DataType.FLOAT_VECTOR, dim=1024),
    ]

    schema = CollectionSchema(fields, description='Llama test collection', enable_dynamic_field=False)
    collection = Collection(macros.COLLECTION_NAME, schema, consistency_level='Strong')
    collection.create_index('embeddings', macros.INDEX)

    return collection