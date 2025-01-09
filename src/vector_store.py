import os
import re

from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams
from langchain_qdrant import Qdrant


def initialize_vector_store(url, grpc_port, collection_name, embed_model, dimension):
    client = QdrantClient(url, grpc_port=grpc_port, prefer_grpc=True)
    store = Qdrant(client, collection_name=collection_name, embeddings=embed_model)

    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
        )

    return client, store


def store_vectorized_info(file_content, filename, qdrant_client, embed_model, collection_name):
    source = filename.strip('.txt').capitalize()
    batch_size = 2048
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=batch_size, chunk_overlap=128)
    splits = text_splitter.split_text(file_content)
    all_splits = text_splitter.create_documents(splits)
    points = []
    identifier = 0

    for document in all_splits:
        chunk = document.page_content
        print(chunk)
        metadata = {'page_content': chunk, 'name': f'{source} Chunk {identifier}'}
        point = models.PointStruct(
            id=identifier,
            vector=embed_model.embed_documents([chunk])[0],
            payload=metadata
        )
        print(f'Processing point {identifier + 1} of {len(all_splits)}...')
        points.append(point)
        identifier += 1

    print('Storing points into the vector store...')
    qdrant_client.upsert(
        collection_name=collection_name,
        points=points
    )

    return identifier


# Chunk size in lines of the file
def intelligent_chunking_large_files(file_path, chunk_size=1):
    file_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'execution', 'to_chunk', file_path)
    chunks = []
    with open(file_path, 'r') as file:
        next(file)
        next(file)
        chunk = ''
        for line in file:
            chunk += line
            if line.strip() == '' or len(
                    chunk.split('\n')) > chunk_size:
                chunk_cleaned = re.sub(r'\| +', '|', chunk)
                chunks.append(chunk_cleaned)
                chunk = ''
        if chunk.strip() != '':  # Append remaining lines as the last chunk
            chunks.append(chunk)
        return chunks


def store_vectorized_chunks(chunks_to_save, filename, qdrant_client, embed_model, collection_name):
    source = filename.strip('.txt').capitalize()
    if filename == 'object_summary.txt':
        pattern = r'ocel:oid:\s*([^|]+)'
        meta_search = 'ocel:oid'
    elif filename == 'objects_ot_count.txt':
        pattern = r'event:\d+\s*'
        meta_search = 'event:id'
    else:
        pattern = r'ocel:timestamp:\s*([^|]+)'
        meta_search = 'ocel:timestamp'

    points = []
    identifier = 0

    for chunk in chunks_to_save:
        match = re.search(pattern, chunk)
        if match is not None:
            if filename == 'objects_ot_count.txt':
                meta_value = match.group(0).strip()
            else:
                meta_value = match.group(1).strip()
        else:
            meta_value = ''
        metadata = {'page_content': chunk, 'name': f'{source} Chunk {identifier}', meta_search: meta_value}
        point = models.PointStruct(
            id=identifier,
            vector=embed_model.embed_documents([chunk])[0],
            payload=metadata
        )
        points.append(point)
        identifier += 1
        print(f'Processing point {identifier} of {len(chunks_to_save)}...')


    print("Created points for this phase!")
    qdrant_client.upsert(
        collection_name=collection_name,
        points=points
    )

    return identifier


def intelligent_chunking_json(json_dict):
    chunks_list = []
    items = list(json_dict.items())
    chunk = ''
    for i in range(0, len(items)):
        key, value = items[i]
        chunk = chunk.join(key + ' : ' + str(value))
        chunks_list.append(chunk)
        chunk = ''

    return chunks_list


def retrieve_context(vector_index, query, num_chunks, key=None, search_filter=None):
    retrieved = vector_index.similarity_search(query, num_chunks)
    if key is not None and search_filter is not None:
        filter_ = models.Filter(
            must=[
                models.FieldCondition(
                    key=key,
                    match=models.MatchValue(value=search_filter)
                )
            ]
        )
        meta_retrieved = vector_index.similarity_search(query, filter=filter_, k=num_chunks)
        if len(meta_retrieved) > 0:
            retrieved = meta_retrieved
    retrieved_text = ''
    for i in range(len(retrieved)):
        content = retrieved[i].page_content
        retrieved_text += f'\n{i + 1}. {content}'

    return retrieved_text


def delete_qdrant_collection(q_client, q_collection_name):
    q_client.delete_collection(q_collection_name)
