import json
import os
import re
import subprocess
from typing import List, Dict, Any, Tuple

from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore
from langchain_core.embeddings import Embeddings

import utility as u

# --- Constants for Configuration ---
# This makes it easier to adjust these values without searching through the code.



def initialize_vector_store(url: str, grpc_port: int, collection_name: str, embed_model: Embeddings, dimension: int, rebuild_db: bool = False) -> Tuple[QdrantClient, QdrantVectorStore]:
    try:
        client = QdrantClient(url, grpc_port=grpc_port, prefer_grpc=True)

        collection_exists = client.collection_exists(collection_name=collection_name)
        
        if rebuild_db and collection_exists:
            print(f"Rebuilding DB: Deleting existing collection '{collection_name}'...")
            client.delete_collection(collection_name=collection_name)
            collection_exists = False

        if not collection_exists:
            print(f"Creating new collection: '{collection_name}' with dimension: {dimension}")
            client.create_collection(collection_name=collection_name,
                                     vectors_config=VectorParams(size=dimension, distance=Distance.COSINE))
        else:
            collection_info = client.get_collection(collection_name=collection_name)
            existing_dimension = collection_info.config.params.vectors.size
            
            if existing_dimension != dimension:
                print(f"Dimension mismatch! Collection '{collection_name}' has dimension {existing_dimension}, but model requires {dimension}.")
                print("Recreating collection with correct dimensions...")
                client.delete_collection(collection_name=collection_name)
                client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
                )

        store = QdrantVectorStore(client, collection_name=collection_name, embedding=embed_model)
        return client, store

    except Exception as e:
        print(f"Error initializing vector store: {e}")
        raise



def delete_qdrant_collection(client: QdrantClient, collection_name: str):
    try:
        client.delete_collection(collection_name=collection_name)
        print(f"Successfully deleted collection: {collection_name}")
    except Exception as e:
        print(f"Error deleting collection {collection_name}: {e}")


def rebuild_and_populate_vector_db(base_path, q_client, embed_model, collection_name, batch_size=100, chunk_size=2048, chunk_overlap=128):
    try:
        preprocessing_path = os.path.join(base_path, 'src', 'preprocessing.py')
        subprocess.run(['python3', preprocessing_path], check=True)
        print("Building and populating the vector collection... (1/3)")
        execution_path = os.path.join(base_path, 'data', 'execution')
        files = os.listdir(execution_path)
        general_info = []
        for f in files:
            if f.endswith('.txt'):
                try:
                    content = u.load_process_representation(f)
                    general_info.append((f, content))
                except Exception as file_error:
                    print(f"Error loading file {f}: {str(file_error)}")
                    continue
        actual_id = store_vectorized_info(general_info, q_client, embed_model, collection_name, batch_size=batch_size, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        print("Populating the vector collection... (2/3)")
        to_chunk_path = os.path.join(execution_path, 'to_chunk')
        if os.path.exists(to_chunk_path):
            files_to_chunk = os.listdir(to_chunk_path)
            jsonfile = 'objects_ot_count.txt'
            for f in files_to_chunk:
                try:
                    full_file_path = os.path.join(to_chunk_path, f)
                    if f.endswith('.txt') and f != jsonfile:
                        chunks_to_store = intelligent_chunking_large_files(full_file_path)
                        print(f'Chunking completed for {f}')
                        actual_id = store_vectorized_chunks(chunks_to_store, f, q_client, embed_model, collection_name, actual_id, batch_size=batch_size)
                    elif f == jsonfile:
                        with open(full_file_path, 'r') as jsonf:
                            print("Populating the vector collection... (3/3)")
                            j_dict = json.load(jsonf)
                            j_chunks = intelligent_chunking_json(j_dict)
                            actual_id = store_vectorized_chunks(j_chunks, f, q_client, embed_model, collection_name, actual_id, batch_size=batch_size)
                except Exception as chunk_error:
                    print(f"Error processing file {f}: {str(chunk_error)}")
                    continue
        print("Vector collection successfully created and initialized!")
    except Exception as e:
        print(f"Error populating vector database: {str(e)}")
        raise


def _batch_upsert_points(
    qdrant_client: QdrantClient,
    collection_name: str,
    embed_model: Embeddings,
    metadata_list: List[Dict[str, Any]],
    start_id: int = 0,
    batch_size: int = 100
) -> int:
    """
    A helper function to create embeddings, construct points, and upsert them in batches.

    Args:
        qdrant_client: The Qdrant client.
        collection_name: The name of the collection.
        embed_model: The embedding model.
        metadata_list: A list of metadata dictionaries. Each dict must contain 'page_content'.
        start_id: The starting ID for the points.
        batch_size: The number of points to upsert in each batch.

    Returns:
        The total number of points processed.
    """
    points = []
    current_id = start_id
    total_processed = 0

    for i, metadata in enumerate(metadata_list):
        chunk = metadata.get('page_content')
        if not chunk:
            print(f"Warning: Skipping metadata at index {i} due to missing 'page_content'.")
            continue

        try:
            # Generate embedding for the chunk
            embedding = embed_model.embed_documents([chunk])[0]

            # Create the point structure
            point = models.PointStruct(
                id=current_id,
                vector=embedding,
                payload=metadata
            )
            points.append(point)
            current_id += 1
            total_processed += 1

            # Upsert in batches
            if len(points) >= batch_size:
                qdrant_client.upsert(collection_name=collection_name, points=points)
                print(f"Stored batch of {len(points)} points.")
                points = []

        except Exception as e:
            print(f"Error processing point ID {current_id}: {e}")
            # Ensure ID increments even if an error occurs to avoid ID conflicts
            current_id += 1
            continue
    
    # Store any remaining points
    if points:
        qdrant_client.upsert(collection_name=collection_name, points=points)
        print(f"Stored final batch of {len(points)} points.")

    return total_processed


def store_vectorized_info(file_content_list: List[Tuple[str, str]], qdrant_client: QdrantClient, embed_model: Embeddings, collection_name: str, batch_size=100, chunk_size=2048, chunk_overlap=128) -> int:
    """
    Processes and stores vectorized information from a list of file contents.
    
    This function now uses the `_batch_upsert_points` helper for cleaner logic.
    """
    total_points_stored = 0
    
    try:
        for filename, file_content in file_content_list:
            try:
                print(f"Processing file: {filename}...")
                source = filename.strip('.txt').capitalize()
                
                # Split text into manageable chunks
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                splits = text_splitter.split_text(file_content)
                
                # Prepare metadata for each chunk
                metadata_list = [
                    {'page_content': chunk, 'name': f'{source} Chunk {total_points_stored + i}'}
                    for i, chunk in enumerate(splits)
                ]
                
                # Use the helper function to handle embedding and batch upserting
                points_stored = _batch_upsert_points(
                    qdrant_client,
                    collection_name,
                    embed_model,
                    metadata_list,
                    start_id=total_points_stored,
                    batch_size=batch_size
                )
                total_points_stored += points_stored
                print(f"Finished processing file: {filename}. Stored {points_stored} points.")

            except Exception as file_error:
                print(f"Error processing file {filename}: {file_error}")
                continue
                
    except Exception as e:
        print(f"An unexpected error occurred in store_vectorized_info: {e}")
        raise

    return total_points_stored


def intelligent_chunking_large_files(file_path: str, lines_per_chunk: int = 1, lines_to_skip: int = 2) -> List[str]:
    """
    Intelligently chunks a large file by grouping lines.

    Args:
        file_path: The full path to the file to chunk.
        lines_per_chunk: The number of non-empty lines to include in a chunk.
        lines_to_skip: The number of header lines to skip at the beginning of the file.

    Returns:
        A list of string chunks.
    """
    try:
        chunks = []
        with open(file_path, 'r', encoding='utf-8') as file:
            # Skip header lines
            for _ in range(lines_to_skip):
                next(file)

            chunk = ''
            for line in file:
                chunk += line
                # Create a new chunk after a blank line or if chunk size is exceeded
                if line.strip() == '' or len(chunk.split('\n')) > lines_per_chunk:
                    chunk_cleaned = re.sub(r'\| +', '|', chunk)
                    chunks.append(chunk_cleaned)
                    chunk = ''
            
            # Add the last chunk if it's not empty
            if chunk.strip() != '':
                chunks.append(re.sub(r'\| +', '|', chunk))
        
        return chunks
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return []
    except Exception as e:
        print(f"Error chunking file {file_path}: {e}")
        return []


def store_vectorized_chunks(chunks_to_save: List[str], filename: str, qdrant_client: QdrantClient, embed_model: Embeddings, collection_name: str, actual_identifier: int, batch_size=100) -> int:
    """
    Creates metadata and stores vectorized chunks using the helper function.
    """
    try:
        source = filename.strip('.txt').capitalize()
        # Define patterns to extract specific metadata from chunks
        patterns = {
            'object_summary.txt': (r'ocel:oid:\s*([^|]+)', 'ocel_oid'),
            'objects_ot_count.txt': (r'event:\d+\s*', 'event_id'),
        }
        default_pattern = (r'ocel:timestamp:\s*([^|]+)', 'ocel_timestamp')

        pattern, meta_key = patterns.get(filename, default_pattern)

        metadata_list = []
        for i, chunk in enumerate(chunks_to_save):
            meta_value = ''
            match = re.search(pattern, chunk)
            if match:
                meta_value = match.group(1).strip() if len(match.groups()) > 0 else match.group(0).strip()
            
            metadata = {
                'page_content': chunk,
                'name': f'{source} Chunk {actual_identifier + i + 1}',
                meta_key: meta_value
            }
            metadata_list.append(metadata)

        print(f"Prepared {len(metadata_list)} points for this phase.")
        
        # Use the helper function for embedding and upserting
        points_stored = _batch_upsert_points(
            qdrant_client,
            collection_name,
            embed_model,
            metadata_list,
            start_id=actual_identifier,
            batch_size=batch_size
        )

        return actual_identifier + points_stored
        
    except Exception as e:
        print(f"Error in store_vectorized_chunks: {e}")
        return actual_identifier


def intelligent_chunking_json(json_dict: Dict[str, Any]) -> List[str]:
    """
    Converts a dictionary into a list of "key : value" string chunks.

    Args:
        json_dict: The dictionary to chunk.

    Returns:
        A list of string chunks.
    """
    try:
        return [f"{key} : {str(value)}" for key, value in json_dict.items()]
    except Exception as e:
        print(f"Error chunking JSON: {e}")
        return []


def retrieve_context(vector_index: QdrantVectorStore, query: str, num_chunks: int, key: str = None, search_filter: Any = None) -> str:
    """
    Retrieve context from the vector index, with optional metadata filtering.

    Args:
        vector_index: The vector store to search.
        query: The query string.
        num_chunks: The number of chunks to retrieve.
        key: The metadata key to filter on.
        search_filter: The value to match for the filter.

    Returns:
        A formatted string of the retrieved context.
    """
    try:
        # If a key and filter are provided, use them for the search
        if key and search_filter is not None:
            qdrant_filter = models.Filter(
                must=[models.FieldCondition(key=key, match=models.MatchValue(value=search_filter))]
            )
            retrieved_docs = vector_index.similarity_search(query, k=num_chunks, filter=qdrant_filter)
        else:
            # Otherwise, perform a standard similarity search
            retrieved_docs = vector_index.similarity_search(query, k=num_chunks)

        # Format the retrieved documents into a single string
        retrieved_text = '\n'.join([f'{i + 1}. {doc.page_content}' for i, doc in enumerate(retrieved_docs)])
        
        return retrieved_text if retrieved_text else "No context found for the given query."
        
    except Exception as e:
        print(f"Error retrieving context: {e}")
        return "Context retrieval failed due to an error."


def store_traces(traces: List[str], qdrant_client: QdrantClient, log_name: str, embed_model: Embeddings, collection_name: str) -> None:
    """
    Stores log traces in the vector database using the helper function.
    """
    try:
        print(f"Preparing to store {len(traces)} traces for log: {log_name}")
        metadata_list = [
            {'page_content': trace, 'name': f'{log_name} Trace {i}'}
            for i, trace in enumerate(traces)
        ]
        
        _batch_upsert_points(
            qdrant_client,
            collection_name,
            embed_model,
            metadata_list,
            start_id=0  # Assuming traces are stored in a new or separate context
        )
        print("Successfully stored all traces.")
            
    except Exception as e:
        print(f"Error in store_traces: {e}")
        raise