from argparse import ArgumentParser
import json
from dotenv import load_dotenv
import os
import subprocess
import torch
import warnings

import pipeline as p
import utility as u
import vector_store as vs


DEVICE = f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
load_dotenv()
HF_AUTH = os.getenv('HF_TOKEN')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
URL = os.getenv('QDRANT_URL')
GRPC_PORT = int(os.getenv('QDRANT_GRPC_PORT')) if os.getenv('QDRANT_GRPC_PORT') else 6334
COLLECTION_NAME = 'ocel2-rag'
SEED = 10
warnings.filterwarnings('ignore')
base_path = os.path.join(os.path.dirname(__file__), '..')
eval_datasets = {
    'all': 'validation_dataset.csv',
    'global': 'validation_questions_global_info.csv',
    'events': 'validation_events_questions.csv',
    'objects': 'validation_objects_questions.csv',
    'ts': 'validation_timestamps_questions.csv',
}


def parse_arguments():
    parser.add_argument('--vector_chunk_size', type=int, default=2048, help='Chunk size for text splitting')
    parser.add_argument('--vector_chunk_overlap', type=int, default=128, help='Chunk overlap for text splitting')
    parser = ArgumentParser(description="Run Framework for OCEL2 analysis.")
    parser.add_argument('--embed_model_id', type=str, default='sentence-transformers/all-MiniLM-L12-v2',
                        help='Embedding model identifier')
    parser.add_argument('--vector_dimension', type=int, default=384,
                        help='Vector space dimension')
    parser.add_argument('--llm_id', type=str, default='meta-llama/Meta-Llama-3.1-8B-Instruct',
                        help='LLM model identifier')
    parser.add_argument('--model_max_length', type=int, help='Maximum input length (context window)',
                        default=128000)
    parser.add_argument('--num_documents_in_context', type=int, help='Number of documents in the context',
                        default=5)
    parser.add_argument('--log', type=str, help='The OCEL 2.0 event log in JSON to use',
                        default='ocel2-p2p.json')
    parser.add_argument('--max_new_tokens', type=int, help='Maximum number of tokens to generate',
                        default=1280)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--rebuild_db', type=u.str2bool,
                        help='Rebuild the vector index', default=True)
    args = parser.parse_args()

    return args


def initialize_vector_database(args, embed_model, space_dimension):
    try:
        q_client, q_store = vs.initialize_vector_store(
            URL, GRPC_PORT, COLLECTION_NAME, embed_model, space_dimension, args.rebuild_db
        )
        if args.rebuild_db:
            print("Rebuilding vector database...")
            vs.rebuild_and_populate_vector_db(
                base_path, q_client, embed_model, COLLECTION_NAME,
                batch_size=args.batch_size,
                chunk_size=args.vector_chunk_size,
                chunk_overlap=args.vector_chunk_overlap
            )
        return q_client, q_store
    except Exception as e:
        print(f"Error initializing vector database: {str(e)}")
        raise


def populate_vector_database(q_client, embed_model):
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
        actual_id = vs.store_vectorized_info(general_info, q_client, embed_model, COLLECTION_NAME)
        
        print("Populating the vector collection... (2/3)")
        to_chunk_path = os.path.join(execution_path, 'to_chunk')
        if os.path.exists(to_chunk_path):
            files_to_chunk = os.listdir(to_chunk_path)
            jsonfile = 'objects_ot_count.txt'
            
            for f in files_to_chunk:
                try:
                    full_file_path = os.path.join(to_chunk_path, f)
                    
                    if f.endswith('.txt') and f != jsonfile:
                        chunks_to_store = vs.intelligent_chunking_large_files(full_file_path)
                        print(f'Chunking completed for {f}')
                        actual_id = vs.store_vectorized_chunks(chunks_to_store, f, q_client, embed_model, COLLECTION_NAME, actual_id)
                    elif f == jsonfile:
                        with open(full_file_path, 'r') as jsonf:
                            print("Populating the vector collection... (3/3)")
                            j_dict = json.load(jsonf)
                            j_chunks = vs.intelligent_chunking_json(j_dict)
                            actual_id = vs.store_vectorized_chunks(j_chunks, f, q_client, embed_model, COLLECTION_NAME, actual_id)
                except Exception as chunk_error:
                    print(f"Error processing file {f}: {str(chunk_error)}")
                    continue
                    
    except Exception as e:
        print(f"Error populating vector database: {str(e)}")
        raise

def main():
    args = parse_arguments()

    try:
        embed_model_id = args.embed_model_id
        embed_model, actual_dimension = p.initialize_embedding_model(embed_model_id, DEVICE, args.batch_size)
        space_dimension = actual_dimension
        print(f"Using embedding dimension: {space_dimension}")
        q_client, q_store = initialize_vector_database(args, embed_model, space_dimension)
        num_docs = args.num_documents_in_context


        model_id = args.llm_id
        max_new_tokens = args.max_new_tokens
        os.environ['HF_TOKEN'] = HF_AUTH if HF_AUTH else ""
        os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY if OPENAI_API_KEY else ""
        run_data = {
            'Batch Size': args.batch_size,
            'Embedding Model ID': embed_model_id,
            'Vector Space Dimension': space_dimension,
            'Event Log': args.log,
            'LLM ID': model_id,
            'Context Window LLM': args.model_max_length,
            'Max Generated Tokens LLM': max_new_tokens,
            'Number of Documents in the Context': num_docs,
            'Rebuilt Vector Index': args.rebuild_db,
            'Vector Chunk Size': args.vector_chunk_size,
            'Vector Chunk Overlap': args.vector_chunk_overlap
        }

        print("Using OCEL2Pipeline...")
        pipeline = p.OCEL2Pipeline(model_id, max_new_tokens, HF_AUTH, OPENAI_API_KEY)
        p.live_prompting(pipeline, q_store, num_docs, run_data)
                    

    except Exception as e:
        print(f"Fatal error in main: {str(e)}")
        raise


if __name__ == "__main__":
    u.seed_everything(SEED)
    main()
