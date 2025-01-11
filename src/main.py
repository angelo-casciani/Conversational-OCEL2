from argparse import ArgumentParser
import json
from dotenv import load_dotenv
import os
import torch
import warnings

import pipeline as p
import utility as u
import vector_store as vs

DEVICE = f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu'
load_dotenv()
HF_AUTH = os.getenv('HF_TOKEN')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
URL = os.getenv('QDRANT_URL')
GRPC_PORT = int(os.getenv('QDRANT_GRPC_PORT'))
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
    parser = ArgumentParser(description="Run LLM Generation.")
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
                        help='Rebuild the vector index', default=False)
    parser.add_argument('--modality', type=str, default='live',
                        help='Modality to use between: evaluation-all, evaluation-global, evaluation-events, '
                             'evaluation-objects, evaluation-ts, live.')
    args = parser.parse_args()

    return args


def main():
    args = parse_arguments()

    embed_model_id = args.embed_model_id
    embed_model = p.initialize_embedding_model(embed_model_id, DEVICE, args.batch_size)
    space_dimension = args.vector_dimension

    q_client, q_store = vs.initialize_vector_store(URL, GRPC_PORT, COLLECTION_NAME, embed_model, space_dimension)
    num_docs = args.num_documents_in_context
    test_set_path = os.path.join(base_path, 'tests', 'test_dataset')
    if args.rebuild_db:
        vs.delete_qdrant_collection(q_client, COLLECTION_NAME)
        q_client, q_store = vs.initialize_vector_store(URL, GRPC_PORT, COLLECTION_NAME, embed_model, space_dimension)

        #with open('preprocessing.py') as file:
        #   exec(file.read())
        print("Building and populating the vector collection... (1/3)")
        files = os.listdir(os.path.join(base_path, 'data', 'execution'))
        general_info = []
        for f in files:
            if f.endswith('.txt'):
                content = u.load_process_representation(f)
                general_info.append((f, content))
        actual_id = vs.store_vectorized_info(general_info, q_client, embed_model, COLLECTION_NAME)
        print(f"Populating the vector collection... (2/3)")
        files_to_chunk = os.listdir(os.path.join(base_path, 'data', 'execution', 'to_chunk'))
        jsonfile = 'objects_ot_count.txt'
        for f in files_to_chunk:
            if f.endswith('.txt') and f != jsonfile:
                chunks_to_store = vs.intelligent_chunking_large_files(f)
                print('Chunking completed')
                actual_id = vs.store_vectorized_chunks(chunks_to_store, f, q_client, embed_model, COLLECTION_NAME, actual_id)
            elif f == jsonfile:
                file_path = os.path.join(base_path, 'data', 'execution', 'to_chunk', jsonfile)
                with open(file_path, 'r') as jsonf:
                    print(f"Populating the vector collection... (3/3)")
                    j_dict = json.load(jsonf)
                    j_chunks = vs.intelligent_chunking_json(j_dict)
                    actual_id = vs.store_vectorized_chunks(j_chunks, f, q_client, embed_model, COLLECTION_NAME, actual_id)

        print(f"Vector collection successfully created and initialized!")

    model_id = args.llm_id
    max_new_tokens = args.max_new_tokens
    chain = p.initialize_chain(model_id, HF_AUTH, OPENAI_API_KEY, max_new_tokens)

    run_data = {
        'Batch Size': args.batch_size,
        'Embedding Model ID': embed_model_id,
        'Vector Space Dimension': space_dimension,
        'Evaluation Modality': args.modality,
        'Event Log': args.log,
        'LLM ID': model_id,
        'Context Window LLM': args.model_max_length,
        'Max Generated Tokens LLM': max_new_tokens,
        'Number of Documents in the Context': num_docs,
        'Rebuilt Vector Index': args.rebuild_db
    }

    if 'evaluation' in args.modality:
        modality_suffix = args.modality.split('-')[-1]
        test_list = u.load_csv_questions(eval_datasets[modality_suffix])
        p.evaluate_rag_chain(model_id, chain, q_store, num_docs, test_list, run_data)
    else:
        p.live_prompting(model_id, chain, q_store, num_docs, run_data)


if __name__ == "__main__":
    u.seed_everything(SEED)
    main()
