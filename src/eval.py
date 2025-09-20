import argparse
from dotenv import load_dotenv
import os
import torch

from oracle import AnswerVerificationOracle
import pipeline as p
import utility as u
import vector_store as vs


load_dotenv()
DEVICE = f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu'
HF_AUTH = os.getenv('HF_TOKEN')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
URL = os.getenv('QDRANT_URL')
GRPC_PORT = int(os.getenv('QDRANT_GRPC_PORT')) if os.getenv('QDRANT_GRPC_PORT') else 6334
COLLECTION_NAME = 'ocel2-rag'
SEED = 10
base_path = os.path.join(os.path.dirname(__file__), '..')
eval_datasets = {'all': 'validation_dataset.csv',
                 'global': 'validation_questions_global_info.csv',
                 'events': 'validation_events_questions.csv',
                 'objects': 'validation_objects_questions.csv',
                 'ts': 'validation_timestamps_questions.csv',}


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run OCEL2 Evaluation.")
    parser.add_argument('--vector_chunk_size', type=int, default=2048, help='Chunk size for text splitting')
    parser.add_argument('--vector_chunk_overlap', type=int, default=128, help='Chunk overlap for text splitting')
    parser.add_argument('--embed_model_id', type=str, default='sentence-transformers/all-MiniLM-L12-v2', help='Embedding model identifier')
    parser.add_argument('--vector_dimension', type=int, default=384, help='Vector space dimension')
    parser.add_argument('--llm_id', type=str, default='gemini-2.5-flash', help='LLM model identifier')
    parser.add_argument('--model_max_length', type=int, default=128000, help='Maximum input length (context window)')
    parser.add_argument('--num_documents_in_context', type=int, default=5, help='Number of documents in the context')
    parser.add_argument('--log', type=str, default='data/ocel2-p2p.json', help='The OCEL 2.0 event log in JSON to use')
    parser.add_argument('--max_new_tokens', type=int, default=1280, help='Maximum number of tokens to generate')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--modality', type=str, default='evaluation-all', help='Evaluation modality: evaluation-all, evaluation-global, '
                             'evaluation-events, evaluation-objects, evaluation-ts')
    parser.add_argument('--rebuild_db', type=u.str2bool, help='Rebuild the vector index', default=True)
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    u.seed_everything(SEED)
    embed_model_id = args.embed_model_id
    embed_model, actual_dimension = p.initialize_embedding_model(embed_model_id, DEVICE, args.batch_size)
    space_dimension = actual_dimension
    print(f"Using embedding dimension: {space_dimension}")

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

    q_client, q_store = initialize_vector_database(args, embed_model, space_dimension)
    num_docs = args.num_documents_in_context
    model_id = args.llm_id
    max_new_tokens = args.max_new_tokens
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
        'Rebuilt Vector Index': args.rebuild_db,
        'Vector Chunk Size': args.vector_chunk_size,
        'Vector Chunk Overlap': args.vector_chunk_overlap
    }
    print("Using OCEL2Pipeline for evaluation...")
    pipeline = p.OCEL2Pipeline(model_id, max_new_tokens, HF_AUTH, OPENAI_API_KEY)
    test_set_path = os.path.join(base_path, 'tests', 'test_dataset')
    modality_suffix = args.modality.split('-')[-1]
    dataset_name = eval_datasets[modality_suffix]
    if modality_suffix != 'all':
        test_set_path = os.path.join(test_set_path, 'divided_dataset', dataset_name)
    else:
        test_set_path = os.path.join(test_set_path, dataset_name)
    test_list = u.load_csv_questions(test_set_path)
    oracle = AnswerVerificationOracle(run_data)
    count = 0
    for q, a in test_list:
        try:
            oracle.add_question_expected_answer_pair(q, a)
            prompt, answer = pipeline.produce_answer(q, q_store, num_docs, run_data)
            oracle.verify_answer(prompt, q, answer)
            count += 1
            print(f'Processing answer for trace {count} of {len(test_list)}...')
        except Exception as question_error:
            print(f"Error processing question {count + 1}: {str(question_error)}")
            count += 1
            continue
    print('Evaluation process completed. Check the output file.')
    oracle.write_results_to_file()


if __name__ == "__main__":
    main()
