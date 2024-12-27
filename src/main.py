from argparse import ArgumentParser
from dotenv import load_dotenv
import os
import torch
import warnings

import log_preprocessing as lp
import pipeline as p
import utility as u
import vector_store as vs

DEVICE = f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
load_dotenv()
HF_AUTH = os.getenv('HF_TOKEN')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
URL = os.getenv('QDRANT_URL')
GRPC_PORT = int(os.getenv('QDRANT_GRPC_PORT'))
COLLECTION_NAME = 'nap-rag'
SEED = 10
warnings.filterwarnings('ignore')


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
    parser.add_argument('--log', type=str, help='The event log to use for the next activity prediction',
                        default='Hospital_log.xes')
    parser.add_argument('--prefix_base', type=int, help='Maximum number of tokens to generate',
                        default=1)
    parser.add_argument('--prefix_gap', type=int, help='Maximum number of tokens to generate',
                        default=3)
    parser.add_argument('--max_new_tokens', type=int, help='Maximum number of tokens to generate',
                        default=1280)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--rebuild_db_and_tests', type=u.str2bool,
                        help='Rebuild the vector index and the test set', default=False)
    parser.add_argument('--modality', type=str, default='evaluation-concept_names',
                        help='Modality to use between: evaluation-concept_names, evaluation-attributes, '
                             'live-concept_names, live-attributes')
    args = parser.parse_args()

    return args


def main():
    args = parse_arguments()

    embed_model_id = args.embed_model_id
    embed_model = p.initialize_embedding_model(embed_model_id, DEVICE, args.batch_size)
    space_dimension = args.vector_dimension

    q_client, q_store = vs.initialize_vector_store(URL, GRPC_PORT, COLLECTION_NAME, embed_model, space_dimension)
    num_docs = args.num_documents_in_context
    test_set_path = os.path.join(os.path.dirname(__file__), '..', 'tests', 'test_sets',
                                 f"test_set_{args.log.split('.xes')[0]}_{args.modality}.csv")
    event_attributes = []
    activities_set = set()
    total_traces_size = 0
    test_set_size = 0
    traces_to_store_size = 0
    test_size_prefixes = 0
    if args.rebuild_db_and_tests:
        vs.delete_qdrant_collection(q_client, COLLECTION_NAME)
        q_client, q_store = vs.initialize_vector_store(URL, GRPC_PORT, COLLECTION_NAME, embed_model, space_dimension)
        content = lp.read_event_log(args.log)
        #if 'attributes' in args.modality:
        traces, event_attributes, activities_set = lp.extract_traces_with_attributes(content)
        total_traces_size = len(traces)
        #else:
            #traces = lp.extract_traces_concept_names(content)
        #if 'evaluation' in args.modality:
        test_set = lp.generate_test_set(traces, 0.2)
        test_set_size = len(test_set)
        traces = [trace for trace in traces if trace not in test_set]
        traces = lp.process_traces_with_last_attribute_values(traces)
        traces_to_store_size = len(traces)
        prefix_prediction = lp.create_prefixes_with_attribute_last_values(test_set, base=args.prefix_base, gap=args.prefix_gap)
        prefix_prediction_pairs = lp.process_prefixes_prediction_with_last_attribute_values(prefix_prediction)
        test_size_prefixes = len(prefix_prediction_pairs)
        lp.generate_csv_from_test_set(test_set=prefix_prediction_pairs, test_path=test_set_path)
        vs.store_traces(traces, q_client, args.log, embed_model, COLLECTION_NAME)

    model_id = args.llm_id
    max_new_tokens = args.max_new_tokens
    chain = p.initialize_chain(model_id, HF_AUTH, OPENAI_API_KEY, max_new_tokens)
    
    run_data = {
        'Batch Size': args.batch_size,
        'Embedding Model ID': embed_model_id,
        'Vector Space Dimension': space_dimension,
        'Evaluation Modality': args.modality,
        'Event Log': args.log,
        'Total Traces in Log': total_traces_size,
        'Test Set Size': test_set_size,
        'Traces Stored Size': traces_to_store_size,
        'Test Set Prefixes Size': test_size_prefixes,
        'LLM ID': model_id,
        'Context Window LLM': args.model_max_length,
        'Max Generated Tokens LLM': max_new_tokens,
        'Number of Documents in the Context': num_docs,
        'Rebuilt Vector Index and Test Set': args.rebuild_db_and_tests
    }
    if event_attributes:
        run_data['Event Attributes'] = str(event_attributes)
        run_data['Activities'] = str(activities_set)

    if 'evaluation' in args.modality:
        test_list = u.load_csv_questions(test_set_path)
        p.evaluate_rag_pipeline(model_id, chain, q_store, num_docs, test_list, run_data)
    else:
        p.live_prompting(model_id, chain, q_store, num_docs, run_data)


if __name__ == "__main__":
    u.seed_everything(SEED)
    main()
