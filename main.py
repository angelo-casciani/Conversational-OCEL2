from qdrant_client import QdrantClient, models
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig, AutoConfig
from torch import cuda, bfloat16
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.qdrant import Qdrant
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
import csv
import datetime
import json
import os
import re
import subprocess
from oracle import AnswerVerificationOracle

"""Initializing the Hugging Face Embedding Pipeline

Hugging Face Embedding model: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
"""


def initialize_embedding_model(embedding_model_id, dev):
    embedding_model = HuggingFaceEmbeddings(
        model_name=embedding_model_id,
        model_kwargs={'device': dev},
        encode_kwargs={'device': dev, 'batch_size': 32}
        # multi_process=True
    )

    return embedding_model


"""Once defined embeddings model and vectorDB with its indexing methods in-place, we proceed to the indexing process.

We need to describe the process: we incorporate the representation and generate the indexes.
"""


def load_process_representation(filename):
    filepath = os.path.join('data', 'execution', filename)
    with open(filepath, 'r') as file:
        file_content = file.read()
        return file_content


def store_vectorized_info(file_content, filename, embeds_model, address, port):
    source = filename.strip('.txt').capitalize()
    title = f"{source}"
    batch_size = 2048
    qdrant_store = 'No qDrant store.'
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=batch_size, chunk_overlap=128)
    # all_splits = text_splitter.split_documents(file_content)
    splits = text_splitter.split_text(file_content)
    all_splits = text_splitter.create_documents(splits)

    for document in all_splits:
        chunk = document.page_content
        metadata = {'text': chunk, 'source': source, 'title': title}
        qdrant_store = Qdrant.from_texts(
            [chunk],
            embeds_model,
            metadatas=[metadata],
            url=address,
            prefer_grpc=True,
            grpc_port=port,
            collection_name="llama-2-rag",
        )
    return qdrant_store


# Chunk size in lines of the file
def intelligent_chunking_large_files(file_path, chunk_size=1):
    file_path = os.path.join('data', 'execution', 'to_chunk', file_path)
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


def store_vectorized_chunks(chunks_to_save, filename, embeds_model, qdrant_client, identifier):
    source = filename.strip('.txt').capitalize()
    title = f"{source}"
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
    identifier += 1
    for chunk in chunks_to_save:
        match = re.search(pattern, chunk)
        if match is not None:
            if filename == 'objects_ot_count.txt':
                meta_value = match.group(0).strip()
            else:
                meta_value = match.group(1).strip()
        else: meta_value = ''
        metadata = {'page_content': chunk, 'source': source, 'title': title, meta_search: meta_value}
        point = models.PointStruct(
            id=identifier,
            vector=embeds_model.embed_documents([chunk])[0],
            payload=metadata
        )
        points.append(point)
        identifier += 1

    print("Created points for this phase!")
    qdrant_client.upsert(
        collection_name="llama-2-rag",
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


def retrieve_context(vector_index, query, search_filter='', key=''):
    retrieved = vector_index.similarity_search(query, k=3)
    if search_filter != '':
        filter_ = models.Filter(
            must=[
                models.FieldCondition(
                    key=key,
                    match=models.MatchValue(value=search_filter)
                )
            ]
        )
        meta_retrieved = vector_index.similarity_search(query, filter=filter_, k=3)
        if len(meta_retrieved) > 0:
            retrieved = meta_retrieved
    retrieved_text = ''
    for i in range(len(retrieved)):
        content = retrieved[i].page_content
        if i != len(retrieved) - 1:
            retrieved_text += f'{content}\n\n'
        else:
            retrieved_text += f'{content}'

    return retrieved_text


def initialize_pipeline(model_identifier, hf_auth):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=bfloat16
    )

    model_config = AutoConfig.from_pretrained(
        model_identifier,
        token=hf_auth
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_identifier,
        trust_remote_code=True,
        config=model_config,
        quantization_config=bnb_config,
        device_map='auto',
        token=hf_auth
    )
    # model.eval()
    # print(f"Model loaded on {device}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_identifier,
        token=hf_auth
    )

    generate_text = pipeline(
        model=model, tokenizer=tokenizer,
        return_full_text=True,
        task='text-generation',
        do_sample=True,
        max_new_tokens=512,
        repetition_penalty=1.1
    )

    return generate_text


def produce_answer(question, llm_chain, vectdb):
    # sys_mess = "Use the following pieces of context to answer with 'True' or 'False' the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer."
    sys_mess = "Use the following pieces of context to answer with 'True' or 'False' the question at the end."

    # Take the question and extract the metadata for the filtering if any.
    # Pattern in the question: metadata_key "metadata_value", while for events "event:id_num"
    pattern_oid = r'ocel:oid\s*"([^"]+)"'
    match_oid = re.search(pattern_oid, question)
    meta_value_oid = match_oid.group(1).strip() if match_oid else ''
    meta_search_oid = 'ocel:oid'

    pattern_ts = r'ocel:timestamp\s*"([^"]+)"'
    match_ts = re.search(pattern_ts, question)
    meta_value_ts = match_ts.group(1).strip() if match_ts else ''
    meta_search_ts = 'ocel:timestamp'

    pattern_js = r'"event:\d+"'
    match_js = re.search(pattern_js, question)
    meta_value_js = match_js.group(0).strip('"') if match_js else ''
    meta_search_js = 'event:id'

    if meta_value_oid:
        search_filter = meta_value_oid
        context = retrieve_context(vectdb, question, search_filter, meta_search_oid)
    elif meta_value_ts:
        search_filter = meta_value_ts
        context = retrieve_context(vectdb, question, search_filter, meta_search_ts)
    elif meta_value_js:
        search_filter = meta_value_js
        context = retrieve_context(vectdb, question, search_filter, meta_search_js)
    else:
        context = retrieve_context(vectdb, question)
    complete_answer = llm_chain.invoke({"question": question,
                                        "system_message": sys_mess,
                                        "context": context})
    index = complete_answer.find('[/INST]')
    prompt = complete_answer[:index + len('[/INST]')]
    answer = complete_answer[index + len('[/INST]'):]
    return prompt, answer


def delete_qdrant_collection():
    qdrant_client = QdrantClient(url="192.168.1.240:6333", grpc_port=6334, prefer_grpc=True)
    qdrant_client.delete_collection('llama-2-rag')
    qdrant_client.close()


def evaluate_rag_chain_zero_shot(eval_oracle, lang_chain, vect_db, filename):
    path_tests_data = filename
    questions = {}
    with open(path_tests_data, newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            question, answer = row
            questions[question] = answer

    count = 0
    for q, a in questions.items():
        eval_oracle.add_prompt_expected_answer_pair(q, a)
        prompt, answer = produce_answer(q, lang_chain, vect_db)
        eval_oracle.verify_answer(answer, prompt)
        count += 1
        print(f'Processing answer for trace {count} of {len(questions)}...')

    print('Validation process completed. Check the output file.')
    eval_oracle.write_results_to_file()


if __name__ == "__main__":
    device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
    hf_token = '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'  # HuggingFace Token
    # Qdrant Credentials
    url = "192.168.1.240:6333"
    grpc_port = 6334

    embed_model_id = 'sentence-transformers/all-MiniLM-L6-v2'

    embed_model = initialize_embedding_model(embed_model_id, device)

    index_ans = input("Create vector collection?\ny - yes, create it for the first time/rebuild it;\nn - no, maintain it.\n")
    index_bool = False
    while index_ans.lower() not in ['y','n']:
        if index_ans == 'y':
            index_bool = True
        elif index_ans == 'n':
            index_bool = False
        else:
            print("Insert a valid choice!")
            print("Rebuild vector collection?\ny - yes, rebuild it/create it for the first time.\nn - no, maintain it.\n")

    client = QdrantClient(url, grpc_port=grpc_port, prefer_grpc=True)
    qdrant = Qdrant(client, collection_name='llama-2-rag', embeddings=embed_model)

    if index_bool:
        result = subprocess.run(['python', 'preprocessing.py'], capture_output=True, text=True)
        print(result.stdout)
        print(result.stderr)
        id = 0
        print("Building and populating the vector collection... (1/3)")
        files = os.listdir(os.path.join('data', 'execution'))
        for f in files:
            if f.endswith('.txt'):
                content = load_process_representation(f)
                qdrant = store_vectorized_info(content, f, embed_model, url, grpc_port)
        print(f"Populating the vector collection... (2/3)")
        files_to_chunk = os.listdir(os.path.join('data', 'execution', 'to_chunk'))
        jsonfile = 'objects_ot_count.txt'
        for f in files_to_chunk:
            if f.endswith('.txt') and f != jsonfile:
                chunks_to_store = intelligent_chunking_large_files(f)
                id = store_vectorized_chunks(chunks_to_store, f, embed_model, client, id)
            elif f == jsonfile:
                file_path = os.path.join('data', 'execution', 'to_chunk', jsonfile)
                with open(file_path, 'r') as jsonf:
                    print(f"Populating the vector collection... (3/3)")
                    j_dict = json.load(jsonf)
                    j_chunks = intelligent_chunking_json(j_dict)
                    id = store_vectorized_chunks(j_chunks, f, embed_model, client, id)
        print(f"vector collection successfully created and initialized!")
    else:
        print("Using existing vector collection.")

    model_choice = input("Choose the language model to use:\na - Llama 2 7B;\nb - Llama 2 13B;\nc - Llama 3 8B.\n")
    model_id = 'MODEL'
    while model_choice.lower() not in ['a','b','c']:
        if model_choice == 'a':
            model_id = 'meta-llama/Llama-2-7b-chat-hf'
        elif model_choice == 'b':
            model_id = 'meta-llama/Llama-2-13b-chat-hf'
        elif model_choice == 'c':
            model_id = 'meta-llama/Meta-Llama-3-8B-Instruct'
        else:
            print("Insert a valid choice!")
            model_choice = input("Choose the language model to use:\na - Llama 2 7B;\nb - Llama 2 13B;\nc - Llama 3 8B.\n")

    pipeline = initialize_pipeline(model_id, hf_token)
    hf_pipeline = HuggingFacePipeline(pipeline=pipeline)

    template = """<s>[INST]
                    <<SYS>>
                    {system_message}
                    <</SYS>>
                    <<CONTEXT>>
                    {context}
                    <</CONTEXT>>
                    <<QUESTION>>
                    {question}
                    <</QUESTION>>
                    <<ANSWER>> [/INST]"""

    prompt = PromptTemplate.from_template(template)
    chain = prompt | hf_pipeline

    oracle = AnswerVerificationOracle()
    filepaths = [ 
                'validation_dataset.csv',
                'validation_questions_global_info.csv',
                'validation_events_questions.csv',
                'validation_objects_questions.csv',
                'validation_timestamps_questions.csv'
                ]

    eval_choice = input("Choose the dataset (based on the P2P OCEL 2.0 log) for the evaluation:\na - Complete Dataset;\nb - Global Stats Dataset;\nc - Event Dataset;\nd - Object Dataset;\ne - Timestamp Dataset.\n").lower()
    while eval_choice not in ['a','b','c','d','e']:
        if eval_choice == 'a':
            evaluate_rag_chain_zero_shot(oracle, chain, qdrant, os.path.join('tests', 'test_dataset', filepaths[0]))
        elif eval_choice == 'b':
            evaluate_rag_chain_zero_shot(oracle, chain, qdrant, os.path.join('tests', 'test_dataset', 'divided_dataset', filepaths[1]))
        elif eval_choice == 'c':
            evaluate_rag_chain_zero_shot(oracle, chain, qdrant, os.path.join('tests', 'test_dataset', 'divided_dataset', filepaths[2]))
        elif eval_choice == 'd':
            evaluate_rag_chain_zero_shot(oracle, chain, qdrant, os.path.join('tests', 'test_dataset', 'divided_dataset', filepaths[3]))
        elif eval_choice == 'e':
            evaluate_rag_chain_zero_shot(oracle, chain, qdrant, os.path.join('tests', 'test_dataset', 'divided_dataset', filepaths[4]))
        else:
            print("Insert a valid choice!")
            eval_choice = input("Choose the dataset (based on the P2P OCEL 2.0 log) for the evaluation:\na - Complete Dataset;\nb - Global Stats Dataset;\nc - Event Dataset;\nd - Object Dataset;\ne - Timestamp Dataset.\n").lower()
    
    indexd_choice = input("Maintain vector collection?\ny - yes, maintain it.\nn - no, delete it.\n").lower()
    collection_bool = False
    while indexd_choice.lower() not in ['y','n']:
        if indexd_choice == 'y':
            collection_bool = True
        elif indexd_choice == 'n':
            collection_bool = False
        else:
            print("Insert a valid choice!")
            indexd_choice = input("Maintain vector collection?\ny - yes, maintain it.\nn - no, delete it.\n").lower()
    
    if not(collection_bool):
        delete_qdrant_collection()
