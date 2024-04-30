from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig, AutoConfig
from torch import cuda, bfloat16
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.qdrant import Qdrant
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
import os
import datetime

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


def retrieve_context(vector_index, query):
    retrieved = vector_index.similarity_search(query, k=3)
    retrieved_text = ''
    for i in range(len(retrieved)):
        content = retrieved[i].page_content
        if i != len(retrieved) - 1:
            retrieved_text += f'{content}\n\n'
        else: retrieved_text += f'{content}'

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


def log_to_file(message, curr_datetime):
    folder = 'tests'
    sub_folder = 'outputs'
    filename = f"output_{curr_datetime}.txt"
    filepath = os.path.join(folder, sub_folder, filename)
    with open(filepath, 'a') as file1:
        file1.write(message)


def produce_answer(question, curr_datetime, llm_chain, vectdb):
    sys_mess = "Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer."
    context = retrieve_context(vectdb, question)
    complete_answer = llm_chain.invoke({"question": question,
                               "system_message": sys_mess,
                               "context": context})
    index = complete_answer.find('[/INST]')
    prompt = complete_answer[:index + len('[/INST]')]
    answer = complete_answer[index + len('[/INST]'):]
    print(f'Prompt: {prompt}\n')
    print(f'Answer: {answer}\n')
    print('--------------------------------------------------')

    log_to_file(f'Query: {sys_mess}\n{context}\n{question}\n\nAnswer: {answer}\n\n\n', curr_datetime)


def live_prompting(model1, vect_db):
    current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    while True:
        query = input('Insert the query (type "quit" to exit): ')

        if query.lower() == 'quit':
            print("Exiting the chat.")
            break

        produce_answer(query, current_datetime, model1, vect_db)
        print()


if __name__ == "__main__":
    device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
    hf_token = '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'  # HuggingFace Token
    # Qdrant Credentials
    url = "192.168.1.240:6333"
    grpc_port = 6334

    embed_model_id = 'sentence-transformers/all-MiniLM-L6-v2'

    embed_model = initialize_embedding_model(embed_model_id, device)

    files = os.listdir(os.path.join('data', 'execution'))
    for f in files:
        if f.endswith('.txt'):
            content = load_process_representation(f)
            qdrant = store_vectorized_info(content, f, embed_model, url, grpc_port)

    model_id = 'meta-llama/Llama-2-13b-chat-hf'
    pipeline = initialize_pipeline(model_id, hf_token)
    hf_pipeline = HuggingFacePipeline(pipeline=pipeline)

    # template = """<s>[INST] {question} [/INST]"""
    template = """[INST]
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

    live_prompting(chain, qdrant)
