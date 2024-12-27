import datetime
import json
import os
import random

from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig, AutoConfig
from torch import bfloat16

from oracle import AnswerVerificationOracle
from utility import log_to_file
from vector_store import retrieve_context

llama3_models = ['meta-llama/Meta-Llama-3-8B-Instruct', 'meta-llama/Meta-Llama-3.1-8B-Instruct',
                 'meta-llama/Llama-3.2-1B-Instruct', 'meta-llama/Llama-3.2-3B-Instruct']
mistral_models = ['mistralai/Mistral-7B-Instruct-v0.2', 'mistralai/Mistral-7B-Instruct-v0.3',
                  'mistralai/Mistral-Nemo-Instruct-2407', 'mistralai/Ministral-8B-Instruct-2410']
qwen_models = ['Qwen/Qwen2.5-7B-Instruct']
openai_models = ['gpt-4o-mini']


def initialize_embedding_model(embedding_model_id, dev, batch_size):
    embedding_model = HuggingFaceEmbeddings(
        model_name=embedding_model_id,
        model_kwargs={'device': dev},
        encode_kwargs={'device': dev, 'batch_size': batch_size}
    )

    return embedding_model


def initialize_pipeline(model_identifier, hf_token, max_new_tokens):
    """
    Initializes a pipeline for text generation using a pre-trained language model and its tokenizer.

    Args:
        model_identifier (str): The identifier of the pre-trained language model.
        hf_token (str): The token used for the language model.
        max_new_tokens (int): The maximum number of tokens to generate.

    Returns:
        generate_text: The pipeline for text generation.
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=bfloat16
    )
    model_config = AutoConfig.from_pretrained(
        model_identifier,
        token=hf_token
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_identifier,
        trust_remote_code=True,
        config=model_config,
        quantization_config=bnb_config,
        device_map='auto',
        token=hf_token
    )
    model.eval()
    # print(f"Model loaded on {device}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_identifier,
        token=hf_token
    )
    if model_identifier in llama3_models:
        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        generate_text = pipeline(
            model=model, tokenizer=tokenizer,
            return_full_text=True,
            task='text-generation',
            eos_token_id=terminators,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            max_new_tokens=max_new_tokens,
            repetition_penalty=1.1
        )
    elif model_identifier in mistral_models:
        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("[/INST]]")
        ]
        generate_text = pipeline(
            model=model, tokenizer=tokenizer,
            return_full_text=True,
            task='text-generation',
            eos_token_id=terminators,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            max_new_tokens=max_new_tokens,
            repetition_penalty=1.1
        )
    elif model_identifier in qwen_models:
        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|im_end|>")
        ]
        generate_text = pipeline(
            model=model, tokenizer=tokenizer,
            return_full_text=True,
            task='text-generation',
            eos_token_id=terminators,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            max_new_tokens=max_new_tokens,
            repetition_penalty=1.1
        )
    else:
        generate_text = pipeline(
            model=model, tokenizer=tokenizer,
            return_full_text=True,
            task='text-generation',
            do_sample=True,
            max_new_tokens=max_new_tokens,
            repetition_penalty=1.1
        )

    return generate_text


def generate_prompt_template(model_id):
    path_prompts = os.path.join(os.path.dirname(__file__), 'prompts.json')
    with open(path_prompts, 'r') as prompt_file:
        prompts = json.load(prompt_file)

    if model_id in llama3_models:
        template = prompts.get('template-llama_instruct', '')
    elif model_id in mistral_models:
        template = prompts.get('template-mistral', '')
    elif model_id in qwen_models:
        template = prompts.get('template-qwen', '')
    else:
        template = prompts.get('template-generic', '')
    prompt = PromptTemplate.from_template(template)

    return prompt


def initialize_chain(model_id, hf_auth, openai_auth, max_new_tokens):
    if model_id not in openai_models:
        generate_text = initialize_pipeline(model_id, hf_auth, max_new_tokens)
        hf_pipeline = HuggingFacePipeline(pipeline=generate_text)
        prompt = generate_prompt_template(model_id)
        chain = prompt | hf_pipeline
    else:
        chain = OpenAI(
            api_key=openai_auth,
        )

    return chain


def produce_answer(question, model_id, llm_chain, vectdb, num_chunks, info_run):
    # modality = info_run['Evaluation Modality']
    path_prompts = os.path.join(os.path.dirname(__file__), 'prompts.json')
    with open(path_prompts, 'r') as prompt_file:
        prompts = json.load(prompt_file)
    sys_mess = prompts.get('system_message', '')
    sys_mess += prompts.get('evaluation-attributes', '').replace('REPLACE', info_run['Event Attributes']).replace(
        'ACTIVITIES', info_run['Activities'])

    context = retrieve_context(vectdb, question, num_chunks)

    if model_id not in openai_models:
        complete_answer = llm_chain.invoke({"question": question,
                                            "system_message": sys_mess,
                                            "context": context})
        prompt, answer = parse_llm_answer(complete_answer, model_id)
    else:
        prompt = f'{sys_mess}\nHere is the most similar past traces: {context}\n' + f'Here is the prefix to predict: {question}\nAnswer: '
        completion = llm_chain.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": f'{sys_mess}\nHere is the most similar past traces: {context}\n'},
                {"role": "user", "content": f'Here is the prefix to predict: {question}\nAnswer: '},
            ]
        )
        answer = completion.choices[0].message.content.strip()

    return prompt, answer


def parse_llm_answer(compl_answer, llm_choice):
    if llm_choice in llama3_models:
        delimiter = '<|start_header_id|>assistant<|end_header_id|>'
    elif llm_choice in mistral_models:
        delimiter = '[/INST]'
    elif llm_choice in qwen_models:
        delimiter = '<|im_start|>assistant'
    else:
        delimiter = 'Answer:'

    index = compl_answer.find(delimiter)
    prompt = compl_answer[:index + len(delimiter)]
    answer = compl_answer[index + len(delimiter):]

    return prompt, answer


def generate_live_response(question, curr_datetime, model_chain, vectordb, choice_llm, num_chunks, info_run):
    complete_prompt, answer = produce_answer(question, choice_llm, model_chain, vectordb, num_chunks, info_run)
    print(f'Prompt: {complete_prompt}\n')
    print(f'Answer: {answer}\n')
    print('--------------------------------------------------')

    log_to_file(f'Query: {complete_prompt}\n\nAnswer: {answer}\n\n##########################\n\n',
                curr_datetime, info_run)


def live_prompting(choice_llm, model1, vect_db, num_chunks, info_run):
    current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    while True:
        query = input('Insert the query (type "quit" to exit): ')

        if query.lower() == 'quit':
            print("Exiting the chat.")
            break

        generate_live_response(query, current_datetime, model1, vect_db, choice_llm, num_chunks, info_run)
        print()


def evaluate_rag_pipeline(choice_llm, lang_chain, vect_db, num_chunks, list_questions, info_run):
    oracle = AnswerVerificationOracle(info_run)
    count = 0
    for el in list_questions:
        prefix = el[0]
        expected_prediction = el[1]
        oracle.add_prefix_with_expected_answer_pair(prefix, expected_prediction)
        prompt, answer = produce_answer(prefix, choice_llm, lang_chain, vect_db, num_chunks, info_run)
        #print(f'Prompt: {prompt}\n')
        oracle.verify_answer(prompt, prefix, answer)
        count += 1
        print(f'Processing prediction for prefix {count} of {len(list_questions)}...')

    print('Validation process completed. Check the output file.')
    oracle.write_results_to_file()
