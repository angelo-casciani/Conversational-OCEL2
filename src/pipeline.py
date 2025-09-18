import datetime
import json
import os
import re
from typing import Dict, List, Tuple

from langchain.chat_models import init_chat_model
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig, AutoConfig
from torch import bfloat16
from sentence_transformers import SentenceTransformer

from utility import log_to_file
from vector_store import retrieve_context


class OCEL2Pipeline:   
    MODELS = {
        'api': {
            'openai': ['gpt-4o-mini', 'gpt-4.1-mini', 'gpt-4.1-nano', 'gpt-4.1', 'gpt-4o'],
            'google_genai': ['gemini-2.0-flash', 'gemini-2.5-flash-preview-05-20', 'gemini-2.5-pro', 'gemini-2.5-flash'],
            'deepseek': ['deepseek-chat', 'deepseek-reasoner'],
            'anthropic': [],
        },
        'local': {
            'metaai': ['meta-llama/Meta-Llama-3-8B-Instruct', 'meta-llama/Meta-Llama-3.1-8B-Instruct',
                       'meta-llama/Llama-3.2-1B-Instruct', 'meta-llama/Llama-3.2-3B-Instruct'],
            'mistral': ['mistralai/Mistral-7B-Instruct-v0.2','mistralai/Mistral-7B-Instruct-v0.3',
                        'mistralai/Mistral-Nemo-Instruct-2407', 'mistralai/Ministral-8B-Instruct-2410'],
            'qwen': ['Qwen/Qwen2.5-7B-Instruct'],
            'google_genai': ['google/gemma-2-9b-it'],
            'microsoft': ['microsoft/phi-4'],
            'deepseek': ['deepseek-ai/DeepSeek-R1-Distill-Qwen-7B', 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B'],
        }
    }
    
    TERMINATOR_TOKENS = {
        'metaai': "<|eot_id|>",
        'mistral': "[/INST]]",
        'qwen': "<|im_end|>",
        'microsoft': "<|im_sep|>"
    }
    
    TEMPLATE_MAPPING = {
        'metaai': 'template-llama_instruct',
        'mistral': 'template-mistral',
        'qwen': 'template-qwen',
        'microsoft': 'template-phi',
        'deepseek': 'template-deepseek',
        'openai': 'template-generic',
        'google_genai': 'template-generic',
        'anthropic': 'template-generic'
    }
    
    RESPONSE_DELIMITERS = {
        'metaai': '<|start_header_id|>assistant<|end_header_id|>',
        'mistral': '[/INST]',
        'qwen': '<|im_start|>assistant',
        'microsoft': '<|im_start|>assistant<|im_sep|>',
        'deepseek': 'Assistant: '
    }

    def __init__(self, model_id: str, max_new_tokens: int, hf_auth: str = None, openai_auth: str = None) -> None:
        self.model_id = model_id
        self.model_family, self.model_type = self._get_model_family_type()
        if not self.model_family:
            raise ValueError(f"Could not determine model family for ID: {self.model_id}. Please check MODELS definition.")
        self.max_new_tokens = max_new_tokens
        self.hf_auth = hf_auth
        self.openai_auth = openai_auth
        self.llm = self._initialize_model()
        self.prompt_template = self._generate_prompt_template()
        if self.model_type == 'local':
            self.chain = self.prompt_template | self.llm
        else:
            self.chain = self.llm

    def _get_model_family_type(self) -> Tuple[str, str]:
        model_id_lower = self.model_id.lower()
        for model_type, families in self.MODELS.items():
            for family, models_in_family in families.items():
                if any(model_id_lower == model.lower() or model_id_lower in model.lower() for model in models_in_family):
                    return family, model_type
        
        print(f"Warning: Model ID '{self.model_id}' not found in predefined MODELS. Defaulting to 'openai', 'api'.")
        return "openai", "api"

    def _initialize_model(self):
        try:
            if self.model_type == 'local':
                return self._initialize_local_model()
            elif self.model_type == 'api':
                return self._initialize_api_model()
            else:
                raise ValueError(f"Unsupported model_type: {self.model_type}. Must be 'local' or 'api'.")
        except Exception as e:
            print(f"Error initializing model {self.model_id}: {str(e)}")
            raise

    def _initialize_local_model(self):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=bfloat16
        )
        model_config = AutoConfig.from_pretrained(self.model_id, token=self.hf_auth)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            config=model_config,
            quantization_config=bnb_config,
            device_map='auto',
            token=self.hf_auth
        )
        model.eval()
        
        tokenizer = AutoTokenizer.from_pretrained(self.model_id, token=self.hf_auth)

        pipeline_params = {
            "model": model,
            "tokenizer": tokenizer,
            "return_full_text": True,
            "task": "text-generation",
            "do_sample": True,
            "temperature": 0.1,
            "max_new_tokens": self.max_new_tokens,
            "repetition_penalty": 1.1
        }

        model_family_key = self.model_family.lower()
        if model_family_key in self.TERMINATOR_TOKENS:
            terminators = [
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids(self.TERMINATOR_TOKENS[model_family_key])
            ]
            pipeline_params["eos_token_id"] = terminators
            pipeline_params["pad_token_id"] = tokenizer.eos_token_id
        
        generate_text = pipeline(**pipeline_params)
        return HuggingFacePipeline(pipeline=generate_text)

    def _initialize_api_model(self):
        if self.model_family == 'openai':
            return OpenAI(api_key=self.openai_auth)
        else:
            return init_chat_model(
                self.model_id,
                model_provider=self.model_family,
                temperature=0.1,
                max_tokens=self.max_new_tokens
            )

    def _generate_prompt_template(self) -> PromptTemplate:
        path_prompts = os.path.join(os.path.dirname(__file__), 'prompts.json')
        
        try:
            with open(path_prompts, 'r') as prompt_file:
                prompts = json.load(prompt_file)
        except FileNotFoundError:
            print(f"Warning: Prompts file not found at {path_prompts}. Using basic template.")
            return PromptTemplate.from_template("System: {system_message}\nContext: {context}\nQuestion: {question}\nAnswer:")

        model_family_key = self.model_family.lower()
        template_key = self.TEMPLATE_MAPPING.get(model_family_key, 'template-generic')
        template_content = prompts.get(template_key, '')
        
        if not template_content:
            print(f"Warning: Prompt template content for key '{template_key}' not found. Using basic fallback.")
            template_content = "System: {system_message}\nContext: {context}\nQuestion: {question}\nAnswer:"
        
        return PromptTemplate.from_template(template_content)

    def _get_system_message(self, info_run: Dict) -> str:
        path_prompts = os.path.join(os.path.dirname(__file__), 'prompts.json')
        
        try:
            with open(path_prompts, 'r') as prompt_file:
                prompts = json.load(prompt_file)
        except FileNotFoundError:
            return "You are an AI assistant specialized in analyzing OCEL2 event logs."
        
        modality = info_run.get('Evaluation Modality', 'live')
        if 'evaluation' in modality:
            sys_mess = prompts.get('system_message-eval', '')
        else:
            sys_mess = prompts.get('system_message-live', '')
            
        return sys_mess or "You are an AI assistant specialized in analyzing OCEL2 event logs."

    def produce_answer(self, question: str, vectdb, num_chunks: int, info_run: Dict) -> Tuple[str, str]:
        sys_mess = self._get_system_message(info_run)
        
        # Extract metadata for filtering
        context = self._get_filtered_context(question, vectdb, num_chunks)
        
        try:
            if self.model_type == 'local':
                payload = {
                    "question": question,
                    "system_message": sys_mess,
                    "context": context
                }
                result = self.chain.invoke(payload)
                prompt, answer = self._parse_llm_answer(result)
            else:
                prompt = f'{sys_mess}\nHere is the context: {context}\n' + f'Question: {question}\nAnswer: '
                if self.model_family == 'openai':
                    completion = self.chain.chat.completions.create(
                        model=self.model_id,
                        messages=[
                            {"role": "system", "content": f'{sys_mess}\nHere is the context: {context}\n'},
                            {"role": "user", "content": f'Question: {question}\nAnswer: '},
                        ]
                    )
                    answer = completion.choices[0].message.content.strip()
                else:
                    # For other API models using langchain
                    result = self.chain.invoke([
                        {"role": "system", "content": f'{sys_mess}\nHere is the context: {context}\n'},
                        {"role": "user", "content": f'Question: {question}\nAnswer: '},
                    ])
                    answer = result.content if hasattr(result, 'content') else str(result)
        except Exception as e:
            print(f"Error generating answer: {str(e)}")
            return f"Error in prompt generation: {str(e)}", f"Sorry, I encountered an error: {str(e)}"

        return prompt, answer

    def _get_filtered_context(self, question: str, vectdb, num_chunks: int) -> str:
        # Pattern matching for metadata filtering
        pattern_oid = r'ocel:oid\s*"([^"]+)"'
        match_oid = re.search(pattern_oid, question)
        meta_value_oid = match_oid.group(1).strip() if match_oid else ''
        meta_search_oid = 'ocel_oid'

        pattern_ts = r'ocel:timestamp\s*"([^"]+)"'
        match_ts = re.search(pattern_ts, question)
        meta_value_ts = match_ts.group(1).strip() if match_ts else ''
        meta_search_ts = "ocel_timestamp"

        pattern_js = r'"event:\d+"'
        match_js = re.search(pattern_js, question)
        meta_value_js = match_js.group(0).strip('"') if match_js else ''
        meta_search_js = "event_id"

        try:
            if meta_value_oid:
                return retrieve_context(vectdb, question, num_chunks, meta_search_oid, meta_value_oid)
            elif meta_value_ts:
                return retrieve_context(vectdb, question, num_chunks, meta_search_ts, meta_value_ts)
            elif meta_value_js:
                return retrieve_context(vectdb, question, num_chunks, meta_search_js, meta_value_js)
            else:
                return retrieve_context(vectdb, question, num_chunks)
        except Exception as e:
            print(f"Error retrieving context: {str(e)}")
            return "Context retrieval failed."

    def _parse_llm_answer(self, complete_answer: str) -> Tuple[str, str]:
        model_family_key = self.model_family.lower()
        delimiter = self.RESPONSE_DELIMITERS.get(model_family_key, 'Answer:')

        index = complete_answer.find(delimiter)
        if index != -1:
            prompt = complete_answer[:index + len(delimiter)]
            answer = complete_answer[index + len(delimiter):]
        else:
            prompt = complete_answer
            answer = ""

        return prompt, answer


EMBEDDING_MODELS = {
    'jina': [
        'jinaai/jina-embeddings-v4',
        'jinaai/jina-embeddings-v3',
        'jinaai/jina-embeddings-v2-base-en'
    ],
    'sentence_transformers': [
        'sentence-transformers/all-MiniLM-L12-v2',
        'sentence-transformers/all-MiniLM-L6-v2',
        'sentence-transformers/all-mpnet-base-v2'
    ]
}


def create_jina_v4_embeddings(model_name: str, device: str = 'cpu', default_task: str = 'retrieval'):
    class JinaV4Wrapper(HuggingFaceEmbeddings):
        def __init__(self):
            super().__init__(  # Dummy model to satisfy parent
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': device}
            )
            object.__setattr__(self, '_jina_model', SentenceTransformer(model_name, trust_remote_code=True, device=device))
            object.__setattr__(self, '_default_task', default_task)
            
        def embed_query(self, text: str) -> List[float]:   # Embed single query
            embedding = self._jina_model.encode(
                sentences=[text],
                task=self._default_task,
                prompt_name="query"
            )
            return embedding[0].tolist()
        
        def embed_documents(self, texts: List[str]) -> List[List[float]]: # Embed multiple document texts
            embeddings = self._jina_model.encode(
                sentences=texts,
                task=self._default_task,
                prompt_name="passage"
            )
            return embeddings.tolist()
    
    return JinaV4Wrapper()


def initialize_embedding_model(model_id: str, device: str = 'cpu', batch_size: int = 1) -> Tuple[object, int]:
    try:
        print(f"Initializing embedding model: {model_id}")
        if model_id == "jinaai/jina-embeddings-v4":
            print("Using custom JinaV4Embeddings wrapper")
            embedding_model = create_jina_v4_embeddings(model_id, device=device)
            test_embedding = embedding_model.embed_query("test")   # Test the model to get dimensions
            actual_dimension = len(test_embedding)
            print(f"Jina v4 embedding dimensions: {actual_dimension}")
            return embedding_model, actual_dimension
        
        fallback_models = []
        for model_family in EMBEDDING_MODELS.values():
            fallback_models.extend(model_family)
        models_to_try = [model_id] + [m for m in fallback_models if m != model_id]
        
        for model_name in models_to_try:
            try:
                print(f"Attempting to load embedding model: {model_name}")
                if model_name == "jinaai/jina-embeddings-v4":
                    print("Using custom JinaV4Embeddings wrapper for fallback")
                    embedding_model = create_jina_v4_embeddings(model_name, device=device)
                    test_embedding = embedding_model.embed_query("test")
                    actual_dimension = len(test_embedding)
                    print(f"Jina v4 embedding dimensions: {actual_dimension}")
                    return embedding_model, actual_dimension
                
                embedding_model = HuggingFaceEmbeddings(   # Standard model loading
                    model_name=model_name,
                    model_kwargs={'device': device},
                    encode_kwargs={'device': device, 'batch_size': batch_size}
                )
                test_embedding = embedding_model.embed_query("test")   # Get dimensions
                actual_dimension = len(test_embedding)
                print(f"Successfully loaded {model_name} with {actual_dimension} dimensions")
                return embedding_model, actual_dimension
                
            except Exception as e:
                print(f"Failed to load {model_name}: {str(e)}")
                continue
        raise ValueError(f"Unable to load any embedding model. Last tried: {models_to_try}")
        
    except Exception as e:
        print(f"Error initializing embedding model: {str(e)}")
        raise


def live_prompting(pipeline: OCEL2Pipeline, vect_db, num_chunks: int, info_run: Dict):
    current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print("Starting live prompting mode...")
    
    while True:
        try:
            query = input('Insert the query (type "quit" to exit): ')

            if query.lower() == 'quit':
                print("Exiting the chat.")
                break

            complete_prompt, answer = pipeline.produce_answer(query, vect_db, num_chunks, info_run)
            print(f'Prompt: {complete_prompt}\n')
            print(f'Answer: {answer}\n')
            print('--------------------------------------------------')

            log_to_file(f'Query: {complete_prompt}\n\nAnswer: {answer}\n\n##########################\n\n',
                       current_datetime, info_run)
            print()
        except KeyboardInterrupt:
            print("\nExiting the chat.")
            break
        except Exception as e:
            print(f"Error in live prompting: {str(e)}")
            continue

