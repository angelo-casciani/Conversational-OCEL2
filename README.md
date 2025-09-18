# Conversational-OCEL2

This repository contains the code and data to reproduce the experiments from the paper *A Conversational Framework for Object-centric Event Logs Analysis*.
Conversational-OCEL2 is a conversational framework designed to facilitate process mining analysis over object-centric event logs following the OCEL 2.0 standard (in JSON). 
The approach leverages an architecture that combines Large Language Models (LLMs) with Retrieval Augmented Generation (RAG) to handle users' queries about OCEL 2.0 event logs and generate contextually relevant responses in natural language.

Additionally, the repository contains a dataset for evaluating the conversational framework, derived from a standard [OCEL 2.0 Procure-to-Pay (P2P) event log](https://www.ocel-standard.org/event-logs/simulations/p2p/).
This dataset functions as a benchmark for evaluating the effectiveness of conversational techniques in analyzing such event log from multiple perspectives.

## Structure of the repository

```
.
└── data/
│   ├── execution              # Knowledge extracted from the event log
│   └── ocel2-p2p.json         # Event log used for the evaluation
├── src/ 
│   ├── cmd4tests.sh
│   ├── eval.py                # Logic for Evaluation
│   ├── main.py                # Main logic for live interaction
│   ├── oracle.py              # Verification oracle for evaluation
│   ├── pipeline.py            # LLM pipeline setup
│   ├── preprocessing.py       # OCEL2 log preprocessing
│   ├── prompts.json           # LLM prompt templates
│   ├── utility.py             # Helper functions
│   └── vector_store.py        # Vector store management with Qdrant
├── tests/                     # sources for evaluation
│   ├── outputs/               # outputs of the live conversations
│   ├── test_sets/             # test sets employed during the evaluation
│   └── validation/            # evaluation results for each run
├── logs.zip                   # zipped folder with the tested log (to unzip)
├── .env                       # Environment variables (create/fill this)
├── requirements.txt           # Requirements to install
├── LICENSE                    # License file
└── README.md                  # This file
```

## Getting Started

For a quick setup and test run:

1. **Clone and setup:**
```bash
git clone https://github.com/angelo-casciani/Conversational-OCEL2
cd Conversational-OCEL2
```

Create a Python virtual environment.

**Option 1: Using venv**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Option 2: Using conda**
```bash
conda create --name xes2pddl python=3.10
conda activate xes2pddl
```

Install the required dependencies.

```bash
pip install -r requirements.txt
```

2. **Start Qdrant (vector database):**
This project uses [Docker](https://www.docker.com/) to run the vector store [Qdrant](https://qdrant.tech/).

Ensure Docker is installed and running on your system.

Download the latest Qdrant image from Docker Hub and  run the Qdrant service:
```bash
docker pull qdrant/qdrant
docker run -p 6333:6333 -p 6334:6334 \
    -v "$(pwd)/qdrant_storage:/qdrant/storage:z" \
    qdrant/qdrant
```

3. **Configure environment (create `.env` file):**

Create a `.env` file in the root directory and configure the following variables:

```env
HF_TOKEN=<your HuggingFace token>
DEEPSEEK_API_KEY=<your DeepSeek API key (if using DeepSeek models)>
GOOGLE_API_KEY=<your Gemini API key (if using Google models)>
OPENAI_API_KEY=<your OpenAI API key (if using OpenAI models)>
QDRANT_URL=127.0.0.0
QDRANT_GRPC_PORT=6334
```

**Required configurations:**
- **HF_TOKEN**: Your HuggingFace token for accessing open-source language models and embedding models
- **QDRANT_URL**: URL where Qdrant is running (default: `QDRANT_URL=127.0.0.0`)
- **QDRANT_GRPC_PORT**: gRPC port for Qdrant (default: `6334`)

The other configurations are optional.


4. **Run the application:**
```bash
cd src
python3 main.py --rebuild_db True
```

## LLMs Requirements

Please note that this software leverages open-source LLMs reported in the table:

| Model | HuggingFace Link |
|-----------|-----------|
| meta-llama/Meta-Llama-3-8B-Instruct | [HF link](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) |
| meta-llama/Meta-Llama-3.1-8B-Instruct | [HF link](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct) |
| meta-llama/Llama-3.2-1B-Instruct | [HF Link](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)|
| meta-llama/Llama-3.2-3B-Instruct | [HF link](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) |
| mistralai/Mistral-7B-Instruct-v0.2 | [HF link](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) |
| mistralai/Mistral-7B-Instruct-v0.3 | [HF link](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) |
| mistralai/Mistral-Nemo-Instruct-2407 | [HF link](https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407) |
| mistralai/Ministral-8B-Instruct-2410 | [HF link](https://huggingface.co/mistralai/Ministral-8B-Instruct-2410) |
| Qwen/Qwen2.5-7B-Instruct | [HF link](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) |
| google/gemma-2-9b-it | [HF link](https://huggingface.co/google/gemma-2-9b-it) |
| gpt-4o-mini | [OpenAI link](https://platform.openai.com/docs/models) |

Request in advance the permission to use each Llama model for your HuggingFace account.
Retrive your OpenAI API key to use the supported GPT model.

Please note that each of the selected models have specific requirements in terms of GPU availability.
It is recommended to have access to a GPU-enabled environment meeting at least the minimum requirements for these models to run the software effectively.

## Running the Project

### Basic Usage

Navigate to the project directory and run the project in the preferred configuration:

```bash
cd src
python3 main.py
```

### Enhanced Pipeline (Recommended)

The project now includes an enhanced pipeline with better error handling and performance. To use it:

```bash
python3 main.py --modality=live
```

### Evaluation Modes

To run evaluations for different aspects of the OCEL2 analysis:

**Global information evaluation:**
```bash
python3 main.py --llm_id Qwen/Qwen2.5-7B-Instruct --modality evaluation-global --max_new_tokens 512
```

**Events analysis evaluation:**
```bash
python3 main.py --llm_id Qwen/Qwen2.5-7B-Instruct --modality evaluation-events --max_new_tokens 512
```

**Objects analysis evaluation:**
```bash
python3 main.py --llm_id Qwen/Qwen2.5-7B-Instruct --modality evaluation-objects --max_new_tokens 512
```

**Timestamps analysis evaluation:**
```bash
python3 main.py --llm_id Qwen/Qwen2.5-7B-Instruct --modality evaluation-ts --max_new_tokens 512
```

**Complete evaluation (all categories):**
```bash
python3 main.py --llm_id Qwen/Qwen2.5-7B-Instruct --modality evaluation-all --max_new_tokens 512
```

### Database Rebuilding

If you need to rebuild the vector database (e.g., after changing the OCEL2 log or updating embeddings):

```bash
python3 main.py --rebuild_db=true 
```

### Configuration Parameters

The framework supports various configuration parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--embed_model_id` | `sentence-transformers/all-MiniLM-L12-v2` | Embedding model identifier |
| `--vector_dimension` | `384` | Vector space dimension (auto-detected if using enhanced pipeline) |
| `--llm_id` | `meta-llama/Meta-Llama-3.1-8B-Instruct` | LLM model identifier |
| `--model_max_length` | `128000` | Maximum input length (context window) |
| `--num_documents_in_context` | `5` | Number of documents retrieved for context |
| `--max_new_tokens` | `1280` | Maximum number of tokens to generate |
| `--batch_size` | `32` | Batch size for embedding processing |
| `--rebuild_db` | `false` | Whether to rebuild the vector index |
| `--use_enhanced_pipeline` | `true` | Use enhanced pipeline with better error handling |

### Custom OCEL2 Logs

It is possible to upload a different OCEL 2.0 log (in JSON format) in the *data* folder by replacing the provided *ocel2-p2p.json* log. After uploading a new log, rebuild the database:

```bash
python3 main.py --rebuild_db=true --log=your-new-log.json
```

### Usage with Enhanced Features
```bash
# Use enhanced pipeline (recommended)
python3 main.py 

# Automatic embedding dimension detection
python3 main.py --embed_model_id=sentence-transformers/all-mpnet-base-v2

# Better error recovery and batch processing
python3 main.py --rebuild_db=true --batch_size=16
```
