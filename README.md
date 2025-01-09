# Conversational-OCEL2

This repository contains the code and data to reproduce the experiments from the paper *A Conversational Framework for Object-centric Event Logs Analysis*.
Conversational-OCEL2 is a conversational framework designed to facilitate process mining analysis over object-centric event logs following the OCEL 2.0 standard (in JSON). 
The approach leverages an architecture that combines Large Language Models (LLMs) with Retrieval Augmented Generation (RAG) to handle users' queries about OCEL 2.0 event logs and generate contextually relevant responses in natural language.

Additionally, the repository contains a dataset for evaluating the conversational framework, derived from a standard [OCEL 2.0 Procure-to-Pay (P2P) event log](https://www.ocel-standard.org/event-logs/simulations/p2p/).
This dataset functions as a benchmark for evaluating the effectiveness of conversational techniques in analyzing such event log from multiple perspectives.

## Installation

First, you need to clone the repository:
```bash
git clone https://github.com/angelo-casciani/Conversational-OCEL2
cd Conversational-OCEL2
```

(Optional) Set up a conda environment for the project.
```bash
conda create -n conv_ocel python=3.10 --yes
conda activate conv_ocel
```

To install the required Python packages for this project, you can use *pip* along with the *requirements.txt* file.
Run the following command to install the n
ecessary dependencies using pip:
```bash
pip install -r requirements.txt
```

This command will read the requirements.txt file and install all the specified packages along with their dependencies.

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
Before running the project, it is necessary to insert in the *.env* file:
- your personal *HuggingFace token* (request the permission to use the Llama models for this token in advance);
- the *URL* and the *gRPC port* of your *Qdrant* instance.

Eventually, you can proceed by going in the project directory and run the project in the preferred configuration.
```bash
python3 src/main.py
```

To run an evaluation for the simulation (*evaluation-simulation*), for the verification (*evaluation-verification*), or for the routing (*evaluation-routing*):
```bash
python3 main.py --llm_id Qwen/Qwen2.5-7B-Instruct --modality evaluation-simulation --max_new_tokens 512
```

It is possible to upload a different OCEL 2.0 log (in JSON) in the *data* folder, deleting the provided *ocel2-p2p.json* log.