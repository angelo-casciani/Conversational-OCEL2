# Conversational-OCEL2

This repository contains the code and data to reproduce the experiments from the paper *A Conversational Framework for Object-centric Event Logs Analysis*.
Conversational-OCEL2 is a conversational framework designed to facilitate process mining analysis over object-centric event logs following the OCEL 2.0 standard (in JSON). 
The approach leverages an architecture that combines Large Language Models (LLMs) with Retrieval Augmented Generation (RAG) to handle users' queries about OCEL 2.0 event logs and generate contextually relevant responses in natural language.

Additionally, the repository contains a dataset for evaluating the conversational framework, derived from a standard [OCEL 2.0 Procure-to-Pay (P2P) event log](https://www.ocel-standard.org/event-logs/simulations/p2p/).
This dataset functions as a benchmark for evaluating the effectiveness of conversational techniques in analyzing such event log from multiple perspectives.

## Installation

To install the required Python packages for this project, you can use *pip* along with the *requirements.txt* file.

First, you need to clone the repository:
```bash
git clone https://github.com/angelo-casciani/Conversational-OCEL2
cd Conversational-OCEL2
```

Run the following command to install the necessary dependencies using pip:
```bash
pip install -r requirements.txt
```

This command will read the requirements.txt file and install all the specified packages along with their dependencies.

## LLMs Requirements

Please note that this software leverages open-source LLMs reported in the table:

| Model | HuggingFace Link |
|-----------|-----------|
| Llama 2 7B | [HF link](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) |
| Llama 2 13B | [HF link](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf) |
| Llama 3 8B | [HF link](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) |

Request in advance the permission to use each Llama model for your HuggingFace account.

Please note that each of the selected models have specific requirements in terms of GPU availability.
It is recommended to have access to a GPU-enabled environment meeting at least the minimum requirements for these models to run the software effectively.

## Running the Project
Before running the project, it is necessary to insert in the *.env* file:
- your personal HuggingFace token (request the permission to use the Llama models for this token in advance);
- the URL and the gRPC port of your Qdrant instance.

Eventually, you can proceed by going in the project directory and executing the following command:
```bash
python3 main.py
```

It is possible to upload a different OCEL 2.0 log (in JSON) in the *data* folder, deleting the provided *ocel2-p2p.json* log.