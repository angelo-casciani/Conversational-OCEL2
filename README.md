# Conversational-OCEL2


## Installing Requirements

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

## GPU Requirements
Please note that this software leverages open-source LLMs such as [Llama 2 7B](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf), [Llama 2 13B](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf), and [Llama 3 8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
), which have specific requirements in terms of GPU availability.
It is recommended to have access to a GPU-enabled environment to run the software effectively.

## Running the Project
Before running the project, it is necessary to insert in the *.env* file:
- your personal HuggingFace token (request the permission to use the Llama models for this token in advance);
- the URL and the gRPC port of your Qdrant instance.

Eventually, you can proceed by going in the project directory and executing the following command:
```bash
python3 main.py
```
It is possible to upload a different OCEL 2.0 log (in JSON) in the *data* folder, deleting the provided *ocel2-p2p.json* log.