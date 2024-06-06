# Conversational-OCEL2


This project seeks to assess the capabilities of Large Language Models (LLMs) in the task of Entity Resolution.
The primary goal of the experiment is to verify whether the LLM can discern if an execution trace in XES format of a business process matches another one.

## Installing Requirements

To install the required Python packages for this project, you can use *pip* along with the *requirements.txt* file.

First, you need to clone the repository:
```bash
git clone https://github.com/AngeloC99/Trace_similarity_LLM
cd Trace_similarity_LLM
```

Run the following command to install the necessary dependencies using pip:
```bash
pip install -r requirements.txt
```

This command will read the requirements.txt file and install all the specified packages along with their dependencies.

## GPU Requirements
Please note that this software leverages open-source LLMs such as [Mistral](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) and [DeciLM](https://huggingface.co/Deci/DeciLM-7B), which have specific requirements in terms of GPU availability.
It is recommended to have access to a GPU-enabled environment to run the software effectively.

## Running the Project
Before running the project, it is necessary to replace in the *main.py* file the predefined string in the *hf_token* with your personal HuggingFace token.
Then, you can proceed by going in the project directory and executing the following command:
```bash
python3 main.py
```
