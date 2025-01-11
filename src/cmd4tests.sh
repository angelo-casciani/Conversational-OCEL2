#!/bin/bash

# python3 src/main.py --modality live --rebuild_db False --llm_id meta-llama/Meta-Llama-3.1-8B-Instruct --max_new_tokens 768
python3 src/main.py --modality evaluation-global --rebuild_db False --llm_id meta-llama/Meta-Llama-3.1-8B-Instruct --max_new_tokens 768
python3 src/main.py --modality evaluation-events --rebuild_db False --llm_id meta-llama/Meta-Llama-3.1-8B-Instruct --max_new_tokens 768
python3 src/main.py --modality evaluation-objects --rebuild_db False --llm_id meta-llama/Meta-Llama-3.1-8B-Instruct --max_new_tokens 768
python3 src/main.py --modality evaluation-ts --rebuild_db False --llm_id meta-llama/Meta-Llama-3.1-8B-Instruct --max_new_tokens 768
python3 src/main.py --modality evaluation-all --rebuild_db False --llm_id meta-llama/Meta-Llama-3.1-8B-Instruct --max_new_tokens 768