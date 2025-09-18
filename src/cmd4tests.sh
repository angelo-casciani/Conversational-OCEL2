#!/bin/bash

# python3 src/eval.py --modality live --rebuild_db True --llm_id meta-llama/Meta-Llama-3.1-8B-Instruct --max_new_tokens 1280

python3 src/eval.py --modality evaluation-global --rebuild_db True --llm_id meta-llama/Meta-Llama-3.1-8B-Instruct --max_new_tokens 768
python3 src/eval.py --modality evaluation-events --rebuild_db False --llm_id meta-llama/Meta-Llama-3.1-8B-Instruct --max_new_tokens 768
python3 src/eval.py --modality evaluation-objects --rebuild_db False --llm_id meta-llama/Meta-Llama-3.1-8B-Instruct --max_new_tokens 768
python3 src/eval.py --modality evaluation-ts --rebuild_db False --llm_id meta-llama/Meta-Llama-3.1-8B-Instruct --max_new_tokens 768
python3 src/eval.py --modality evaluation-all --rebuild_db False --llm_id meta-llama/Meta-Llama-3.1-8B-Instruct --max_new_tokens 768

python3 src/eval.py --modality evaluation-global --rebuild_db False --llm_id meta-llama/Meta-Llama-3-8B-Instruct --max_new_tokens 768
python3 src/eval.py --modality evaluation-events --rebuild_db False --llm_id meta-llama/Meta-Llama-3-8B-Instruct --max_new_tokens 768
python3 src/eval.py --modality evaluation-objects --rebuild_db False --llm_id meta-llama/Meta-Llama-3-8B-Instruct --max_new_tokens 768
python3 src/eval.py --modality evaluation-ts --rebuild_db False --llm_id meta-llama/Meta-Llama-3-8B-Instruct --max_new_tokens 768
python3 src/eval.py --modality evaluation-all --rebuild_db False --llm_id meta-llama/Meta-Llama-3-8B-Instruct --max_new_tokens 768

python3 src/eval.py --modality evaluation-global --rebuild_db False --llm_id meta-llama/Llama-3.2-1B-Instruct --max_new_tokens 768
python3 src/eval.py --modality evaluation-events --rebuild_db False --llm_id meta-llama/Llama-3.2-1B-Instruct --max_new_tokens 768
python3 src/eval.py --modality evaluation-objects --rebuild_db False --llm_id meta-llama/Llama-3.2-1B-Instruct --max_new_tokens 768
python3 src/eval.py --modality evaluation-ts --rebuild_db False --llm_id meta-llama/Llama-3.2-1B-Instruct --max_new_tokens 768
python3 src/eval.py --modality evaluation-all --rebuild_db False --llm_id meta-llama/Llama-3.2-1B-Instruct --max_new_tokens 768

python3 src/eval.py --modality evaluation-global --rebuild_db False --llm_id meta-llama/Llama-3.2-3B-Instruct --max_new_tokens 768
python3 src/eval.py --modality evaluation-events --rebuild_db False --llm_id meta-llama/Llama-3.2-3B-Instruct --max_new_tokens 768
python3 src/eval.py --modality evaluation-objects --rebuild_db False --llm_id meta-llama/Llama-3.2-3B-Instruct --max_new_tokens 768
python3 src/eval.py --modality evaluation-ts --rebuild_db False --llm_id meta-llama/Llama-3.2-3B-Instruct --max_new_tokens 768
python3 src/eval.py --modality evaluation-all --rebuild_db False --llm_id meta-llama/Llama-3.2-3B-Instruct --max_new_tokens 768

python3 src/eval.py --modality evaluation-global --rebuild_db False --llm_id mistralai/Mistral-7B-Instruct-v0.2 --max_new_tokens 768
python3 src/eval.py --modality evaluation-events --rebuild_db False --llm_id mistralai/Mistral-7B-Instruct-v0.2 --max_new_tokens 768
python3 src/eval.py --modality evaluation-objects --rebuild_db False --llm_id mistralai/Mistral-7B-Instruct-v0.2 --max_new_tokens 768
python3 src/eval.py --modality evaluation-ts --rebuild_db False --llm_id mistralai/Mistral-7B-Instruct-v0.2 --max_new_tokens 768
python3 src/eval.py --modality evaluation-all --rebuild_db False --llm_id mistralai/Mistral-7B-Instruct-v0.2 --max_new_tokens 768

python3 src/eval.py --modality evaluation-global --rebuild_db False --llm_id mistralai/Mistral-7B-Instruct-v0.3 --max_new_tokens 768
python3 src/eval.py --modality evaluation-events --rebuild_db False --llm_id mistralai/Mistral-7B-Instruct-v0.3 --max_new_tokens 768
python3 src/eval.py --modality evaluation-objects --rebuild_db False --llm_id mistralai/Mistral-7B-Instruct-v0.3 --max_new_tokens 768
python3 src/eval.py --modality evaluation-ts --rebuild_db False --llm_id mistralai/Mistral-7B-Instruct-v0.3 --max_new_tokens 768
python3 src/eval.py --modality evaluation-all --rebuild_db False --llm_id mistralai/Mistral-7B-Instruct-v0.3 --max_new_tokens 768

python3 src/eval.py --modality evaluation-global --rebuild_db False --llm_id mistralai/Mistral-Nemo-Instruct-2407 --max_new_tokens 768
python3 src/eval.py --modality evaluation-events --rebuild_db False --llm_id mistralai/Mistral-Nemo-Instruct-2407 --max_new_tokens 768
python3 src/eval.py --modality evaluation-objects --rebuild_db False --llm_id mistralai/Mistral-Nemo-Instruct-2407 --max_new_tokens 768
python3 src/eval.py --modality evaluation-ts --rebuild_db False --llm_id mistralai/Mistral-Nemo-Instruct-2407 --max_new_tokens 768
python3 src/eval.py --modality evaluation-all --rebuild_db False --llm_id mistralai/Mistral-Nemo-Instruct-2407 --max_new_tokens 768

python3 src/eval.py --modality evaluation-global --rebuild_db False --llm_id mistralai/Ministral-8B-Instruct-2410 --max_new_tokens 768
python3 src/eval.py --modality evaluation-events --rebuild_db False --llm_id mistralai/Ministral-8B-Instruct-2410 --max_new_tokens 768
python3 src/eval.py --modality evaluation-objects --rebuild_db False --llm_id mistralai/Ministral-8B-Instruct-2410 --max_new_tokens 768
python3 src/eval.py --modality evaluation-ts --rebuild_db False --llm_id mistralai/Ministral-8B-Instruct-2410 --max_new_tokens 768
python3 src/eval.py --modality evaluation-all --rebuild_db False --llm_id mistralai/Ministral-8B-Instruct-2410 --max_new_tokens 768

python3 src/eval.py --modality evaluation-global --rebuild_db False --llm_id Qwen/Qwen2.5-7B-Instruct --max_new_tokens 768
python3 src/eval.py --modality evaluation-events --rebuild_db False --llm_id Qwen/Qwen2.5-7B-Instruct --max_new_tokens 768
python3 src/eval.py --modality evaluation-objects --rebuild_db False --llm_id Qwen/Qwen2.5-7B-Instruct --max_new_tokens 768
python3 src/eval.py --modality evaluation-ts --rebuild_db False --llm_id Qwen/Qwen2.5-7B-Instruct --max_new_tokens 768
python3 src/eval.py --modality evaluation-all --rebuild_db False --llm_id Qwen/Qwen2.5-7B-Instruct --max_new_tokens 768

python3 src/eval.py --modality evaluation-global --rebuild_db False --llm_id gpt-4o-mini --max_new_tokens 768
python3 src/eval.py --modality evaluation-events --rebuild_db False --llm_id gpt-4o-mini --max_new_tokens 768
python3 src/eval.py --modality evaluation-objects --rebuild_db False --llm_id gpt-4o-mini --max_new_tokens 768
python3 src/eval.py --modality evaluation-ts --rebuild_db False --llm_id gpt-4o-mini --max_new_tokens 768
python3 src/eval.py --modality evaluation-all --rebuild_db False --llm_id gpt-4o-mini --max_new_tokens 768
