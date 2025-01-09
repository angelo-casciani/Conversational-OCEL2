#!/bin/bash

# Evaluation of Llama 3.1 Instruct on Road Traffic Fine Management Log ONLY concept names
python3 src/main.py --log Road_Traffic_Fine_Management_Process.xes --modality evaluation-concept_names --rebuild_db_and_tests True --log_gap 3 --llm_id gpt-4o-mini