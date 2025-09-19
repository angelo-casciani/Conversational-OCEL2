import argparse
import csv
import numpy as np
import os
import torch
import random
from eval import SEED


def seed_everything(seed=10):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def load_process_representation(filename):
    try:
        filepath = os.path.join(os.path.dirname(__file__), '..', 'data', 'execution', filename)
        with open(filepath, 'r', encoding='utf-8') as file:
            file_content = file.read()
            return file_content
    except FileNotFoundError:
        print(f"Error: File {filename} not found")
        return ""
    except Exception as e:
        print(f"Error loading {filename}: {str(e)}")
        return ""


def load_csv_questions(filepath):
    questions = []
    try:
        with open(filepath, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip header
            for row_num, row in enumerate(reader, start=2):
                try:
                    if len(row) >= 2:
                        question, answer = row[0], row[1]
                        questions.append([question, answer])
                    else:
                        print(f"Warning: Row {row_num} in {filepath} has insufficient columns")
                except Exception as row_error:
                    print(f"Error processing row {row_num} in {filepath}: {str(row_error)}")
                    continue
    except FileNotFoundError:
        print(f"Error: CSV file {filepath} not found")
    except Exception as e:
        print(f"Error loading CSV {filepath}: {str(e)}")

    try:
        random.seed(SEED) # For reproducibility
    except ImportError:
        random.seed(10)
    random.shuffle(questions)
    return questions


def log_to_file(conversation, curr_datetime, info_run):
    try:
        output_dir = os.path.join(os.path.dirname(__file__), "..", "tests", "outputs")
        os.makedirs(output_dir, exist_ok=True)
        
        filepath = os.path.join(output_dir, f"output_{curr_datetime}.txt")
        with open(filepath, 'a', encoding='utf-8') as file:
            file.write('INFORMATION ON THE RUN\n\n')
            for key in info_run.keys():
                file.write(f"{key}: {info_run[key]}\n")
            file.write('\n-----------------------------------\n\n')
            file.write(conversation)
    except Exception as e:
        print(f"Error logging to file: {str(e)}")
