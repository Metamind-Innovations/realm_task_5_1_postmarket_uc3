import json
import os

from src.utils.time_conversion import unix_to_datetime


def do_expert_knowledge_check(synthetic_samples_path):
    expert_knowledge_results = {}

    for filename in os.listdir(synthetic_samples_path):
        if filename.endswith('.json'):
            file_path = os.path.join(synthetic_samples_path, filename)
            with open(file_path, 'r') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    data = None
            print(file_path)

    return expert_knowledge_results
