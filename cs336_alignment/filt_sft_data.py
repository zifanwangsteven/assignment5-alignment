import torch
import json
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn


def filt_sft_data(data_path: str, save_path: str):
    filtered_data = []
    original_data_points = 0
    with open(data_path, 'r') as f:
        for line in f:
            original_data_points += 1
            data = json.loads(line)
            prompt = data['prompt']
            response = data['response']
            ground_truth = data['ground_truth']
            reward = r1_zero_reward_fn(response, ground_truth)
            if reward['reward'] == 1.0 :
                filtered_data.append(data)

    print(f"Original data size: {original_data_points}")
    print(f"Filtered data size: {len(filtered_data)}")
    with open(save_path, 'w') as f:
        for data in filtered_data:
            f.write(json.dumps(data) + '\n')


if __name__ == '__main__':
    filt_sft_data('data/MATH/sft.jsonl', 'data/MATH/sft_filtered.jsonl')