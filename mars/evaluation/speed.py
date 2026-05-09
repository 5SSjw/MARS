import json
from transformers import AutoTokenizer
import numpy as np

tokenizer=AutoTokenizer.from_pretrained("/workspace/models/Qwen3-8B")
# jsonl_file = "/workspace/tmp1/MARS/mars/outputs/test_0103_2/humaneval_qwen3-8b_non-thinking_t1.0_r.jsonl"
# jsonl_file_base = "/workspace/tmp2/MARS/mars/outputs/test_1124/humaneval_qwen3-8b_baseline_t1.0.jsonl"
jsonl_file = "/workspace/tmp2/MARS/mars/final/qwen3-8b/sum_t6_t1.0_mars.jsonl"
# "ratio0.9" "mars"
jsonl_file_base = "/workspace/tmp2/MARS/mars/final/qwen3-8b/sum_t6_t1.0_baseline.jsonl"
data = []
with open(jsonl_file, 'r', encoding='utf-8') as file:
    for line in file:
        json_obj = json.loads(line)
        data.append(json_obj)



speeds=[]
for datapoint in data:
    qid=datapoint["question_id"]
    answer=datapoint["choices"][0]['turns']
    tokens=sum(datapoint["choices"][0]['new_tokens'])
    times = sum(datapoint["choices"][0]['wall_time'])
    speeds.append(tokens/times)


data = []
with open(jsonl_file_base, 'r', encoding='utf-8') as file:
    for line in file:
        json_obj = json.loads(line)
        data.append(json_obj)


total_time=0
total_token=0
speeds0=[]
for datapoint in data:
    qid=datapoint["question_id"]
    answer=datapoint["choices"][0]['turns']
    tokens = 0
    for i in answer:
        tokens += (len(tokenizer(i).input_ids) - 1)
    times = sum(datapoint["choices"][0]['wall_time'])
    speeds0.append(tokens / times)
    total_time+=times
    total_token+=tokens



print('speed',np.array(speeds).mean())
print('speed0',np.array(speeds0).mean())
print("ratio",np.array(speeds).mean()/np.array(speeds0).mean())

# avg acc len
file_path = jsonl_file

new_tokens_sum = 0
idxs_sum = 0

with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:

        data = json.loads(line.strip())
        if "choices" in data:
            for choice in data["choices"]:
                new_tokens_sum += sum(choice.get("new_tokens", []))
                idxs_sum += sum(choice.get("idxs", []))

result = new_tokens_sum / idxs_sum if idxs_sum != 0 else None
print("=================================================")
print(f"New Tokens Sum: {new_tokens_sum}")
print(f"Idxs Sum: {idxs_sum}")
print(f"Result (New Tokens / Idxs): {result}")