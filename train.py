from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model
import torch
from rich import print
# import math
from grpo import GRPO
from math_verify import parse, verify
import re
import math_dapo

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


# SYSTEM_PROMPT = "Respond in following format:<thinking>{step by step reasoning}</thinking><answer>{number}</answer>"

instruction_following = "Let's think step by step and output the final answer within \\boxed{}."


def extract_solution(solution_str):
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
    assert solution is not None
    final_solution = solution.group(0)
    final_solution = final_solution.split('#### ')[1].replace(',', '')
    return final_solution
    

def prepare_dataset(dataset) -> Dataset:
    # extract_hash_answer = (
    #     lambda text: text.split("####")[1].strip() if "####" in text else None
    # )

    def process_example(example: dict):
        
        answer_raw = example["answer"]
        answer = extract_solution(answer_raw)
        question = example["question"]
        question = question + " " + instruction_following
        
        # answer = extract_hash_answer(example["answer"])
        # if answer is None:
        #     return None
        return {
            "prompt": [
                {"role": "system", "content": instruction_following},
                {"role": "user", "content": question},
            ],
            "answer": answer,
        }

    dataset = dataset.map(
        process_example,
        remove_columns=[
            col for col in dataset.column_names if col not in ["prompt", "answer"]
        ],
    )
    # dataset = dataset.filter(lambda x: x is not None)

    return dataset


model_name = "Qwen/Qwen2.5-0.5B-Instruct"
# small models are kind of dumb, they need a little push so using this fine-tuned model
# source: https://github.com/joey00072/nanoGRPO/blob/master/cold_start/cold_start_finetune.py
# you can totally use the base model, it will just take longer to converge
# model_name = "joey00072/Llama-3.2-1B-Instruct-cold-start-ft2"

# model_name = "unsloth/Llama-3.2-3B-Instruct"


model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.padding_side = "left"

lora_config = LoraConfig(
    r=32,
    lora_alpha=32,
)
model = get_peft_model(model, lora_config)
model = model.to(torch.bfloat16)




# def reward_func_len(sample: dict, s: str, *args, **kwargs):
#     return 4 - (len(s)/1000)

def math_verify_reward(sample: dict, s: str, *args, **kwargs):
    
    sample_answer = sample["answer"]
    response_end = s[-300:]
    
    res = math_dapo.compute_score(response_end, sample_answer, is_longcot=False, is_use_math_verify=True)
    
    return res
    
    # print("Sample answer:", sample_answer)
    # print("Response end:", response_end)
    
    # breakpoint()
    
    # ground_truth = parse(sample_answer)
    # pred_answer = parse(response_end)
    
    # return verify(ground_truth, pred_answer)


dataset = load_dataset("openai/gsm8k", "main")["train"]
dataset = prepare_dataset(dataset)

test_dataset = load_dataset("openai/gsm8k", "main")["test"]
test_dataset = prepare_dataset(test_dataset)

group_size = 8
micro_group_size = 8
lr = 5e-5
weight_decay = 1e-2
reward_functions = [
    # response_format_reward,
    math_verify_reward,
]

# print(model)

enable_wandb = True

trainer = GRPO(
    model,
    tokenizer=tokenizer,
    batch_size=4,
    max_new_tokens=1024,
    group_size=group_size,
    micro_group_size=micro_group_size,
    dataset=dataset,
    test_dataset=test_dataset,
    reward_functions=reward_functions,
    log_wandb=enable_wandb,
    lr=lr,
    weight_decay=weight_decay
)

eval_results = trainer.evaluate(num_samples=64)

# print(eval_results)
# breakpoint()

trainer.train()
