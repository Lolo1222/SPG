import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import random
import torch
from accelerate.utils import set_seed
from data_utils import get_math_questions_from_local
import pandas as pd
from datasets import Dataset
    # {
    #   "id": 126336,
    #   "content": "<|mdm_mask|>",
    #   "single_word": false,
    #   "lstrip": false,
    #   "rstrip": false,
    #   "normalized": false,
    #   "special": true
    # },
# Constants for prompts
SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""    
def update_example(example, idx, masked_problems, masked_generations):
    """更新单个example"""
    example["problem"] = masked_problems[idx]
    example["generation"] = masked_generations[idx]
    
    # 更新prompt（基于新的problem）
    example["prompt"] = [
        {
            "role": "user",
            "content": f"{SYSTEM_PROMPT}\n\nYou are a math expert. You will be given a question to solve. Solve it step by step. Wrap the final answer in a \\boxed{{}}. \n\n{masked_problems[idx]}",
        },
    ]
    
    return example
    
def generate_masked_sequence(
    dataset,
    tokenizer,
    p_question_mask=0,
    p_gen_mask=0.15,
    seed=42,
    gen_masking_strategy="random",
    mask_id=126336,
):
    """
    Generate a masked sequence based on the given prompt using random masking.
    """
    # Set random seeds for reproducibility
    set_seed(seed)
    special_ids = set(tokenizer.all_special_ids)  # 我们不会把 special token 随机 mask
    questions_text = dataset['problem']
    generations_text = dataset['generation']
    number_of_questions = len(questions_text)
    # if mask_id == 126336:
    #     mask_content = "<|mdm_mask|>"
    # else:
    #     print("ERROR: Non-standard mask_id!")
    #     exit()

    # Tokenize questions
    if p_question_mask > 0:
        masked_questions_text_list = []

        for b in range(number_of_questions):
            enc = tokenizer(questions_text[b], add_special_tokens=False, return_tensors="pt")
            questions_ids = enc["input_ids"][0]  # 1D tensor, 长度 Lp
            Lp = questions_ids.size(0)
            # seq_len = Lp
            # Question masking: 对非 special token 以 p_prompt_mask 概率替换为 mask_id
            for i in range(Lp):
                tok = int(questions_ids[i].item())
                # tok = int(input_ids[b, i].item())
                if tok in special_ids:
                    continue
                if random.random() < p_question_mask:
                    questions_ids[i] = mask_id
                    # input_ids[b, i] = mask_id
            masked_question_text = tokenizer.decode(questions_ids.tolist(), skip_special_tokens=False)
            # masked_question_text = tokenizer.decode(input_ids[b].tolist(), skip_special_tokens=False)
            masked_questions_text_list.append(masked_question_text)
    else:
        masked_questions_text_list = questions_text


    # Tokenize generations
    if p_gen_mask > 0:
        if gen_masking_strategy == "random":
            masked_generations_text_list = []

            for b in range(number_of_questions):
                enc = tokenizer(generations_text[b], add_special_tokens=False, return_tensors="pt")
                generations_ids = enc["input_ids"][0]  # 1D tensor, 长度 Lg
                Lg = generations_ids.size(0)
                # Prompt masking: 对非 special token 以 p_prompt_mask 概率替换为 mask_id
                for i in range(Lg):
                    if random.random() < p_gen_mask:
                        generations_ids[i] = mask_id

                masked_generation_text = tokenizer.decode(generations_ids.tolist(), skip_special_tokens=False)
                masked_generations_text_list.append(masked_generation_text)
        else:
            print("ERROR: Unsupported gen_masking_strategy!")
            exit()

    else:
        print("No generation masking applied.")
        masked_generations_text_list = generations_text

    # 使用map函数批量修改
    modified_dataset = dataset.map(
        lambda example, idx: update_example(example, idx, masked_questions_text_list, masked_generations_text_list),
        with_indices=True,
    )
    return modified_dataset

if __name__ == "__main__":
    semi_offline_data_path = "/home/jwliu/dlm/SPG/deal_data/math_sample_3output.jsonl"
    dataset = get_math_questions_from_local(semi_offline_data_path)
    tokenizer=AutoTokenizer.from_pretrained("/home/jwliu/dlm/SPG/save_dir/hf_models/LLaDA-8B-Instruct", trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    new_dataset = generate_masked_sequence(
        dataset,
        tokenizer=tokenizer,
        p_question_mask=0,
        p_gen_mask=0.5,
        seed=42,
        mask_id=126336,
    )
    print(new_dataset[0])