import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import random
import re
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
import torch
from accelerate.utils import set_seed
from data_utils import get_math_questions_from_local, get_countdown_questions_from_local, get_sudoku_questions_new_from_local
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

def update_example_for_countdown(example, idx, masked_generations):
    """更新单个example"""
    example["generation"] = masked_generations[idx]
    return example
    
def generate_masked_sequence(
    dataset,
    tokenizer,
    model=None,
    p_question_mask=0,
    p_gen_mask=0.15,
    seed=42,
    gen_masking_strategy="random",
    mask_id=126336,
    batch_size=8,
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

        elif gen_masking_strategy == "mask_answer":
            masked_generations_text_list = []
            for b in range(number_of_questions):
                text = generations_text[b]
                enc = tokenizer(text, add_special_tokens=False, return_tensors="pt", return_offsets_mapping=True)
                input_ids = enc["input_ids"][0]
                offsets = enc["offset_mapping"][0]

                mask_indices = set()

                # Strategy 1: <answer>...</answer>
                matches = list(re.finditer(r"<answer>(.*?)</answer>", text, re.DOTALL))
                for match in matches:
                    start_char, end_char = match.span(1)
                    for idx, (t_start, t_end) in enumerate(offsets):
                        if t_end > start_char and t_start < end_char:
                            mask_indices.add(idx)

                # Strategy 2: \boxed{...}
                boxed_starts = [m.end() for m in re.finditer(r"\\boxed\{", text)]
                for start_content in boxed_starts:
                    depth = 1
                    curr = start_content
                    while curr < len(text):
                        if text[curr] == '{':
                            depth += 1
                        elif text[curr] == '}':
                            depth -= 1
                            if depth == 0:
                                # Found matching closing brace. Content is [start_content, curr)
                                # Mask the content inside \boxed{}
                                start_char = start_content
                                end_char = curr
                                for idx, (t_start, t_end) in enumerate(offsets):
                                    if t_end > start_char and t_start < end_char:
                                        mask_indices.add(idx)
                                break
                        curr += 1

                for idx in mask_indices:
                     input_ids[idx] = mask_id
                
                # Others
                for i in range(len(input_ids)):
                    if i not in mask_indices:
                        if random.random() < p_gen_mask:
                           input_ids[i] = mask_id
                           
                masked_generation_text = tokenizer.decode(input_ids.tolist(), skip_special_tokens=False)
                masked_generations_text_list.append(masked_generation_text)

        elif gen_masking_strategy == "low_confidence":
            # XXX(Lolo1222): prompt only for MATH!
            if model is None:
                raise ValueError("Model must be provided for low_confidence masking strategy.")
            
            masked_generations_text_list = []
            all_prompts = []
            for q in questions_text:
                content = f"{SYSTEM_PROMPT}\n\nYou are a math expert. You will be given a question to solve. Solve it step by step. Wrap the final answer in a \\boxed{{}}. \n\n{q}"
                messages = [{"role": "user", "content": content}]
                try:
                    p_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                except Exception:
                     p_str = content
                all_prompts.append(p_str)
            
            for i in range(0, number_of_questions, batch_size):
                batch_prompts = all_prompts[i : i + batch_size]
                batch_gens = generations_text[i : i + batch_size]
                
                prompt_enc = tokenizer(batch_prompts, return_tensors="pt", padding=True, add_special_tokens=False)
                gen_enc = tokenizer(batch_gens, return_tensors="pt", padding=True, add_special_tokens=False)
                
                bs_curr = len(batch_prompts)
                
                gen_input_ids = torch.full_like(gen_enc["input_ids"], mask_id)
                
                input_ids = torch.cat([prompt_enc["input_ids"], gen_input_ids], dim=1).to(model.device)
                attention_mask = torch.cat([prompt_enc["attention_mask"], gen_enc["attention_mask"]], dim=1).to(model.device)
                
                with torch.no_grad():
                    logits = model(input_ids, attention_mask=attention_mask).logits
                    
                prompt_len = prompt_enc["input_ids"].shape[1]
                gen_logits = logits[:, prompt_len:, :]
                
                target_ids = gen_enc["input_ids"].to(model.device)
                target_mask = gen_enc["attention_mask"].to(model.device)
                
                # Avoid log(0) if any issue, but float32 usually fine.
                log_probs = F.log_softmax(gen_logits.float(), dim=-1)
                token_log_probs = torch.gather(log_probs, 2, target_ids.unsqueeze(-1)).squeeze(-1)
                
                for b_idx in range(bs_curr):
                    length = target_mask[b_idx].sum().item()
                    num_to_mask = int(length * p_gen_mask)
                    
                    if num_to_mask > 0:
                        scores = token_log_probs[b_idx, :length]
                        _, indices = torch.topk(scores, k=num_to_mask, largest=False) 
                        
                        curr_ids = target_ids[b_idx].clone()[:length]
                        curr_ids[indices] = mask_id
                        masked_text = tokenizer.decode(curr_ids, skip_special_tokens=False)
                    else:
                        masked_text = batch_gens[b_idx]

                    masked_generations_text_list.append(masked_text)

        elif gen_masking_strategy == "mask_answer_low_confidence":
            if model is None:
                raise ValueError("Model must be provided for mask_answer_low_confidence masking strategy.")
            
            masked_generations_text_list = []
            all_prompts = []
            for q in questions_text:
                content = f"{SYSTEM_PROMPT}\n\nYou are a math expert. You will be given a question to solve. Solve it step by step. Wrap the final answer in a \\boxed{{}}. \n\n{q}"
                messages = [{"role": "user", "content": content}]
                try:
                    p_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                except Exception:
                     p_str = content
                all_prompts.append(p_str)
            
            for i in range(0, number_of_questions, batch_size):
                batch_prompts = all_prompts[i : i + batch_size]
                batch_gens = generations_text[i : i + batch_size]
                
                prompt_enc = tokenizer(batch_prompts, return_tensors="pt", padding=True, add_special_tokens=False)
                gen_enc = tokenizer(batch_gens, return_tensors="pt", padding=True, add_special_tokens=False, return_offsets_mapping=True)
                
                bs_curr = len(batch_prompts)
                
                gen_input_ids = torch.full_like(gen_enc["input_ids"], mask_id)
                
                input_ids = torch.cat([prompt_enc["input_ids"], gen_input_ids], dim=1).to(model.device)
                attention_mask = torch.cat([prompt_enc["attention_mask"], gen_enc["attention_mask"]], dim=1).to(model.device)
                
                with torch.no_grad():
                    logits = model(input_ids, attention_mask=attention_mask).logits
                    
                prompt_len = prompt_enc["input_ids"].shape[1]
                gen_logits = logits[:, prompt_len:, :]
                
                target_ids = gen_enc["input_ids"].to(model.device)
                target_mask = gen_enc["attention_mask"].to(model.device)
                
                # Avoid log(0)
                log_probs = F.log_softmax(gen_logits.float(), dim=-1)
                token_log_probs = torch.gather(log_probs, 2, target_ids.unsqueeze(-1)).squeeze(-1)
                
                offsets_batch = gen_enc["offset_mapping"]

                for b_idx in range(bs_curr):
                    length = target_mask[b_idx].sum().item()
                    text = batch_gens[b_idx]
                    offsets = offsets_batch[b_idx]

                    # 1. Find Answer indices
                    mask_indices = set()

                    # <answer>...</answer>
                    matches = list(re.finditer(r"<answer>(.*?)</answer>", text, re.DOTALL))
                    for match in matches:
                        start_char, end_char = match.span(1)
                        for idx, (t_start, t_end) in enumerate(offsets):
                            if idx < length:
                                if t_end > start_char and t_start < end_char:
                                    mask_indices.add(idx)

                    # \boxed{...}
                    boxed_starts = [m.end() for m in re.finditer(r"\\boxed\{", text)]
                    for start_content in boxed_starts:
                        depth = 1
                        curr = start_content
                        while curr < len(text):
                            if text[curr] == '{':
                                depth += 1
                            elif text[curr] == '}':
                                depth -= 1
                                if depth == 0:
                                    start_char = start_content
                                    end_char = curr
                                    for idx, (t_start, t_end) in enumerate(offsets):
                                        if idx < length:
                                            if t_end > start_char and t_start < end_char:
                                                mask_indices.add(idx)
                                    break
                            curr += 1

                    # 2. Low Confidence on the Rest
                    is_answer = torch.zeros(length, dtype=torch.bool, device=model.device)
                    for idx in mask_indices:
                        if idx < length:
                            is_answer[idx] = True
                    
                    num_rest = (~is_answer).sum().item()
                    num_to_mask = int(num_rest * p_gen_mask)
                    
                    final_mask_indices = is_answer.clone()
                    
                    if num_to_mask > 0:
                        scores = token_log_probs[b_idx, :length].clone()
                        scores[is_answer] = float('inf') # exclude answer tokens
                        _, indices = torch.topk(scores, k=num_to_mask, largest=False) 
                        final_mask_indices[indices] = True
                    
                    curr_ids = target_ids[b_idx].clone()[:length]
                    curr_ids[final_mask_indices] = mask_id
                    masked_text = tokenizer.decode(curr_ids, skip_special_tokens=False)

                    masked_generations_text_list.append(masked_text)

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

def generate_masked_sequence_for_countdown(
    dataset,
    tokenizer,
    model=None,
    p_question_mask=0,
    p_gen_mask=0.15,
    seed=42,
    gen_masking_strategy="random",
    mask_id=126336,
    batch_size=8,
):
    """
    Generate a masked sequence based on the given prompt using random masking.
    """
    # Set random seeds for reproducibility
    set_seed(seed)
    special_ids = set(tokenizer.all_special_ids)  # 我们不会把 special token 随机 mask
    prompt_text = dataset['prompt']
    generations_text = dataset['generation']
    number_of_questions = len(prompt_text)


    # Tokenize generations
    if p_gen_mask > 0:
        # if gen_masking_strategy == "random":
        #     masked_generations_text_list = []

        #     for b in range(number_of_questions):
        #         enc = tokenizer(generations_text[b], add_special_tokens=False, return_tensors="pt")
        #         generations_ids = enc["input_ids"][0]  # 1D tensor, 长度 Lg
        #         Lg = generations_ids.size(0)
        #         # Prompt masking: 对非 special token 以 p_prompt_mask 概率替换为 mask_id
        #         for i in range(Lg):
        #             if random.random() < p_gen_mask:
        #                 generations_ids[i] = mask_id

        #         masked_generation_text = tokenizer.decode(generations_ids.tolist(), skip_special_tokens=False)
        #         masked_generations_text_list.append(masked_generation_text)

        # elif gen_masking_strategy == "mask_answer":
        #     masked_generations_text_list = []
        #     for b in range(number_of_questions):
        #         text = generations_text[b]
        #         enc = tokenizer(text, add_special_tokens=False, return_tensors="pt", return_offsets_mapping=True)
        #         input_ids = enc["input_ids"][0]
        #         offsets = enc["offset_mapping"][0]

        #         mask_indices = set()

        #         # Strategy 1: <answer>...</answer>
        #         matches = list(re.finditer(r"<answer>(.*?)</answer>", text, re.DOTALL))
        #         for match in matches:
        #             start_char, end_char = match.span(1)
        #             for idx, (t_start, t_end) in enumerate(offsets):
        #                 if t_end > start_char and t_start < end_char:
        #                     mask_indices.add(idx)

        #         # Strategy 2: \boxed{...}
        #         boxed_starts = [m.end() for m in re.finditer(r"\\boxed\{", text)]
        #         for start_content in boxed_starts:
        #             depth = 1
        #             curr = start_content
        #             while curr < len(text):
        #                 if text[curr] == '{':
        #                     depth += 1
        #                 elif text[curr] == '}':
        #                     depth -= 1
        #                     if depth == 0:
        #                         # Found matching closing brace. Content is [start_content, curr)
        #                         # Mask the content inside \boxed{}
        #                         start_char = start_content
        #                         end_char = curr
        #                         for idx, (t_start, t_end) in enumerate(offsets):
        #                             if t_end > start_char and t_start < end_char:
        #                                 mask_indices.add(idx)
        #                         break
        #                 curr += 1

        #         for idx in mask_indices:
        #              input_ids[idx] = mask_id
                
        #         # Others
        #         for i in range(len(input_ids)):
        #             if i not in mask_indices:
        #                 if random.random() < p_gen_mask:
        #                    input_ids[i] = mask_id
                           
        #         masked_generation_text = tokenizer.decode(input_ids.tolist(), skip_special_tokens=False)
        #         masked_generations_text_list.append(masked_generation_text)

        # elif gen_masking_strategy == "low_confidence":
        #     # XXX(Lolo1222): prompt only for MATH!
        #     if model is None:
        #         raise ValueError("Model must be provided for low_confidence masking strategy.")
            
        #     masked_generations_text_list = []
        #     all_prompts = []
        #     for q in prompt_text:
        #         content = q[0]["content"]
        #         # content = f"{SYSTEM_PROMPT}\n\nYou are a math expert. You will be given a question to solve. Solve it step by step. Wrap the final answer in a \\boxed{{}}. \n\n{q}"
        #         messages = [{"role": "user", "content": content}]
        #         try:
        #             p_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        #         except Exception:
        #              p_str = content
        #         all_prompts.append(p_str)
            
        #     for i in range(0, number_of_questions, batch_size):
        #         batch_prompts = all_prompts[i : i + batch_size]
        #         batch_gens = generations_text[i : i + batch_size]
                
        #         prompt_enc = tokenizer(batch_prompts, return_tensors="pt", padding=True, add_special_tokens=False)
        #         gen_enc = tokenizer(batch_gens, return_tensors="pt", padding=True, add_special_tokens=False)
                
        #         bs_curr = len(batch_prompts)
                
        #         gen_input_ids = torch.full_like(gen_enc["input_ids"], mask_id)
                
        #         input_ids = torch.cat([prompt_enc["input_ids"], gen_input_ids], dim=1).to(model.device)
        #         attention_mask = torch.cat([prompt_enc["attention_mask"], gen_enc["attention_mask"]], dim=1).to(model.device)
                
        #         with torch.no_grad():
        #             logits = model(input_ids, attention_mask=attention_mask).logits
                    
        #         prompt_len = prompt_enc["input_ids"].shape[1]
        #         gen_logits = logits[:, prompt_len:, :]
                
        #         target_ids = gen_enc["input_ids"].to(model.device)
        #         target_mask = gen_enc["attention_mask"].to(model.device)
                
        #         # Avoid log(0) if any issue, but float32 usually fine.
        #         log_probs = F.log_softmax(gen_logits.float(), dim=-1)
        #         token_log_probs = torch.gather(log_probs, 2, target_ids.unsqueeze(-1)).squeeze(-1)
                
        #         for b_idx in range(bs_curr):
        #             length = target_mask[b_idx].sum().item()
        #             num_to_mask = int(length * p_gen_mask)
                    
        #             if num_to_mask > 0:
        #                 scores = token_log_probs[b_idx, :length]
        #                 _, indices = torch.topk(scores, k=num_to_mask, largest=False) 
                        
        #                 curr_ids = target_ids[b_idx].clone()[:length]
        #                 curr_ids[indices] = mask_id
        #                 masked_text = tokenizer.decode(curr_ids, skip_special_tokens=False)
        #             else:
        #                 masked_text = batch_gens[b_idx]

        #             masked_generations_text_list.append(masked_text)

        if gen_masking_strategy == "mask_answer_low_confidence":
            if model is None:
                raise ValueError("Model must be provided for mask_answer_low_confidence masking strategy.")
            
            masked_generations_text_list = []
            all_prompts = []
            for q in prompt_text:
                # print(q)
                content = q[0]["content"]
                messages = [{"role": "user", "content": content}]
                try:
                    p_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                except Exception:
                     p_str = content
                all_prompts.append(p_str)
            
            for i in range(0, number_of_questions, batch_size):
                batch_prompts = all_prompts[i : i + batch_size]
                batch_gens = generations_text[i : i + batch_size]
                
                prompt_enc = tokenizer(batch_prompts, return_tensors="pt", padding=True, add_special_tokens=False)
                gen_enc = tokenizer(batch_gens, return_tensors="pt", padding=True, add_special_tokens=False, return_offsets_mapping=True)
                
                bs_curr = len(batch_prompts)
                
                gen_input_ids = torch.full_like(gen_enc["input_ids"], mask_id)
                
                input_ids = torch.cat([prompt_enc["input_ids"], gen_input_ids], dim=1).to(model.device)
                attention_mask = torch.cat([prompt_enc["attention_mask"], gen_enc["attention_mask"]], dim=1).to(model.device)
                
                with torch.no_grad():
                    logits = model(input_ids, attention_mask=attention_mask).logits
                    
                prompt_len = prompt_enc["input_ids"].shape[1]
                gen_logits = logits[:, prompt_len:, :]
                
                target_ids = gen_enc["input_ids"].to(model.device)
                target_mask = gen_enc["attention_mask"].to(model.device)
                
                # Avoid log(0)
                log_probs = F.log_softmax(gen_logits.float(), dim=-1)
                token_log_probs = torch.gather(log_probs, 2, target_ids.unsqueeze(-1)).squeeze(-1)
                
                offsets_batch = gen_enc["offset_mapping"]

                for b_idx in range(bs_curr):
                    length = target_mask[b_idx].sum().item()
                    text = batch_gens[b_idx]
                    offsets = offsets_batch[b_idx]

                    # 1. Find Answer indices
                    mask_indices = set()

                    # <answer>...</answer>
                    matches = list(re.finditer(r"<answer>(.*?)</answer>", text, re.DOTALL))
                    for match in matches:
                        start_char, end_char = match.span(1)
                        for idx, (t_start, t_end) in enumerate(offsets):
                            if idx < length:
                                if t_end > start_char and t_start < end_char:
                                    mask_indices.add(idx)

                    # \boxed{...}
                    boxed_starts = [m.end() for m in re.finditer(r"\\boxed\{", text)]
                    for start_content in boxed_starts:
                        depth = 1
                        curr = start_content
                        while curr < len(text):
                            if text[curr] == '{':
                                depth += 1
                            elif text[curr] == '}':
                                depth -= 1
                                if depth == 0:
                                    start_char = start_content
                                    end_char = curr
                                    for idx, (t_start, t_end) in enumerate(offsets):
                                        if idx < length:
                                            if t_end > start_char and t_start < end_char:
                                                mask_indices.add(idx)
                                    break
                            curr += 1

                    # 2. Low Confidence on the Rest
                    is_answer = torch.zeros(length, dtype=torch.bool, device=model.device)
                    for idx in mask_indices:
                        if idx < length:
                            is_answer[idx] = True
                    
                    num_rest = (~is_answer).sum().item()
                    num_to_mask = int(num_rest * p_gen_mask)
                    
                    final_mask_indices = is_answer.clone()
                    
                    if num_to_mask > 0:
                        scores = token_log_probs[b_idx, :length].clone()
                        scores[is_answer] = float('inf') # exclude answer tokens
                        _, indices = torch.topk(scores, k=num_to_mask, largest=False) 
                        final_mask_indices[indices] = True
                    
                    curr_ids = target_ids[b_idx].clone()[:length]
                    curr_ids[final_mask_indices] = mask_id
                    masked_text = tokenizer.decode(curr_ids, skip_special_tokens=False)

                    masked_generations_text_list.append(masked_text)

        else:
            print("ERROR: Unsupported gen_masking_strategy!")
            exit()

    else:
        print("No generation masking applied.")
        masked_generations_text_list = generations_text

    # 使用map函数批量修改
    modified_dataset = dataset.map(
        lambda example, idx: update_example_for_countdown(example, idx, masked_generations_text_list),
        with_indices=True,
    )
    return modified_dataset

def generate_masked_sequence_for_sudoku_new(
    dataset,
    tokenizer,
    model=None,
    p_question_mask=0,
    p_gen_mask=0.15,
    seed=42,
    gen_masking_strategy="random",
    mask_id=126336,
    batch_size=8,
):
    """
    Generate a masked sequence based on the given prompt using random masking.
    """
    # Set random seeds for reproducibility
    set_seed(seed)
    special_ids = set(tokenizer.all_special_ids)  # 我们不会把 special token 随机 mask
    prompt_text = dataset['prompt']
    generations_text = dataset['generation']
    number_of_questions = len(prompt_text)


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

        elif gen_masking_strategy == "mask_answer":
            masked_generations_text_list = []
            for b in range(number_of_questions):
                text = generations_text[b]
                enc = tokenizer(text, add_special_tokens=False, return_tensors="pt", return_offsets_mapping=True)
                input_ids = enc["input_ids"][0]
                offsets = enc["offset_mapping"][0]

                mask_indices = set()

                # Strategy 1: <answer>...</answer>
                matches = list(re.finditer(r"<answer>(.*?)</answer>", text, re.DOTALL))
                for match in matches:
                    start_char, end_char = match.span(1)
                    for idx, (t_start, t_end) in enumerate(offsets):
                        if t_end > start_char and t_start < end_char:
                            mask_indices.add(idx)

                # Strategy 2: \boxed{...}
                boxed_starts = [m.end() for m in re.finditer(r"\\boxed\{", text)]
                for start_content in boxed_starts:
                    depth = 1
                    curr = start_content
                    while curr < len(text):
                        if text[curr] == '{':
                            depth += 1
                        elif text[curr] == '}':
                            depth -= 1
                            if depth == 0:
                                # Found matching closing brace. Content is [start_content, curr)
                                # Mask the content inside \boxed{}
                                start_char = start_content
                                end_char = curr
                                for idx, (t_start, t_end) in enumerate(offsets):
                                    if t_end > start_char and t_start < end_char:
                                        mask_indices.add(idx)
                                break
                        curr += 1

                for idx in mask_indices:
                     input_ids[idx] = mask_id
                
                # Others
                for i in range(len(input_ids)):
                    if i not in mask_indices:
                        if random.random() < p_gen_mask:
                           input_ids[i] = mask_id
                           
                masked_generation_text = tokenizer.decode(input_ids.tolist(), skip_special_tokens=False)
                masked_generations_text_list.append(masked_generation_text)

        # elif gen_masking_strategy == "low_confidence":
        #     # XXX(Lolo1222): prompt only for MATH!
        #     if model is None:
        #         raise ValueError("Model must be provided for low_confidence masking strategy.")
            
        #     masked_generations_text_list = []
        #     all_prompts = []
        #     for q in prompt_text:
        #         content = q[0]["content"]
        #         # content = f"{SYSTEM_PROMPT}\n\nYou are a math expert. You will be given a question to solve. Solve it step by step. Wrap the final answer in a \\boxed{{}}. \n\n{q}"
        #         messages = [{"role": "user", "content": content}]
        #         try:
        #             p_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        #         except Exception:
        #              p_str = content
        #         all_prompts.append(p_str)
            
        #     for i in range(0, number_of_questions, batch_size):
        #         batch_prompts = all_prompts[i : i + batch_size]
        #         batch_gens = generations_text[i : i + batch_size]
                
        #         prompt_enc = tokenizer(batch_prompts, return_tensors="pt", padding=True, add_special_tokens=False)
        #         gen_enc = tokenizer(batch_gens, return_tensors="pt", padding=True, add_special_tokens=False)
                
        #         bs_curr = len(batch_prompts)
                
        #         gen_input_ids = torch.full_like(gen_enc["input_ids"], mask_id)
                
        #         input_ids = torch.cat([prompt_enc["input_ids"], gen_input_ids], dim=1).to(model.device)
        #         attention_mask = torch.cat([prompt_enc["attention_mask"], gen_enc["attention_mask"]], dim=1).to(model.device)
                
        #         with torch.no_grad():
        #             logits = model(input_ids, attention_mask=attention_mask).logits
                    
        #         prompt_len = prompt_enc["input_ids"].shape[1]
        #         gen_logits = logits[:, prompt_len:, :]
                
        #         target_ids = gen_enc["input_ids"].to(model.device)
        #         target_mask = gen_enc["attention_mask"].to(model.device)
                
        #         # Avoid log(0) if any issue, but float32 usually fine.
        #         log_probs = F.log_softmax(gen_logits.float(), dim=-1)
        #         token_log_probs = torch.gather(log_probs, 2, target_ids.unsqueeze(-1)).squeeze(-1)
                
        #         for b_idx in range(bs_curr):
        #             length = target_mask[b_idx].sum().item()
        #             num_to_mask = int(length * p_gen_mask)
                    
        #             if num_to_mask > 0:
        #                 scores = token_log_probs[b_idx, :length]
        #                 _, indices = torch.topk(scores, k=num_to_mask, largest=False) 
                        
        #                 curr_ids = target_ids[b_idx].clone()[:length]
        #                 curr_ids[indices] = mask_id
        #                 masked_text = tokenizer.decode(curr_ids, skip_special_tokens=False)
        #             else:
        #                 masked_text = batch_gens[b_idx]

        #             masked_generations_text_list.append(masked_text)

        elif gen_masking_strategy == "mask_answer_low_confidence":
            if model is None:
                raise ValueError("Model must be provided for mask_answer_low_confidence masking strategy.")
            
            masked_generations_text_list = []
            all_prompts = []
            for q in prompt_text:
                # print(q)
                content = q[0]["content"]
                messages = [{"role": "user", "content": content}]
                try:
                    p_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                except Exception:
                     p_str = content
                all_prompts.append(p_str)
            
            for i in range(0, number_of_questions, batch_size):
                batch_prompts = all_prompts[i : i + batch_size]
                batch_gens = generations_text[i : i + batch_size]
                
                prompt_enc = tokenizer(batch_prompts, return_tensors="pt", padding=True, add_special_tokens=False)
                gen_enc = tokenizer(batch_gens, return_tensors="pt", padding=True, add_special_tokens=False, return_offsets_mapping=True)
                
                bs_curr = len(batch_prompts)
                
                gen_input_ids = torch.full_like(gen_enc["input_ids"], mask_id)
                
                input_ids = torch.cat([prompt_enc["input_ids"], gen_input_ids], dim=1).to(model.device)
                attention_mask = torch.cat([prompt_enc["attention_mask"], gen_enc["attention_mask"]], dim=1).to(model.device)
                
                with torch.no_grad():
                    logits = model(input_ids, attention_mask=attention_mask).logits
                    
                prompt_len = prompt_enc["input_ids"].shape[1]
                gen_logits = logits[:, prompt_len:, :]
                
                target_ids = gen_enc["input_ids"].to(model.device)
                target_mask = gen_enc["attention_mask"].to(model.device)
                
                # Avoid log(0)
                log_probs = F.log_softmax(gen_logits.float(), dim=-1)
                token_log_probs = torch.gather(log_probs, 2, target_ids.unsqueeze(-1)).squeeze(-1)
                
                offsets_batch = gen_enc["offset_mapping"]

                for b_idx in range(bs_curr):
                    length = target_mask[b_idx].sum().item()
                    text = batch_gens[b_idx]
                    offsets = offsets_batch[b_idx]

                    # 1. Find Answer indices
                    mask_indices = set()

                    # <answer>...</answer>
                    matches = list(re.finditer(r"<answer>(.*?)</answer>", text, re.DOTALL))
                    for match in matches:
                        start_char, end_char = match.span(1)
                        for idx, (t_start, t_end) in enumerate(offsets):
                            if idx < length:
                                if t_end > start_char and t_start < end_char:
                                    mask_indices.add(idx)

                    # \boxed{...}
                    boxed_starts = [m.end() for m in re.finditer(r"\\boxed\{", text)]
                    for start_content in boxed_starts:
                        depth = 1
                        curr = start_content
                        while curr < len(text):
                            if text[curr] == '{':
                                depth += 1
                            elif text[curr] == '}':
                                depth -= 1
                                if depth == 0:
                                    start_char = start_content
                                    end_char = curr
                                    for idx, (t_start, t_end) in enumerate(offsets):
                                        if idx < length:
                                            if t_end > start_char and t_start < end_char:
                                                mask_indices.add(idx)
                                    break
                            curr += 1

                    # 2. Low Confidence on the Rest
                    is_answer = torch.zeros(length, dtype=torch.bool, device=model.device)
                    for idx in mask_indices:
                        if idx < length:
                            is_answer[idx] = True
                    
                    num_rest = (~is_answer).sum().item()
                    num_to_mask = int(num_rest * p_gen_mask)
                    
                    final_mask_indices = is_answer.clone()
                    
                    if num_to_mask > 0:
                        scores = token_log_probs[b_idx, :length].clone()
                        scores[is_answer] = float('inf') # exclude answer tokens
                        _, indices = torch.topk(scores, k=num_to_mask, largest=False) 
                        final_mask_indices[indices] = True
                    
                    curr_ids = target_ids[b_idx].clone()[:length]
                    curr_ids[final_mask_indices] = mask_id
                    masked_text = tokenizer.decode(curr_ids, skip_special_tokens=False)

                    masked_generations_text_list.append(masked_text)

        else:
            print("ERROR: Unsupported gen_masking_strategy!")
            exit()

    else:
        print("No generation masking applied.")
        masked_generations_text_list = generations_text

    # 使用map函数批量修改
    modified_dataset = dataset.map(
        lambda example, idx: update_example_for_countdown(example, idx, masked_generations_text_list),
        with_indices=True,
    )
    return modified_dataset

if __name__ == "__main__":
    semi_offline_data_path = "dataset/sudoku/llada_math_generations_rank0_converted.jsonl"
    dataset = get_sudoku_questions_new_from_local(semi_offline_data_path)
    tokenizer=AutoTokenizer.from_pretrained("/root/Models/LLaDA-8B-Instruct", trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model_path = "/root/Models/LLaDA-8B-Instruct"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )    
    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
    ).to(device)

    new_dataset = generate_masked_sequence_for_sudoku_new(
        dataset,
        tokenizer=tokenizer,
        model=model,
        p_question_mask=0,
        p_gen_mask=0.95,
        seed=42,
        mask_id=126336,
        gen_masking_strategy='mask_answer_low_confidence',
    )
    # semi_offline_data_path = "dataset/llada_countdown_generation_3_converted.jsonl"
    # dataset = get_countdown_questions_from_local(semi_offline_data_path)
    # tokenizer=AutoTokenizer.from_pretrained("/root/Models/LLaDA-8B-Instruct", trust_remote_code=True)
    # tokenizer.pad_token = tokenizer.eos_token
    # model_path = "/root/Models/LLaDA-8B-Instruct"
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=torch.bfloat16,
    # )    
    # model = AutoModel.from_pretrained(
    #     model_path,
    #     trust_remote_code=True,
    #     torch_dtype=torch.bfloat16,
    #     quantization_config=bnb_config,
    # ).to(device)

    # new_dataset = generate_masked_sequence_for_countdown(
    #     dataset,
    #     tokenizer=tokenizer,
    #     model=model,
    #     p_question_mask=0,
    #     p_gen_mask=0.01,
    #     seed=42,
    #     mask_id=126336,
    #     gen_masking_strategy='mask_answer',
    # )
    print(new_dataset[0])
    print(new_dataset[1])