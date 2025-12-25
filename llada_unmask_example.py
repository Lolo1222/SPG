# 示例: llada_unmask_example.py
# 说明: 在本地运行前请确保 `transformers` 已安装并且模型目录存在: save_dir/hf_models/LLaDA-8B-Instruct/

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import random
import torch

# 使用 1 号 GPU（cuda:1）优先；若只有一张 GPU 则使用 cuda:0；若无 GPU 则回退到 CPU
if torch.cuda.is_available():
    device = torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0")
else:
    device = torch.device("cpu")

# 配置（按需修改）
model_dir = "/home/jwliu/dlm/SPG/save_dir/hf_models/LLaDA-8B-Instruct"
prompt_text = "<|startoftext|><|start_header_id|>user<|end_header_id|>\n\nYou are a math expert. You will be given a question to solve. Solve it step by step. Wrap the final answer in a \\boxed{}.\nRespond in the following format:\n<reasoning>\nYour reasoning here\n</reasoning>\n<answer>\n\\boxed{...}\n</answer>\n\n\nAn equilateral triangle is inscribed in the parabola $x^2 = 8y,$ such that one of the vertices of the triangle coincides with the vertex of the parabola.  Find the side length of this equilateral triangle.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n<reasoning>"
batch_size = 2
gen_length = 512   # 你想生成的长度
p_prompt_mask = 0.15
p_gen_mask = 0.4
temperature = 0.9
seed = 42

torch.manual_seed(seed)
random.seed(seed)

# 加载 tokenizer 和 模型（本地）
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModel.from_pretrained(model_dir, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device)
model.eval()


# 确定 mask id（回退到 eos）
mask_id = tokenizer.mask_token_id if tokenizer.mask_token_id is not None else tokenizer.eos_token_id
pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
vocab_size = tokenizer.vocab_size
special_ids = set(tokenizer.all_special_ids)  # 我们不会把 special token 随机 mask

# Tokenize prompt
enc = tokenizer(prompt_text, add_special_tokens=False, return_tensors="pt")
prompt_ids = enc["input_ids"][0]  # 1D tensor, 长度 Lp
Lp = prompt_ids.size(0)

# Prepare batch (batch_size copies, each will have independent random masking / init)
# total sequence length = prompt + gen_length
seq_len = Lp + gen_length
input_ids = torch.full((batch_size, seq_len), pad_id, dtype=torch.long, device=device)

for b in range(batch_size):
    # place prompt
    input_ids[b, :Lp] = prompt_ids.to(device).clone()

    # Prompt masking: 对非 special token 以 p_prompt_mask 概率替换为 mask_id
    for i in range(Lp):
        tok = int(input_ids[b, i].item())
        if tok in special_ids:
            continue
        if random.random() < p_prompt_mask:
            input_ids[b, i] = mask_id

    # Generation region initialization:
    # 使用给定的初始生成 tokens（若较短则重复或截断以填满 gen_length）
    # 可以通过修改 gen_init_text 来指定要作为初始生成区的文本，或直接提供 gen_init_ids = [id1, id2, ...]
    gen_init_text = "\nTo find the side length of the equilateral triangle inscribed in the parabola \(x^2 = 8y\), we start by identifying the vertex of the parabola. The standard form of a parabola that opens upwards is \(x^2 = 4py\), where the vertex is at \((0, \frac{p}{2})\). Comparing, we see that \(4p = 8\), so \(p = 2\). Therefore, the vertex of the parabola is at \((0, 2)\).\n\nNext, we need to determine the coordinates of the equilateral triangle. vertices. Since the triangle is inscribed in the parabola and one vertex coincides with the vertex, the other two vertices must lie on the parabola. Let the coordinates of the other two vertices be \((x, y)\) and \((-x, y)\). Since these points lie on the parabola, they satisfy \(x^2 = 8y\).\n\nThe the distance between the points \((x, y)\) and \((-x, y)\) is \(2x\). This distance is also the side length of the equilateral triangle. Therefore, we side length of the equilateral triangle is \(2x\).\n\nTo find \(x\), we use the fact that the vertices of the equilateral triangle are on the parabola and the the distance between the points \((x, y)\) and \((-x, y)\) is \(2x\). Since the vertices of the equilateral triangle are on the parabola, the distance between the points \((x, y)\) and \((-x, y)\) is \(2x\).\n\nSince the vertices of the equilateral triangle are on the parabola, the distance between the points \((x, y)\) and \((-x, y)\) is \(2x\). Since the vertices of the equilateral triangle are on the parabola, the distance between the points \((x, y)\) and \((-x, y)\) is \(2x\).\n\nThus, the vertices of the equilateral triangle are at \((0, 2)\), \((2, 2)\), and \((-2, 2)\). The side length of the equilateral triangle is \(2\sqrt{2}\).\n</reasoning>\n<answer>\n\boxed{4}\n</answer><|eot_id|><|endoftext|>"
    gen_init_ids = tokenizer(gen_init_text, add_special_tokens=False).get("input_ids", [])
    if len(gen_init_ids) == 0:
        # fallback to pad_id if tokenizer produced nothing
        gen_ids_tensor = torch.full((gen_length,), pad_id, dtype=torch.long, device=device)
    else:
        reps = (gen_length + len(gen_init_ids) - 1) // len(gen_init_ids)
        tiled = (gen_init_ids * reps)[:gen_length]
        gen_ids_tensor = torch.tensor(tiled, dtype=torch.long, device=device)

    input_ids[b, Lp:] = gen_ids_tensor

    # 然后以 p_gen_mask 概率把这些生成位置置为 mask（需要模型来 unmask）
    gen_mask = torch.rand(gen_length, device=device) < p_gen_mask
    input_ids[b, Lp:][gen_mask] = mask_id

# 注意: 我们保持 attention_mask 为 1，意味着模型可以看到 mask token 的 embedding 并据此进行预测
attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)

# Single forward pass -> 仅对当前 mask 位置采样并替换（一次 unmask）
with torch.no_grad():
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    # outputs 可能是 CausalLMOutputWithPast 或自定义输出
    logits = outputs.logits  # shape: (B, seq_len, vocab_size)

    # 我们只在被 mask 的位置上采样 new tokens
    is_mask = input_ids == mask_id  # (B, seq_len) bool

    # 准备结果 tensor
    sampled_ids = input_ids.clone()

    # 对每个被 mask 的位置进行带温度的采样（softmax + multinomial）
    B, S, V = logits.shape
    # 将 logits 除以温度后做 softmax
    probs = F.softmax(logits / temperature, dim=-1)  # (B, S, V)

    # 为效率仅对 mask 的位置采样
    mask_positions = is_mask.nonzero(as_tuple=False)  # shape: (N_mask, 2) 每行 (b, pos)
    if mask_positions.numel() > 0:
        # 按位置索引并采样
        for (b, pos) in mask_positions:
            p = probs[b, pos]  # (vocab_size,)
            new_tok = torch.multinomial(p, num_samples=1)  # (1,)
            sampled_ids[b, pos] = new_tok

# Decode masked input (before unmask) and perform two sampling rounds; write results to file (overwrite each run)
out_path = "llada_unmask_outputs.txt"
masked_texts = [tokenizer.decode(input_ids[b].tolist(), skip_special_tokens=False) for b in range(batch_size)]

# perform second independent sampling (two rounds) from the same logits/probs
probs = F.softmax(logits / temperature, dim=-1)  # (B, S, V)

sampled_ids_round1 = sampled_ids.clone()
sampled_ids_round2 = input_ids.clone()

mask_positions = is_mask.nonzero(as_tuple=False)
if mask_positions.numel() > 0:
    # Round 1 (we already sampled into sampled_ids)
    # Round 2: sample again from the same probs
    for (b, pos) in mask_positions:
        p = probs[b, pos]
        new_tok = torch.multinomial(p, num_samples=1)
        sampled_ids_round2[b, pos] = new_tok

# Write masked input and both rounds to file (overwrite)
with open(out_path, "w", encoding="utf-8") as f:
    f.write("Masked inputs (with mask tokens):\n")
    for b in range(batch_size):
        f.write(f"--- Sample {b+1} masked input ---\n")
        f.write(masked_texts[b] + "\n\n")

    f.write("Round 1 generations:\n")
    for b in range(batch_size):
        t = tokenizer.decode(sampled_ids_round1[b].tolist(), skip_special_tokens=False)
        f.write(f"--- Sample {b+1} round 1 ---\n")
        f.write(t + "\n\n")

    f.write("Round 2 generations:\n")
    for b in range(batch_size):
        t2 = tokenizer.decode(sampled_ids_round2[b].tolist(), skip_special_tokens=False)
        f.write(f"--- Sample {b+1} round 2 ---\n")
        f.write(t2 + "\n\n")

print(f"Wrote masked inputs and two rounds of generations to {out_path} (overwritten).")