export CUDA_VISIBLE_DEVICES=1
python sampling/llm_llm_shift_analysis.py \
  --dataset gsm8k \
  --num_samples 1024 \
  --sample_seed 42 \
  --model_a_path ~/OpenLLMs/Qwen/Qwen3-4B-Base \
  --model_b_path ~/OpenLLMs/Qwen/Qwen3-4B-Base \
  --tokenizer_path ~/OpenLLMs/Qwen/Qwen3-4B-Base \
  --max_new_tokens 256 \
  --top_k 50 \
  --temperature 0.9 \
  --top_p 1.0 \
  --do_sample \
  --output_dir sampling/results \
  --run_name qwen3_4b_vs_4b_gsm8k_ns1024
python sampling/llm_llm_shift_analysis.py \
  --dataset gsm8k \
  --num_samples 1024 \
  --sample_seed 42 \
  --model_a_path ~/OpenLLMs/Qwen/Qwen3-4B-Base \
  --model_b_path ~/OpenLLMs/Qwen/Qwen3-1.7B-Base \
  --tokenizer_path ~/OpenLLMs/Qwen/Qwen3-4B-Base \
  --max_new_tokens 256 \
  --top_k 50 \
  --temperature 0.9 \
  --top_p 1.0 \
  --do_sample \
  --output_dir sampling/results \
  --run_name qwen3_4b_vs_1.7b_gsm8k_ns1024



# python sampling/diffusion_llm_shift_analysis.py \
#   --dataset gsm8k \
#   --num_samples 1024 \
#   --sample_seed 42 \
#   --diffusion_model_path /home/jwliu/dlm/SPG/save_dir/hf_models/LLaDA-8B-Instruct \
#   --llm_model_path Qwen/Qwen3-4B-Base \
#   --block_size 8 \
#   --gen_length 256 \
#   --top_k 50 \
#   --top_p 1.0 \
#   --output_dir sampling/results \
#   --allow_tokenizer_mismatch \
#   --run_name llada_8b_vs_qwen3_4b_gsm8k_bs8_ns1024

# python sampling/diffusion_llm_shift_analysis.py \
#   --dataset gsm8k \
#   --num_samples 1024 \
#   --sample_seed 42 \
#   --diffusion_model_path /home/jwliu/dlm/SPG/save_dir/hf_models/LLaDA-8B-Instruct \
#   --llm_model_path Qwen/Qwen3-4B-Base \
#   --block_size 1 \
#   --gen_length 256 \
#   --top_k 50 \
#   --top_p 1.0 \
#   --output_dir sampling/results \
#   --allow_tokenizer_mismatch \
#   --run_name llada_8b_vs_qwen3_4b_gsm8k_bs1_ns1024
