# Sampling Analysis

This directory contains standalone analysis code for comparing diffusion generation against an autoregressive LLM.

## Main entrypoint

Run the blockwise distribution-shift analysis with:

```bash
/home/jwliu/miniconda3/envs/spg/bin/python sampling/diffusion_llm_shift_analysis.py \
  --dataset gsm8k \
  --num_samples 64 \
  --sample_seed 42 \
  --diffusion_model_path /path/to/diffusion/model \
  --llm_model_path /path/to/llm/model \
  --top_k 50 \
  --top_p 1.0 \
  --output_dir sampling/results \
  --run_name gsm8k_n2
```

## Outputs

- `details.jsonl`: sample-level records with generated text and per-block probabilities
- `sample_summary.csv`: flattened sample-level statistics
- `block_summary.csv`: flattened block-level statistics
- `run_summary.json`: run configuration and summary metrics
- `plots/`: histograms, scatter plots, and block-index curves

## Notes

- The default block size is 2.
- The script compares block log probabilities using log-prob sums.
- The script reports tokenizer vocabulary consistency in `run_summary.json` under `vocab_check`.
- Use `--strict_vocab_match` if you want to stop the run when vocab/id mapping differs.
- If the LLM tokenizer differs from the diffusion tokenizer, block alignment is performed by text spans.
- `top_k/top_p` are available for diffusion sampling; for low-noise comparison experiments, use `--top_k 0 --top_p 1.0`.
