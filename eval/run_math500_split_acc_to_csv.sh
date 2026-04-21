python eval/batch_compute_math500_split_acc_to_csv.py \
  --results_root save_dir/new_eval_results \
  --model_dirs \
    math_swift_grpo_generated_num_t3_semi0.8_mask_answer_low_confidence_early0.95_20260327_045612 \
    math_swift_grpo_generated_num_t3_semi0.85_mask_answer_low_confidence_early0.95_20260327_045556 \
    math_swift_grpo_generated_num_t3_semi0.75_mask_answer_low_confidence_early0.95_20260328_161423 \
  --split_file dataset/math500_test_split_seed101.json \
  --output_csv save_dir/new_eval_results/math500_split_acc_summary.csv