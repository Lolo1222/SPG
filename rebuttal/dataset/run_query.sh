python /home/jwliu/dlm/SPG/rebuttal/dataset/extract_math_gsm8k_from_query_file.py\
    --input /home/jwliu/midtrain/data/gpt-mini-math/认知应用-数学_1175_gpt-4.1-mini_bhwei2_1755653526346_queries_new_export_all_shlt_4.1-mini_1000.json \
    --output_dir /home/jwliu/dlm/SPG/rebuttal/dataset \
    --math_output query_file_math_train.jsonl \
    --gsm8k_output query_file_gsm8k_train.jsonl

python /home/jwliu/dlm/SPG/rebuttal/dataset/extract_math_gsm8k_from_query_file.py \
    --input /home/jwliu/midtrain/data/gpt-mini-math/认知应用-数学_1175_gpt-4.1-mini_bhwei2_1755653526346_queries_new_export_all_shlt_4.1-mini_1000.json \
    --output_dir /home/jwliu/dlm/SPG/rebuttal/dataset \
    --math_output query_file_math_train.jsonl \
    --gsm8k_output query_file_gsm8k_train.jsonl \
    --debug_unmatched_sample 30