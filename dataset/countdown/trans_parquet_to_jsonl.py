import pandas as pd
import json
import numpy as np

def _json_default(o):
    # numpy scalar -> python scalar
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    # numpy array -> list
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    # pandas NA
    try:
        import pandas as pd
        if o is pd.NA:
            return None
    except Exception:
        pass
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

# 读取 Parquet 文件
df = pd.read_parquet('dataset/countdown/train-00000-of-00001.parquet')

# 获取前三行
df_head = df.head(3)

# 将前三行写入 JSONL 文件
with open('dataset/countdown/first_three.jsonl', 'w', encoding='utf-8') as f:
    for index, row in df_head.iterrows():
        json.dump(row.to_dict(), f, ensure_ascii=False, default=_json_default)
        f.write('\n')

print("前三行已保存到 dataset/countdown/first_three.jsonl")