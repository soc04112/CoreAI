import glob
import os

import orjson
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast


class STSDataset(Dataset):
  def __init__(self, file_path, tokenizer, max_len=128):
    self.tokenizer = tokenizer
    self.max_len = max_len

    self.data = pd.read_csv(file_path, sep='\t', quoting=3, dtype=str, keep_default_na=False).fillna('') #quoting=3은 따옴표 무시

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    row = self.data.iloc[idx]
    sentence1 = str(row.sentence1)
    sentence2= str(row.sentence2)
    score = float(row.score)

    try:
        inputs1 = self.tokenizer(sentence1, return_tensors='pt', max_length=self.max_len, padding='max_length', truncation=True)
    except Exception as e:
        print(f"[STS ERROR] sentence1 idx={idx}: {repr(sentence1)}")
        print("Exception:", e)
        # 건너뛰려면 리턴 None 혹은 임의값, 혹은 exception 재발생
        return None

    try:
        inputs2 = self.tokenizer(sentence2, return_tensors='pt', max_length=self.max_len, padding='max_length', truncation=True)
    except Exception as e:
        print(f"[STS ERROR] sentence2 idx={idx}: {repr(sentence2)}")
        print("Exception:", e)
        return None

    return {
        'input_ids1': inputs1['input_ids'].squeeze(0),
        'attention_mask1': inputs1['attention_mask'].squeeze(0),
        'input_ids2': inputs2['input_ids'].squeeze(0),
        'attention_mask2': inputs2['attention_mask'].squeeze(0),
        'score': torch.tensor(score, dtype=torch.float)
    }

class JsonlTextDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        tokenizer: PreTrainedTokenizerFast,
        block_size: int = 2048,
        stride: int = 256,
    ):
        self.examples = []

        bos_id = tokenizer.bos_token_id
        eos_id = tokenizer.eos_token_id
        pad_id = tokenizer.pad_token_id

        files = glob.glob(os.path.join(data_dir, "*.jsonl"))
        for path in files:
            with open(path, "rb") as f:
                for line_no, line in enumerate(tqdm(f, desc=f'Loading {os.path.basename(path)}', leave=True)):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        obj = orjson.loads(line)
                    except orjson.JSONDecodeError as e:
                        print(f"[JSONL ERROR] {path} line {line_no}: malformed JSON")
                        print("  >>", line)
                        continue

                    text = obj.get("text", "").strip()
                    if not text:
                        continue

                    # 토크나이저 에러 처리
                    try:
                        raw_ids = tokenizer.encode(text, add_special_tokens=False)
                    except Exception as e:
                        print(f"[TOKENIZER ERROR] {path} line {line_no}: cannot tokenize text")
                        print("  >>", repr(text))
                        print("Exception:", e)
                        continue

                    if len(raw_ids) == 0:
                        continue

                    raw_ids = [bos_id] + raw_ids + [eos_id]

                    # 2) 슬라이딩 윈도우
                    step = block_size - stride
                    for start in range(0, len(raw_ids), step):
                        chunk = raw_ids[start : start + block_size]

                        if len(chunk) < block_size:
                            chunk = chunk + [pad_id] * (block_size - len(chunk))

                        self.examples.append(torch.tensor(chunk, dtype=torch.long))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        input_ids = self.examples[idx]
        return {
            "input_ids": input_ids,
            "labels":    input_ids.clone()  # causal LM
        }
  