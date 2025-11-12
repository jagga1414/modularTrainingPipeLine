import torch
from torch.utils.data import Dataset
import torch.nn as nn
import pandas as pd


class Batch_D(nn.Module):
    def __init__(self):
        super().__init__()


class MyDataset(Dataset):
    def __init__(self, file_path, tokenizer):
        super().__init__()
        self.data = pd.read_csv(file_path, sep=",")
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        query = self.data["query"][idx]
        response = self.data["response"][idx]

        messages = [
            {"role": "user", "content": query},
            {"role": "assistant", "content": response}
        ]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False)

        return text


class MyCollate():
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        tokenized = self.tokenizer(batch, return_tensors="pt", padding=True)
        batch_data = Batch_D()
        batch_data.input_ids = tokenized.input_ids
        batch_data.attention_mask = tokenized.attention_mask
        batch_data.labels = tokenized.input_ids
        return batch_data
