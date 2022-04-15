import torch
from torch.utils.data import Dataset


class IstsDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_input_length, max_target_length):
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.tokenizer = tokenizer
        self.data = []

        for index, row in dataframe.iterrows():
            # TODO: i don't know whether we should predict both values at the same time
            #   or maybe do two separate tasks
            x = f"stsb: sentence1: {row['x1']} sentence2: {row['x2']}"
            y = f"{row['y_type']} {row['y_score']}"
            self.data.append((x, y))

    def __getitem__(self, idx):
        x, y = self.data[idx]

        # cleaning data so as to ensure data is in string type
        x_text = " ".join(x.split())
        target_text = " ".join(y.split())

        x_tokenized = self._tokenize(x_text, self.max_input_length)
        y_tokenized = self._tokenize(target_text, self.max_target_length)

        x_ids = x_tokenized["input_ids"].squeeze().to(dtype=torch.long)
        x_mask = x_tokenized["attention_mask"].squeeze().to(dtype=torch.long)
        target_ids = y_tokenized["input_ids"].squeeze().to(dtype=torch.long)

        return x_ids, x_mask, target_ids

    def __len__(self):
        return len(self.data)

    def _tokenize(self, text, length):
        return self.tokenizer.batch_encode_plus(
            [text],
            max_length=length,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
