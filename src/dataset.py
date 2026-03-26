import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

from config import Config


class PersonaDataset(Dataset):

    def __init__(self, file_path):

        with open(file_path, "r") as f:
            self.data = json.load(f)

        self.tokenizer = AutoTokenizer.from_pretrained(Config.model_name)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        item = self.data[idx]

        text = item["text"]
        label = item["label"]

        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=Config.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long)
        }


def get_dataloaders():

    train_dataset = PersonaDataset(Config.train_path)
    val_dataset = PersonaDataset(Config.val_path)
    test_dataset = PersonaDataset(Config.test_path)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.batch_size,
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.batch_size,
        num_workers=4,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader