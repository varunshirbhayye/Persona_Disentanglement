from dataset import get_dataloaders

train_loader, val_loader, test_loader = get_dataloaders()

batch = next(iter(train_loader))

print("Input IDs shape:", batch["input_ids"].shape)
print("Attention mask shape:", batch["attention_mask"].shape)
print("Labels shape:", batch["label"].shape)