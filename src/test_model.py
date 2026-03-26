import torch
from dataset import get_dataloaders
from model1 import PersonaDisentangleModel
from config import Config


train_loader, _, _ = get_dataloaders()

batch = next(iter(train_loader))

model = PersonaDisentangleModel().to(Config.device)

input_ids = batch["input_ids"].to(Config.device)
attention_mask = batch["attention_mask"].to(Config.device)

z_p, z_c, persona_logits, adv_logits = model(input_ids, attention_mask)

print("Persona embedding:", z_p.shape)
print("Content embedding:", z_c.shape)
print("Persona logits:", persona_logits.shape)
print("Adversarial logits:", adv_logits.shape)