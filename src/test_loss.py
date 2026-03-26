from dataset import get_dataloaders
from model1 import PersonaDisentangleModel
from losses import PersonaLoss
from config import Config

train_loader, _, _ = get_dataloaders()

batch = next(iter(train_loader))

model = PersonaDisentangleModel().to(Config.device)

loss_fn = PersonaLoss()

input_ids = batch["input_ids"].to(Config.device)
attention_mask = batch["attention_mask"].to(Config.device)
labels = batch["label"].to(Config.device)

z_p, z_c, persona_logits, adv_logits = model(input_ids, attention_mask)

loss, cls_loss, adv_loss, orth_loss = loss_fn.compute(
    z_p, z_c, persona_logits, adv_logits, labels
)

print("Total loss:", loss.item())
print("Classification loss:", cls_loss.item())
print("Adversarial loss:", adv_loss.item())
print("Orthogonality loss:", orth_loss.item())