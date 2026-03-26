import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")

from transformers import logging
logging.set_verbosity_error()

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

from torch.cuda.amp import autocast, GradScaler

from dataset import get_dataloaders
from model1 import PersonaDisentangleModel
from losses import PersonaLoss
from config import Config


def evaluate(model, loader):

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for batch in loader:

            input_ids = batch["input_ids"].to(Config.device)
            attention_mask = batch["attention_mask"].to(Config.device)
            labels = batch["label"].to(Config.device)

            _, _, logits, _ = model(input_ids, attention_mask)

            preds = torch.argmax(logits, dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total


def train():

    os.makedirs("results", exist_ok=True)
    os.makedirs(Config.checkpoint_dir, exist_ok=True)

    train_loader, val_loader, _ = get_dataloaders()

    model = PersonaDisentangleModel().to(Config.device)

    optimizer = optim.AdamW(model.parameters(), lr=Config.lr)

    loss_fn = PersonaLoss()

    scaler = GradScaler()

    train_losses = []
    val_accuracies = []

    print("Training on:", Config.device)

    for epoch in range(Config.epochs):

        model.train()

        total_loss = 0

        loop = tqdm(train_loader)

        for batch in loop:

            input_ids = batch["input_ids"].to(Config.device)
            attention_mask = batch["attention_mask"].to(Config.device)
            labels = batch["label"].to(Config.device)

            optimizer.zero_grad()

            with autocast():

                z_p, z_c, persona_logits, adv_logits = model(
                    input_ids,
                    attention_mask
                )

                # Stage 1: Persona learning
                if epoch < 5:

                    cls_loss = loss_fn.ce(persona_logits, labels)
                    loss = cls_loss

                # Stage 2: Disentanglement
                else:

                    cls_loss = loss_fn.ce(persona_logits, labels)
                    adv_loss = loss_fn.ce(adv_logits, labels)
                    orth_loss = loss_fn.orthogonality_loss(z_p, z_c)

                    adv_weight = Config.lambda_adv * (epoch / Config.epochs)

                    loss = (
                        cls_loss
                        + adv_weight * adv_loss
                        + Config.lambda_orth * orth_loss
                    )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            loop.set_description(f"Epoch {epoch+1}")
            loop.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)

        val_acc = evaluate(model, val_loader)

        train_losses.append(avg_loss)
        val_accuracies.append(val_acc)

        print("\nEpoch", epoch + 1)
        print("Train Loss:", avg_loss)
        print("Validation Accuracy:", val_acc)

        torch.save(
            model.state_dict(),
            os.path.join(
                Config.checkpoint_dir,
                f"model_epoch_{epoch+1}.pt"
            )
        )

    plot_training(train_losses, val_accuracies)


def plot_training(train_losses, val_acc):

    plt.figure()

    plt.plot(train_losses)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.savefig("results/training_loss.png")

    plt.close()

    plt.figure()

    plt.plot(val_acc)
    plt.title("Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.savefig("results/validation_accuracy.png")

    plt.close()


if __name__ == "__main__":
    train()