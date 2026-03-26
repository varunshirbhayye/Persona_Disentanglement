import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from dataset import get_dataloaders
from model1 import PersonaDisentangleModel
from config import Config


# ================================
# Setup
# ================================

os.makedirs("results", exist_ok=True)


# ================================
# Load trained model
# ================================

def load_model():

    model = PersonaDisentangleModel().to(Config.device)

    checkpoint = os.path.join(Config.checkpoint_dir, "model_epoch_5.pt")

    model.load_state_dict(
        torch.load(checkpoint, map_location=Config.device)
    )

    model.eval()

    return model


# ================================
# Extract embeddings
# ================================

def extract_embeddings(model, loader):

    persona_embeddings = []
    content_embeddings = []
    labels = []
    preds = []

    with torch.no_grad():

        for batch in loader:

            input_ids = batch["input_ids"].to(Config.device)
            attention_mask = batch["attention_mask"].to(Config.device)
            label = batch["label"].to(Config.device)

            z_p, z_c, logits, _ = model(input_ids, attention_mask)

            prediction = torch.argmax(logits, dim=1)

            persona_embeddings.append(z_p.cpu().numpy())
            content_embeddings.append(z_c.cpu().numpy())

            labels.append(label.cpu().numpy())
            preds.append(prediction.cpu().numpy())

    persona_embeddings = np.concatenate(persona_embeddings)
    content_embeddings = np.concatenate(content_embeddings)
    labels = np.concatenate(labels)
    preds = np.concatenate(preds)

    return persona_embeddings, content_embeddings, labels, preds


# ================================
# 1️⃣ Persona t-SNE
# ================================

def plot_persona_tsne(persona_embeddings, labels):

    print("Generating persona t-SNE...")

    tsne = TSNE(n_components=2, random_state=42)

    reduced = tsne.fit_transform(persona_embeddings[:4000])

    plt.figure(figsize=(8,6))

    plt.scatter(
        reduced[:,0],
        reduced[:,1],
        c=labels[:4000],
        cmap="tab20",
        s=6
    )

    plt.title("t-SNE Visualization of Persona Embeddings")

    plt.savefig("results/persona_tsne.png")

    plt.close()


# ================================
# 2️⃣ Confusion Matrix
# ================================

def plot_confusion_matrix(labels, preds):

    print("Generating confusion matrix...")

    cm = confusion_matrix(labels, preds)

    plt.figure(figsize=(10,8))

    sns.heatmap(cm, cmap="Blues")

    plt.title("Persona Classification Confusion Matrix")

    plt.xlabel("Predicted")
    plt.ylabel("True")

    plt.savefig("results/confusion_matrix.png")

    plt.close()


# ================================
# 3️⃣ Persona Leakage Test
# ================================

def persona_leakage_test(content_embeddings, labels):

    print("Running persona leakage test...")

    X_train, X_test, y_train, y_test = train_test_split(
        content_embeddings,
        labels,
        test_size=0.2,
        random_state=42
    )

    clf = LogisticRegression(max_iter=1000)

    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)

    acc = accuracy_score(y_test, preds)

    print("Persona leakage accuracy:", acc)

    plt.figure()

    plt.bar(["Leakage Accuracy"], [acc])

    plt.ylim(0,1)

    plt.title("Persona Leakage Test")

    plt.savefig("results/persona_leakage_test.png")

    plt.close()


# ================================
# 4️⃣ Persona vs Content Plot
# ================================

def plot_persona_vs_content(persona_embeddings, content_embeddings, labels):

    print("Generating persona vs content embedding comparison...")

    tsne = TSNE(n_components=2, random_state=42)

    p_reduced = tsne.fit_transform(persona_embeddings[:3000])
    c_reduced = tsne.fit_transform(content_embeddings[:3000])

    fig, axes = plt.subplots(1,2, figsize=(12,5))

    axes[0].scatter(
        p_reduced[:,0],
        p_reduced[:,1],
        c=labels[:3000],
        cmap="tab20",
        s=6
    )

    axes[0].set_title("Persona Embedding Space")

    axes[1].scatter(
        c_reduced[:,0],
        c_reduced[:,1],
        c=labels[:3000],
        cmap="tab20",
        s=6
    )

    axes[1].set_title("Content Embedding Space")

    plt.tight_layout()

    plt.savefig("results/persona_vs_content.png")

    plt.close()


# ================================
# Main
# ================================

def main():

    print("Loading model...")

    model = load_model()

    print("Loading test dataset...")

    _, _, test_loader = get_dataloaders()

    print("Extracting embeddings...")

    persona_embeddings, content_embeddings, labels, preds = extract_embeddings(
        model,
        test_loader
    )

    plot_persona_tsne(persona_embeddings, labels)

    plot_confusion_matrix(labels, preds)

    persona_leakage_test(content_embeddings, labels)

    plot_persona_vs_content(
        persona_embeddings,
        content_embeddings,
        labels
    )

    print("\nAnalysis complete.")
    print("All figures saved in results/ folder.")


if __name__ == "__main__":
    main()