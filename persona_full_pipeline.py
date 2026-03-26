import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
import umap
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

# ==========================
# CONFIG
# ==========================
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
PCA_DIM = 100
K_RANGE = range(5, 16)
RANDOM_STATE = 42

RESULTS_DIR = "results"
OUTPUT_DIR = "processed_data"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==========================
# STEP 1: LOAD + PARSE DATA
# ==========================
def load_and_parse_personachat():

    import glob
    import os

    print("Locating PersonaChat dataset...")

    dataset_path = r"C:\Users\varun\AppData\Local\Programs\Python\Python311\Lib\site-packages\data\Persona-Chat\personachat"

    train_file = os.path.join(dataset_path, "train_self_original.txt")
    valid_file = os.path.join(dataset_path, "valid_self_original.txt")

    speakers = []

    def process_file(file_path):

        current_persona = []
        current_dialogues = []

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:

                line = line.strip()
                if not line:
                    continue

                parts = line.split(" ", 1)
                if len(parts) != 2:
                    continue

                number, rest = parts

                # New episode starts when number == "1" AND it's a persona line
                if number == "1" and rest.startswith("your persona:"):

                    # Save previous episode if exists
                    if current_persona and current_dialogues:
                        speakers.append({
                            "persona_text": " ".join(current_persona).lower(),
                            "utterances": current_dialogues
                        })

                    # Reset for new episode
                    current_persona = []
                    current_dialogues = []

                # Persona lines
                if rest.startswith("your persona:"):
                    persona_text = rest.replace("your persona:", "").strip()
                    current_persona.append(persona_text)
                    continue

                # Dialogue lines
                if "\t" in rest:
                    dialog_parts = rest.split("\t")
                    if len(dialog_parts) >= 2:
                        text = dialog_parts[0].strip()
                        response = dialog_parts[1].strip()

                        current_dialogues.append(text)
                        current_dialogues.append(response)

        # Add last episode
        if current_persona and current_dialogues:
            speakers.append({
                "persona_text": " ".join(current_persona).lower(),
                "utterances": current_dialogues
            })

    process_file(train_file)
    process_file(valid_file)

    print("Total speakers parsed:", len(speakers))
    return speakers
# ==========================
# STEP 2: EMBEDDING
# ==========================
def embed_personas(speakers):
    personas = [sp["persona_text"] for sp in speakers]
    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = model.encode(personas, show_progress_bar=True)
    print("Embedding shape:", embeddings.shape)
    return embeddings


# ==========================
# STEP 3: PCA
# ==========================
def reduce_dimension(embeddings):
    pca = PCA(n_components=PCA_DIM, random_state=RANDOM_STATE)
    return pca.fit_transform(embeddings)


# ==========================
# STEP 4: SILHOUETTE SEARCH
# ==========================
def find_best_k(embeddings):
    scores = []
    best_k = None
    best_score = -1

    for k in K_RANGE:
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        score = silhouette_score(embeddings, labels)
        scores.append((k, score))
        print(f"K={k}, Silhouette={score:.4f}")

        if score > best_score:
            best_score = score
            best_k = k

    df = pd.DataFrame(scores, columns=["K", "Silhouette"])
    df.to_csv(os.path.join(RESULTS_DIR, "silhouette_scores.csv"), index=False)

    plt.figure()
    plt.plot(df["K"], df["Silhouette"], marker="o")
    plt.title("Silhouette Score vs K")
    plt.savefig(os.path.join(RESULTS_DIR, "silhouette_plot.png"))
    plt.close()

    print("Best K:", best_k)
    return best_k


# ==========================
# STEP 5: FINAL CLUSTERING
# ==========================
def assign_clusters(embeddings, speakers, k):

    kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = kmeans.fit_predict(embeddings)

    for i, sp in enumerate(speakers):
        sp["cluster_id"] = int(labels[i])

    # Save cluster mapping
    with open(os.path.join(RESULTS_DIR, "persona_clusters.json"), "w") as f:
        json.dump(speakers, f, indent=2)

    return labels


# ==========================
# STEP 6: UMAP VISUALIZATION
# ==========================
def visualize_umap(embeddings, labels):
    reducer = umap.UMAP(random_state=RANDOM_STATE)
    embedding_2d = reducer.fit_transform(embeddings)

    plt.figure(figsize=(8,6))
    plt.scatter(embedding_2d[:,0], embedding_2d[:,1], c=labels, cmap="tab20", s=5)
    plt.title("UMAP Visualization of Persona Clusters")
    plt.savefig(os.path.join(RESULTS_DIR, "umap_visualization.png"))
    plt.close()


# ==========================
# STEP 7: SPLIT SPEAKERS
# ==========================
def split_speakers(speakers):

    train_spk, temp_spk = train_test_split(
        speakers, test_size=0.2, random_state=RANDOM_STATE
    )

    val_spk, test_spk = train_test_split(
        temp_spk, test_size=0.5, random_state=RANDOM_STATE
    )

    return train_spk, val_spk, test_spk


# ==========================
# STEP 8: FLATTEN
# ==========================
def flatten_data(speaker_data):
    dataset = []
    for sp in speaker_data:
        for utt in sp["utterances"]:
            cleaned = utt.strip().lower()
            if len(cleaned) > 3:
                dataset.append({
                    "text": cleaned,
                    "label": sp["cluster_id"]
                })
    return dataset


# ==========================
# STEP 9: SAVE DATASETS
# ==========================
def save_dataset(data, filename):
    with open(os.path.join(OUTPUT_DIR, filename), "w") as f:
        json.dump(data, f, indent=2)


def plot_distribution(data, name):
    labels = [x["label"] for x in data]
    unique, counts = np.unique(labels, return_counts=True)

    plt.figure(figsize=(8,5))
    sns.barplot(x=unique, y=counts)
    plt.title(f"Class Distribution - {name}")
    plt.savefig(os.path.join(OUTPUT_DIR, f"class_distribution_{name}.png"))
    plt.close()


# ==========================
# MAIN
# ==========================
def main():

    speakers = load_and_parse_personachat()

    embeddings = embed_personas(speakers)

    reduced = reduce_dimension(embeddings)

    best_k = find_best_k(reduced)

    labels = assign_clusters(reduced, speakers, best_k)

    visualize_umap(reduced, labels)

    train_spk, val_spk, test_spk = split_speakers(speakers)

    train_data = flatten_data(train_spk)
    val_data = flatten_data(val_spk)
    test_data = flatten_data(test_spk)

    print("Train size:", len(train_data))
    print("Val size:", len(val_data))
    print("Test size:", len(test_data))

    save_dataset(train_data, "train.json")
    save_dataset(val_data, "val.json")
    save_dataset(test_data, "test.json")

    plot_distribution(train_data, "train")
    plot_distribution(test_data, "test")

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()