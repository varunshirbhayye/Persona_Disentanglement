import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import umap


# ==========================
# CONFIG
# ==========================
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
PCA_DIM = 100
K_RANGE = range(5, 16)
RANDOM_STATE = 42

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


# ==========================
# STEP 1: LOAD TXT FILES
# ==========================
def load_personachat_txt():
    dataset_path = r"C:\Users\varun\AppData\Local\Programs\Python\Python311\Lib\site-packages\data\Persona-Chat\personachat"

    train_file = os.path.join(dataset_path, "train_self_original.txt")
    valid_file = os.path.join(dataset_path, "valid_self_original.txt")

    if not os.path.exists(train_file):
        raise Exception("PersonaChat txt files not found.")

    return train_file, valid_file


# ==========================
# STEP 2: PARSE PERSONAS
# ==========================
def parse_personas(train_file, valid_file):
    personas = []

    def process_file(file_path):
        current_persona = []

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("1 your persona:"):
                    current_persona = []

                if "your persona:" in line:
                    persona_line = line.split("your persona:")[1].strip()
                    current_persona.append(persona_line)

                if line.startswith("1 ") and current_persona:
                    persona_text = " ".join(current_persona).lower()
                    personas.append(persona_text)
                    current_persona = []

    process_file(train_file)
    process_file(valid_file)

    print(f"Total personas extracted: {len(personas)}")
    return personas


# ==========================
# STEP 3: EMBEDDING
# ==========================
def embed_personas(personas):
    print("Encoding personas...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = model.encode(personas, show_progress_bar=True)
    print("Embedding shape:", embeddings.shape)
    return embeddings


# ==========================
# STEP 4: PCA REDUCTION
# ==========================
def reduce_dimension(embeddings):
    pca = PCA(n_components=PCA_DIM, random_state=RANDOM_STATE)
    reduced = pca.fit_transform(embeddings)
    return reduced


# ==========================
# STEP 5: FIND BEST K
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

    print(f"\nBest K selected: {best_k}")

    # Save scores
    df = pd.DataFrame(scores, columns=["K", "Silhouette"])
    df.to_csv(os.path.join(RESULTS_DIR, "silhouette_scores.csv"), index=False)

    # Plot silhouette
    plt.figure()
    plt.plot(df["K"], df["Silhouette"], marker="o")
    plt.title("Silhouette Score vs K")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Silhouette Score")
    plt.savefig(os.path.join(RESULTS_DIR, "silhouette_plot.png"))
    plt.close()

    return best_k


# ==========================
# STEP 6: FINAL CLUSTERING
# ==========================
def perform_clustering(embeddings, k, personas):
    kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = kmeans.fit_predict(embeddings)

    # Save cluster assignments
    cluster_data = []
    for i in range(len(personas)):
        cluster_data.append({
            "persona_text": personas[i],
            "cluster_id": int(labels[i])
        })

    pd.DataFrame(cluster_data).to_json(
        os.path.join(RESULTS_DIR, "persona_clusters.json"),
        orient="records",
        indent=2
    )

    return labels


# ==========================
# STEP 7: CLUSTER DISTRIBUTION
# ==========================
def plot_cluster_distribution(labels):
    unique, counts = np.unique(labels, return_counts=True)

    plt.figure(figsize=(8, 5))
    sns.barplot(x=unique, y=counts)
    plt.title("Cluster Size Distribution")
    plt.xlabel("Cluster ID")
    plt.ylabel("Number of Personas")
    plt.savefig(os.path.join(RESULTS_DIR, "cluster_distribution.png"))
    plt.close()


# ==========================
# STEP 8: UMAP VISUALIZATION
# ==========================
def visualize_embeddings(embeddings, labels):
    print("Generating UMAP visualization...")

    reducer = umap.UMAP(random_state=RANDOM_STATE)
    embedding_2d = reducer.fit_transform(embeddings)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        embedding_2d[:, 0],
        embedding_2d[:, 1],
        c=labels,
        cmap="tab20",
        s=5
    )
    plt.title("UMAP Visualization of Persona Clusters")
    plt.savefig(os.path.join(RESULTS_DIR, "embedding_2d_visualization.png"))
    plt.close()


# ==========================
# MAIN
# ==========================
def main():
    train_file, valid_file = load_personachat_txt()

    personas = parse_personas(train_file, valid_file)

    embeddings = embed_personas(personas)

    reduced = reduce_dimension(embeddings)

    best_k = find_best_k(reduced)

    labels = perform_clustering(reduced, best_k, personas)

    plot_cluster_distribution(labels)

    visualize_embeddings(reduced, labels)

    print("\nAll results saved in 'results/' folder.")


if __name__ == "__main__":
    main()