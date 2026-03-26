import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42
RESULTS_DIR = "results"
OUTPUT_DIR = "processed_data"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==========================
# STEP 1: LOAD CLUSTER LABELS
# ==========================
def load_cluster_labels():
    with open(os.path.join(RESULTS_DIR, "persona_clusters.json"), "r") as f:
        cluster_data = json.load(f)

    return cluster_data


# ==========================
# STEP 2: LOAD TXT FILES
# ==========================
def load_personachat_txt():
    dataset_path = r"C:\Users\varun\AppData\Local\Programs\Python\Python311\Lib\site-packages\data\Persona-Chat\personachat"

    train_file = os.path.join(dataset_path, "train_self_original.txt")
    valid_file = os.path.join(dataset_path, "valid_self_original.txt")

    return train_file, valid_file


# ==========================
# STEP 3: PARSE DIALOGUES
# ==========================
def parse_dialogues(train_file, valid_file):

    speakers = []
    speaker_dialogues = []

    def process_file(file_path):

        current_persona = []
        current_dialogues = []

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:

                line = line.strip()

                if not line:
                    continue

                # Split off leading number
                if " " in line:
                    number, rest = line.split(" ", 1)
                else:
                    continue

                # Persona lines
                if rest.startswith("your persona:"):
                    persona_text = rest.replace("your persona:", "").strip()
                    current_persona.append(persona_text)
                    continue

                # Dialogue lines (contain tab)
                if "\t" in rest:
                    parts = rest.split("\t")
                    if len(parts) == 2:
                        utt1 = parts[0].strip()
                        utt2 = parts[1].strip()
                        current_dialogues.append(utt1)
                        current_dialogues.append(utt2)

                # New episode starts when number == "1"
                if number == "1" and current_persona and current_dialogues:
                    speakers.append(" ".join(current_persona).lower())
                    speaker_dialogues.append(current_dialogues)
                    current_persona = []
                    current_dialogues = []

        # Add last episode
        if current_persona and current_dialogues:
            speakers.append(" ".join(current_persona).lower())
            speaker_dialogues.append(current_dialogues)

    process_file(train_file)
    process_file(valid_file)

    print("Total parsed speakers:", len(speakers))
    return speakers, speaker_dialogues
# ==========================
# STEP 4: BUILD DATASET
# ==========================
def build_dataset(cluster_data, speakers, dialogues):

    # cluster_data is ordered exactly like extracted personas
    cluster_ids = [item["cluster_id"] for item in cluster_data]

    if len(cluster_ids) != len(speakers):
        print("WARNING: Persona count mismatch!")
        print("Cluster count:", len(cluster_ids))
        print("Speaker count:", len(speakers))

    speaker_level_data = []

    min_length = min(len(cluster_ids), len(speakers))

    for i in range(min_length):
        speaker_level_data.append({
            "cluster_id": cluster_ids[i],
            "utterances": dialogues[i]
        })

    print("Total matched speakers:", len(speaker_level_data))
    return speaker_level_data

# ==========================
# STEP 5: SPLIT BY SPEAKER
# ==========================
def split_by_speaker(data):

    train_speakers, temp_speakers = train_test_split(
        data, test_size=0.2, random_state=RANDOM_STATE
    )

    val_speakers, test_speakers = train_test_split(
        temp_speakers, test_size=0.5, random_state=RANDOM_STATE
    )

    return train_speakers, val_speakers, test_speakers


# ==========================
# STEP 6: FLATTEN DATA
# ==========================
def flatten_data(speaker_data):
    dataset = []

    for speaker in speaker_data:
        for utt in speaker["utterances"]:
            cleaned = utt.strip().lower()
            if len(cleaned) > 3:
                dataset.append({
                    "text": cleaned,
                    "label": speaker["cluster_id"]
                })

    return dataset


# ==========================
# STEP 7: SAVE DATA
# ==========================
def save_json(data, filename):
    with open(os.path.join(OUTPUT_DIR, filename), "w") as f:
        json.dump(data, f, indent=2)


# ==========================
# STEP 8: PLOT DISTRIBUTION
# ==========================
def plot_distribution(data, name):

    labels = [item["label"] for item in data]
    unique, counts = np.unique(labels, return_counts=True)

    plt.figure(figsize=(8, 5))
    sns.barplot(x=unique, y=counts)
    plt.title(f"Class Distribution - {name}")
    plt.xlabel("Cluster ID")
    plt.ylabel("Count")
    plt.savefig(os.path.join(OUTPUT_DIR, f"class_distribution_{name}.png"))
    plt.close()


# ==========================
# MAIN
# ==========================
def main():

    cluster_data = load_cluster_labels()

    train_file, valid_file = load_personachat_txt()

    speakers, dialogues = parse_dialogues(train_file, valid_file)

    speaker_level_data = build_dataset(cluster_data, speakers, dialogues)

    train_spk, val_spk, test_spk = split_by_speaker(speaker_level_data)

    train_data = flatten_data(train_spk)
    val_data = flatten_data(val_spk)
    test_data = flatten_data(test_spk)

    save_json(train_data, "train.json")
    save_json(val_data, "val.json")
    save_json(test_data, "test.json")

    plot_distribution(train_data, "train")
    plot_distribution(test_data, "test")

    print("Phase 2 completed. Datasets saved.")


if __name__ == "__main__":
    main()