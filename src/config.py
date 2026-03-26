import torch
import os
class Config:

    # ======================
    # DATA
    # ======================
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    train_path = os.path.join(BASE_DIR, "processed_data", "train.json")
    val_path = os.path.join(BASE_DIR, "processed_data", "val.json")
    test_path = os.path.join(BASE_DIR, "processed_data", "test.json")

    max_length = 64

    # ======================
    # MODEL
    # ======================
    model_name = "bert-base-uncased"

    hidden_size = 768
    persona_dim = 128
    content_dim = 128

    num_classes = 14   # from clustering result

    # ======================
    # TRAINING
    # ======================
    batch_size = 64
    epochs = 10
    lr = 1e-5

    lambda_adv = 0.1
    lambda_orth = 0.005

    # ======================
    # DEVICE
    # ======================
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ======================
    # CHECKPOINTS
    # ======================
    checkpoint_dir = "checkpoints/"