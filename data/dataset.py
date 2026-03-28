import os
import numpy as np
import torch
from datasets import load_dataset


DATA_DIR = "data"
DATA_PATH = os.path.join(DATA_DIR, "input.bin")


def prepare_data():
    """
    Downloads WikiText-2 and converts it to pure byte-level binary.
    """
    if os.path.exists(DATA_PATH):
        return

    os.makedirs(DATA_DIR, exist_ok=True)

    print("Downloading WikiText-2 from HuggingFace...")

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

    # ---- combine splits ----
    text = []
    for split in ["train", "validation", "test"]:
        text.extend(dataset[split]["text"])

    # ---- clean empty lines (important) ----
    text = [t for t in text if t.strip() != ""]

    full_text = "\n".join(text)

    # ---- convert to raw bytes ----
    data = np.frombuffer(full_text.encode("utf-8"), dtype=np.uint8)

    # ---- save as binary ----
    with open(DATA_PATH, "wb") as f:
        f.write(data.tobytes())

    print(f"Saved dataset to {DATA_PATH} ({len(data)} bytes)")


def load_data():
    prepare_data()

    data = np.memmap(DATA_PATH, dtype=np.uint8, mode="r")

    split = int(0.9 * len(data))
    train_data = data[:split]
    val_data = data[split:]

    return train_data, val_data


def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(len(data) - block_size, (batch_size,))

    x = torch.stack([
        torch.from_numpy(data[i:i+block_size].astype(np.int64))
        for i in ix
    ])

    y = torch.stack([
        torch.from_numpy(data[i+1:i+1+block_size].astype(np.int64))
        for i in ix
    ])

    return x.to(device), y.to(device)