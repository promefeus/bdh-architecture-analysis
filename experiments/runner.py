import os

from train.train import train


# ---- clear logs before full experiment ----
LOG_PATH = "results/log.jsonl"

if os.path.exists(LOG_PATH):
    os.remove(LOG_PATH)


# ---- run all models ----
def run_all():
    models = [
        "transformer",
        "bdh_base",
        "bdh_nomul",
        "bdh_lowdim",
        "bdh_improved",
    ]

    for name in models:
        print("\n" + "=" * 50)
        print(f"Running: {name}")
        print("=" * 50)

        train(name)


# ---- entry ----
if __name__ == "__main__":
    run_all()