import json
import os


LOG_PATH = "results/log.jsonl"


def log_step(model, step, loss):
    os.makedirs("results", exist_ok=True)

    with open(LOG_PATH, "a") as f:
        f.write(json.dumps({
            "model": model,
            "step": step,
            "loss": loss
        }) + "\n")