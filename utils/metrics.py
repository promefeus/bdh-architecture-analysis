import json
from collections import defaultdict
import numpy as np


def load_logs(path="results/log.jsonl"):
    data = defaultdict(list)

    with open(path) as f:
        for line in f:
            d = json.loads(line)
            data[d["model"]].append((d["step"], d["loss"]))

    return data


def summarize():
    data = load_logs()

    summary = {}

    for model, values in data.items():
        losses = [v[1] for v in values]

        summary[model] = {
            "final_loss": losses[-1],
            "avg_last_50": np.mean(losses[-50:]),
            "std_last_50": np.std(losses[-50:]),
        }

    return summary


def compare():
    summary = summarize()

    print("\n===== RESULTS =====\n")

    for m, s in summary.items():
        print(
            f"{m:20} | final {s['final_loss']:.4f} | avg50 {s['avg_last_50']:.4f} | std {s['std_last_50']:.4f}"
        )

    return summary


def component_analysis():
    s = summarize()

    base = s["bdh_base"]

    print("\n===== COMPONENT IMPACT =====\n")

    def impact(name):
        return s[name]["avg_last_50"] - base["avg_last_50"]

    print(f"Multiplication  : {impact('bdh_nomul'):.4f}")
    print(f"Latent dim      : {impact('bdh_lowdim'):.4f}")
    print(f"Activation      : {impact('bdh_improved'):.4f}")