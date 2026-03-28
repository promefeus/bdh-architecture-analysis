import torch

from configs.config import ModelConfig, TrainConfig
from data.dataset import load_data, get_batch
from utils.logger import log_step

# models
from models.transformer import Transformer
from models.bdh_base import BDH as BDHBase
from models.bdh_nomul import BDH as BDHNoMul
from models.bdh_lowdim import BDH as BDHLowDim
from models.bdh_improved import BDH as BDHImproved


# ---- model selector ----
def get_model(name, config):
    if name == "transformer":
        return Transformer(config)

    elif name == "bdh_base":
        return BDHBase(config)

    elif name == "bdh_nomul":
        return BDHNoMul(config)

    elif name == "bdh_lowdim":
        return BDHLowDim(config)

    elif name == "bdh_improved":
        return BDHImproved(config)

    else:
        raise ValueError(f"Unknown model: {name}")


# ---- validation ----
@torch.no_grad()
def estimate_loss(model, data, train_cfg, device, eval_iters=20):
    model.eval()
    losses = []

    for _ in range(eval_iters):
        x, y = get_batch(
            data,
            train_cfg.block_size,
            train_cfg.batch_size,
            device,
        )
        _, loss = model(x, y)
        losses.append(loss.item())

    return sum(losses) / len(losses)


# ---- training ----
def train(model_name="bdh_base"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_cfg = ModelConfig()
    train_cfg = TrainConfig()

    torch.manual_seed(train_cfg.seed)

    # ---- data ----
    train_data, val_data = load_data()

    # ---- model ----
    model = get_model(model_name, model_cfg).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
    )

    # ---- training loop ----
    for step in range(train_cfg.max_iters):
        model.train()

        x, y = get_batch(
            train_data,
            train_cfg.block_size,
            train_cfg.batch_size,
            device,
        )

        _, loss = model(x, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ---- logging + validation ----
        if step % train_cfg.log_freq == 0:
            val_loss = estimate_loss(model, val_data, train_cfg, device)

            print(
                f"[{model_name}] step {step} | train {loss.item():.4f} | val {val_loss:.4f}"
            )

            log_step(model_name, step, loss.item())

    return model


# ---- entry ----
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="bdh_base")
    args = parser.parse_args()

    train(args.model)