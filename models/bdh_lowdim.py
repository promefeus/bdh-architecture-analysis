import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, n_head, n_embd):
        super().__init__()
        self.n_head = n_head
        self.head_dim = n_embd // n_head

    def forward(self, Q, K, V):
        B, nh, T, D = Q.shape

        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(D)
        mask = torch.tril(torch.ones(T, T, device=Q.device))
        scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        return attn @ V


class BDH(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.n_layer = config.n_layer
        self.n_embd = config.n_embd
        self.n_head = config.n_head

        # ✅ ONLY CHANGE: reduced latent multiplier
        self.mult = 32   # instead of 128

        self.dropout = config.dropout

        self.N = self.n_embd * self.mult // self.n_head

        self.embed = nn.Embedding(config.vocab_size, self.n_embd)

        self.encoder = nn.Parameter(
            torch.randn(self.n_head, self.n_embd, self.N) * 0.02
        )
        self.encoder_v = nn.Parameter(
            torch.randn(self.n_head, self.n_embd, self.N) * 0.02
        )
        self.decoder = nn.Parameter(
            torch.randn(self.n_head * self.N, self.n_embd) * 0.02
        )

        self.attn = Attention(self.n_head, self.n_embd)

        self.ln = nn.LayerNorm(self.n_embd)
        self.drop = nn.Dropout(self.dropout)

        self.lm_head = nn.Linear(self.n_embd, config.vocab_size, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        x = self.embed(idx).unsqueeze(1)
        x = self.ln(x)

        for _ in range(self.n_layer):

            # ---- latent ----
            x_latent = x @ self.encoder
            x_sparse = F.relu(x_latent)

            # ---- attention ----
            y = self.attn(x_sparse, x_sparse, x)
            y = self.ln(y)

            # ---- second branch ----
            y_latent = y @ self.encoder_v
            y_sparse = F.relu(y_latent)

            # ---- multiplicative interaction ----
            xy = x_sparse * y_sparse

            xy = self.drop(xy)

            # ---- decode ----
            y_mlp = (
                xy.transpose(1, 2)
                .reshape(B, 1, T, self.n_head * self.N)
                @ self.decoder
            )

            y_out = self.ln(y_mlp)

            # ---- residual ----
            x = self.ln(x + y_out)

        x = x.view(B, T, self.n_embd)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )

        return logits, loss