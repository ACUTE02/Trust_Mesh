"""LCNN (Light CNN) spoof detector, the SSL layer-weighted head, and a shared training loop.

LCNN is the CNN architecture used by the ASVspoof baselines. Its defining piece is the
Max-Feature-Map (MFM) activation: a conv produces 2k channels, which are split in half and combined
with an element-wise max, halving the channels. That acts as a learned feature selector and keeps
the network small.

Every model outputs one logit per clip, where a higher score means "more bonafide" (label 1 =
bonafide), matching the EER convention in evaluate.py.
"""
import math

import numpy as np
import torch
import torch.nn.functional as F

from .evaluate import compute_eer


class MFM(torch.nn.Module):
    """Max-Feature-Map over channels (dim 1)."""

    def forward(self, x):
        a, b = torch.chunk(x, 2, dim=1)
        return torch.maximum(a, b)


def conv_mfm(in_ch, out_ch, kernel_size, padding):
    return torch.nn.Sequential(
        torch.nn.Conv2d(in_ch, out_ch * 2, kernel_size=kernel_size, padding=padding),
        MFM(),
    )


class LCNN(torch.nn.Module):
    MIN_SIZE = 16  # four 2x max-pools: each input axis must be at least 2**4 long

    def __init__(self, n_features=60, n_frames=401, channels=(16, 24, 32, 32), dropout=0.3, fc_dim=80):
        super().__init__()
        if n_features < self.MIN_SIZE or n_frames < self.MIN_SIZE:
            raise ValueError(f"LCNN needs at least {self.MIN_SIZE} feature rows and frames, got "
                             f"{n_features}x{n_frames} -- use a longer input window")
        c1, c2, c3, c4 = channels
        self.blocks = torch.nn.Sequential(
            conv_mfm(1, c1, 5, 2),
            torch.nn.MaxPool2d(2),
            conv_mfm(c1, c1, 1, 0),
            torch.nn.BatchNorm2d(c1),
            conv_mfm(c1, c2, 3, 1),
            torch.nn.MaxPool2d(2),
            torch.nn.BatchNorm2d(c2),
            conv_mfm(c2, c2, 1, 0),
            torch.nn.BatchNorm2d(c2),
            conv_mfm(c2, c3, 3, 1),
            torch.nn.MaxPool2d(2),
            conv_mfm(c3, c3, 1, 0),
            torch.nn.BatchNorm2d(c3),
            conv_mfm(c3, c4, 3, 1),
            torch.nn.MaxPool2d(2),
        )
        with torch.no_grad():  # measure the flattened size instead of deriving it from the pooling arithmetic
            flat_dim = self.blocks(torch.zeros(1, 1, n_features, n_frames)).numel()
        self.head = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(flat_dim, fc_dim * 2),
            MFM(),  # operates on dim 1, which is the feature dim for a 2D tensor
            torch.nn.BatchNorm1d(fc_dim),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(fc_dim, 1),
        )

    def forward(self, x):
        return self.head(self.blocks(x)).squeeze(-1)


class SSLHead(torch.nn.Module):
    """Classifier over pooled self-supervised encoder features [batch, layers, dim].

    A learned softmax weight per encoder layer mixes the layers into one vector, then a small MLP
    scores it. The layer weights are reported after training: they say which depth of the encoder
    carried the spoofing information.
    """

    def __init__(self, n_layers, feat_dim, hidden=128, dropout=0.3):
        super().__init__()
        self.layer_logits = torch.nn.Parameter(torch.zeros(n_layers))
        self.mlp = torch.nn.Sequential(
            torch.nn.LayerNorm(feat_dim),
            torch.nn.Linear(feat_dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, 1),
        )

    def forward(self, x):
        weights = torch.softmax(self.layer_logits, dim=0)
        return self.mlp((weights.view(1, -1, 1) * x).sum(dim=1)).squeeze(-1)

    def layer_weights(self):
        return torch.softmax(self.layer_logits.detach(), dim=0).cpu().tolist()

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


class SSLSeqHead(torch.nn.Module):
    """Phase 2C Model E: a single-layer bidirectional LSTM over one frozen SSL layer's per-frame
    token sequence [batch, frames, hidden], mean-pooled over time, then a small MLP head.

    Only one SSL layer is used (picked from Model C's learned layer weights, logged in the notebook),
    so there is no per-layer weighting mechanism here -- unlike `SSLHead`, which mixes every layer.
    Kept deliberately small (BiLSTM hidden size 64 by default): the goal is to test whether *any*
    temporal modeling helps, not to add capacity as a confound. `n_added_params` lets the notebook log
    that confound directly against `SSLHead`'s parameter count.
    """

    def __init__(self, feat_dim, lstm_hidden=64, hidden=128, dropout=0.3):
        super().__init__()
        self.lstm = torch.nn.LSTM(feat_dim, lstm_hidden, num_layers=1, batch_first=True, bidirectional=True)
        out_dim = 2 * lstm_hidden
        self.mlp = torch.nn.Sequential(
            torch.nn.LayerNorm(out_dim),
            torch.nn.Linear(out_dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.mlp(out.mean(dim=1)).squeeze(-1)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


def batches(n, batch_size, shuffle, generator=None):
    order = torch.randperm(n, generator=generator) if shuffle else torch.arange(n)
    for start in range(0, n, batch_size):
        yield order[start:start + batch_size]


def score_model(model, fetch, indices, batch_size=64):
    """Scores (higher = more bonafide) for `indices`, in order. `fetch(idx, training=False)`."""
    model.eval()
    indices = np.asarray(indices)
    out = []
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            out.append(model(fetch(indices[start:start + batch_size], False)).detach().float().cpu())
    return torch.cat(out).numpy().astype(np.float64) if out else np.zeros(0)


def train_model(model, fetch, labels, train_idx, pos_weight, device, val_idx=None, epochs=20,
                batch_size=32, lr=3e-4, weight_decay=1e-4, patience=5, seed=42, verbose=True):
    """Class-weighted BCE training with optional early stopping on validation EER.

    `fetch(idx, training)` returns the model input for clip indices `idx` on `device`; data sources
    apply augmentation only when `training` is True, so validation and test inputs are never
    augmented. `labels` is indexed by the same clip indices.
    """
    labels = torch.as_tensor(np.asarray(labels))
    train_idx = np.asarray(train_idx)
    generator = torch.Generator().manual_seed(seed)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    pos_weight_t = torch.tensor([pos_weight], device=device, dtype=torch.float32)

    history, best = [], {"eer": math.inf, "epoch": 0, "state": None}
    since_improve, val_unusable = 0, False
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, n_seen = 0.0, 0
        for positions in batches(len(train_idx), batch_size, shuffle=True, generator=generator):
            if len(positions) < 2:
                continue  # BatchNorm needs at least 2 samples
            idx = train_idx[positions.numpy()]
            xb = fetch(idx, True)
            yb = labels[idx].to(device).float()
            optimizer.zero_grad()
            loss = F.binary_cross_entropy_with_logits(model(xb), yb, pos_weight=pos_weight_t)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(idx)
            n_seen += len(idx)
        epoch_loss = total_loss / max(n_seen, 1)
        if not math.isfinite(epoch_loss):
            raise FloatingPointError(f"training diverged at epoch {epoch} (loss={epoch_loss})")

        entry = {"epoch": epoch, "train_loss": epoch_loss}
        if val_idx is not None and not val_unusable:
            entry["val_eer"] = compute_eer(score_model(model, fetch, val_idx), labels[np.asarray(val_idx)].numpy())["eer"]
            if not math.isfinite(entry["val_eer"]):
                # One-class validation set: EER is undefined, so model selection and early stopping
                # have nothing to act on. Say so loudly rather than silently keeping the last epoch.
                val_unusable = True
                print("  WARNING: validation EER is undefined (validation set has a single class) -- "
                      "disabling early stopping and best-epoch selection for this run")
            elif entry["val_eer"] < best["eer"] - 1e-6:
                best = {"eer": entry["val_eer"], "epoch": epoch,
                        "state": {k: v.detach().clone() for k, v in model.state_dict().items()}}
                since_improve = 0
            else:
                since_improve += 1
        history.append(entry)
        if verbose:
            print(f"  epoch {epoch:3d}  loss {epoch_loss:.4f}" +
                  (f"  val EER {entry['val_eer']:.4f}" if "val_eer" in entry else ""))
        if patience is not None and val_idx is not None and not val_unusable and since_improve >= patience:
            if verbose:
                print(f"  early stopping at epoch {epoch} (best val EER {best['eer']:.4f} at epoch {best['epoch']})")
            break

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    usable_val = val_idx is not None and not val_unusable
    return {"history": history, "best_val_eer": best["eer"] if usable_val else None,
            "best_epoch": best["epoch"] if usable_val else len(history),
            "epochs_run": len(history), "validation_usable": usable_val}


def train_lcnn(model, x_train, y_train, pos_weight, device, epochs=20, batch_size=32, lr=3e-4,
               weight_decay=1e-4, x_val=None, y_val=None, patience=5, seed=42, verbose=True):
    """In-memory wrapper around `train_model` (the Phase 2 notebook's interface)."""
    n_train = len(x_train)
    x_all = x_train if x_val is None else torch.cat([x_train, x_val])
    y_all = y_train if y_val is None else torch.cat([y_train, y_val])
    val_idx = None if x_val is None else np.arange(n_train, n_train + len(x_val))
    return train_model(model, lambda idx, training: x_all[torch.as_tensor(idx)].to(device), y_all,
                       np.arange(n_train), pos_weight, device, val_idx=val_idx, epochs=epochs,
                       batch_size=batch_size, lr=lr, weight_decay=weight_decay, patience=patience,
                       seed=seed, verbose=verbose)


@torch.no_grad()
def predict_scores(model, x, device, batch_size=64):
    """Scores where higher = more bonafide, for an in-memory feature tensor."""
    return score_model(model, lambda idx, training: x[torch.as_tensor(idx)].to(device),
                       np.arange(len(x)), batch_size)
