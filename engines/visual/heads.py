"""The heads trained on cached CLIP embeddings (Task 6): logistic regression, and a one-hidden-layer
MLP for comparison.

Both read L2-normalized embeddings and output one logit per image, higher = more likely fake.
Weights are saved as plain arrays (`.npz`) and the file's SHA-256 is recorded with the results.
"""
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from .embed import file_sha256
from .evaluate import fast_roc_auc

C_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)


def l2_normalize(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


class LinearHead:
    kind = "logreg"

    def __init__(self, coef, intercept, c=None):
        self.coef = np.asarray(coef, dtype=np.float64).reshape(-1)
        self.intercept = float(intercept)
        self.c = c

    def logits(self, x):
        return l2_normalize(x).astype(np.float64) @ self.coef + self.intercept

    def arrays(self):
        return {"kind": np.array(self.kind), "coef": self.coef, "intercept": np.array(self.intercept),
                "c": np.array(np.nan if self.c is None else self.c)}

    def n_params(self):
        return int(self.coef.size + 1)


def fit_logreg_head(x_train, y_train, x_val, y_val, c_grid=C_GRID, max_iter=2000):
    """L2-regularized logistic regression (L-BFGS). `C` is the value with the best validation
    ROC-AUC; ties go to the smaller C (stronger regularization). Given its inputs the fit is
    deterministic -- a seed changes nothing here."""
    xt, xv = l2_normalize(x_train), l2_normalize(x_val)
    tried, best = {}, None
    for c in c_grid:
        model = LogisticRegression(C=c, max_iter=max_iter)
        model.fit(xt, y_train)
        auc = fast_roc_auc(model.decision_function(xv), y_val)
        tried[str(c)] = auc
        if best is None or auc > best[0] + 1e-9:
            best = (auc, c, model)
    head = LinearHead(best[2].coef_, best[2].intercept_[0], best[1])
    return head, {"validation_auc_by_c": tried, "chosen_c": best[1], "validation_auc": best[0]}


class _MLP(torch.nn.Module):
    def __init__(self, dim, hidden, dropout):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(dim, hidden), torch.nn.ReLU(), torch.nn.Dropout(dropout),
                                       torch.nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


class MLPHead:
    kind = "mlp"

    def __init__(self, dim, hidden=256, dropout=0.2, state=None):
        self.dim, self.hidden, self.dropout = int(dim), int(hidden), float(dropout)
        self.module = _MLP(self.dim, self.hidden, self.dropout)
        if state is not None:
            self.module.load_state_dict({k: torch.as_tensor(v) for k, v in state.items()})
        self.module.eval()

    @torch.no_grad()
    def logits(self, x):
        self.module.eval().cpu()
        return self.module(torch.from_numpy(l2_normalize(x))).numpy().astype(np.float64)

    def arrays(self):
        out = {"kind": np.array(self.kind), "dim": np.array(self.dim), "hidden": np.array(self.hidden),
               "dropout": np.array(self.dropout)}
        out.update({f"state/{k}": v.detach().cpu().numpy() for k, v in self.module.state_dict().items()})
        return out

    def n_params(self):
        return int(sum(p.numel() for p in self.module.parameters()))


def fit_mlp_head(x_train, y_train, x_val, y_val, seed=0, hidden=256, dropout=0.2, epochs=60, patience=6,
                 lr=1e-3, weight_decay=1e-4, batch_size=256, device=torch.device("cpu")):
    """One hidden layer, Adam, early stopping on validation ROC-AUC. The seed sets the initial
    weights and the mini-batch order."""
    torch.manual_seed(seed)
    head = MLPHead(x_train.shape[1], hidden, dropout)
    module = head.module.to(device)
    xt = torch.from_numpy(l2_normalize(x_train)).to(device)
    yt = torch.as_tensor(np.asarray(y_train), dtype=torch.float32, device=device)
    xv = torch.from_numpy(l2_normalize(x_val)).to(device)
    optimizer = torch.optim.Adam(module.parameters(), lr=lr, weight_decay=weight_decay)
    generator = torch.Generator().manual_seed(seed)
    best = {"auc": -1.0, "epoch": 0, "state": None}
    history, since = [], 0
    for epoch in range(1, epochs + 1):
        module.train()
        order = torch.randperm(len(xt), generator=generator).to(device)
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            optimizer.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(module(xt[idx]), yt[idx])
            loss.backward()
            optimizer.step()
        module.eval()
        with torch.no_grad():
            auc = fast_roc_auc(module(xv).cpu().numpy(), y_val)
        history.append(auc)
        if auc > best["auc"] + 1e-6:
            best = {"auc": auc, "epoch": epoch, "state": {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}}
            since = 0
        else:
            since += 1
            if since >= patience:
                break
    if best["state"] is not None:
        module.load_state_dict(best["state"])
    module.cpu().eval()
    return head, {"validation_auc": best["auc"], "best_epoch": best["epoch"], "epochs_run": len(history), "seed": seed}


def save_head(path, head):
    """Write the head's arrays to `path` (.npz) and return the file's SHA-256."""
    with open(path, "wb") as handle:
        np.savez(handle, **head.arrays())
    return file_sha256(path)


def load_head(path):
    with np.load(path) as data:
        kind = str(data["kind"])
        if kind == "logreg":
            c = float(data["c"])
            return LinearHead(data["coef"], float(data["intercept"]), None if np.isnan(c) else c)
        state = {k[len("state/"):]: data[k] for k in data.files if k.startswith("state/")}
        return MLPHead(int(data["dim"]), int(data["hidden"]), float(data["dropout"]), state)
