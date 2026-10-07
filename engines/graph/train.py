"""Full-batch training loop shared by every GNN (same protocol as notebook 01's train_model_full_batch)."""
import math

import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score


def train_full_batch(model, optimizer, forward, y, train_mask, pos_weight, epochs,
                     val_mask=None, patience=None, verbose=False):
    """Class-weighted BCE on `train_mask`. `forward(model)` returns one logit per node.

    With `val_mask`, keeps the state with the best validation PR-AUC and optionally early-stops after
    `patience` epochs without improvement. Models may define `auxiliary_loss(y, train_mask,
    pos_weight)` (added to the loss) and `on_epoch_end()` (called after each optimizer step)."""
    loss_history = []
    best_val, best_state, best_epoch, since_improve = -1.0, None, 0, 0
    epoch = 0
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = forward(model)
        loss = F.binary_cross_entropy_with_logits(out[train_mask], y[train_mask].float(), pos_weight=pos_weight)
        if hasattr(model, "auxiliary_loss"):
            loss = loss + model.auxiliary_loss(y, train_mask, pos_weight)
        loss.backward()
        optimizer.step()
        loss_value = loss.item()
        if not math.isfinite(loss_value):
            raise FloatingPointError(f"training diverged at epoch {epoch} (loss={loss_value})")
        loss_history.append(loss_value)
        if hasattr(model, "on_epoch_end"):
            model.on_epoch_end()

        if val_mask is None:
            continue
        val_pr_auc = pr_auc(predict_probs(model, forward), y, val_mask)
        if val_pr_auc > best_val:
            best_val, best_epoch, since_improve = val_pr_auc, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            since_improve += 1
        if patience is not None and since_improve >= patience:
            if verbose:
                print(f"  early stopping at epoch {epoch} (best val PR-AUC {best_val:.4f} at epoch {best_epoch})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return {
        "loss_history": loss_history,
        "best_val_pr_auc": best_val if val_mask is not None else None,
        "best_epoch": best_epoch if val_mask is not None else epoch,
        "epochs_run": epoch,
    }


@torch.no_grad()
def predict_probs(model, forward):
    model.eval()
    return torch.sigmoid(forward(model)).detach().cpu()


def pr_auc(probs, y, mask):
    mask = mask.cpu()
    return float(average_precision_score(y.cpu()[mask].numpy(), probs[mask].numpy()))
