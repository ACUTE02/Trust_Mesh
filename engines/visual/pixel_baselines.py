"""Heavier pixel baselines (Task 9): a small CNN from scratch and a fine-tuned ResNet-50.

**Off by default** (`RUN_SMALL_CNN`, `RUN_RESNET50` in the notebook). Unlike the heads, these train
on pixels, which brings back the disk and time risk that sank Voice Models E and F. So the images
they use are a capped subset held in memory as small uint8 arrays -- never written to disk -- and
they run only after Tasks 1-8 have real results and the user says go.
"""
import numpy as np
import torch

from .evaluate import fast_roc_auc
from .preprocess import CLIP_MEAN
from .preprocess import CLIP_STD
from .preprocess import decode_image
from .preprocess import standardize


def pixel_array(data, side):
    """Image bytes -> uint8 [side, side, 3] after the shared pipeline, or None for an invalid file."""
    record = decode_image(data)
    if not record["ok"]:
        return None
    image = standardize(record["rgb"])[0]
    if side != image.size[0]:
        image = image.resize((side, side))
    return np.asarray(image, dtype=np.uint8)


class SmallCNN(torch.nn.Module):
    """Four conv blocks and a linear layer -- the 'small CNN from scratch' baseline."""

    def __init__(self, channels=(16, 32, 64, 96)):
        super().__init__()
        layers, previous = [], 3
        for c in channels:
            layers += [torch.nn.Conv2d(previous, c, 3, padding=1), torch.nn.BatchNorm2d(c), torch.nn.ReLU(),
                       torch.nn.MaxPool2d(2)]
            previous = c
        self.features = torch.nn.Sequential(*layers, torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten())
        self.head = torch.nn.Linear(previous, 1)

    def forward(self, x):
        return self.head(self.features(x)).squeeze(-1)


def build_resnet50(pretrained=True):
    """torchvision's ResNet-50 with a one-logit head. `pretrained` downloads the ImageNet weights
    (Kaggle, internet on); the smoke test passes False."""
    from torchvision.models import resnet50

    model = resnet50(weights="IMAGENET1K_V2" if pretrained else None)
    model.fc = torch.nn.Linear(model.fc.in_features, 1)
    return model


class _OneLogit(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x).squeeze(-1)


def _to_tensor(images, device):
    x = torch.from_numpy(np.ascontiguousarray(images)).to(device).float().div_(255.0).permute(0, 3, 1, 2)
    mean = torch.tensor(CLIP_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(CLIP_STD, device=device).view(1, 3, 1, 1)
    return (x - mean) / std


@torch.no_grad()
def pixel_logits(model, images, device, batch_size=64):
    model.eval()
    out = [model(_to_tensor(images[i:i + batch_size], device)).float().cpu() for i in range(0, len(images), batch_size)]
    return torch.cat(out).numpy().astype(np.float64) if out else np.zeros(0)


def train_pixel_model(model, x_train, y_train, x_val, y_val, device, epochs=8, patience=3, lr=1e-3, batch_size=64,
                      seed=0, log=print):
    """Train a pixel model on in-memory uint8 images with early stopping on validation ROC-AUC.
    Prints one line per epoch."""
    torch.manual_seed(seed)
    model = (_OneLogit(model) if not isinstance(model, SmallCNN) else model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    labels = torch.as_tensor(np.asarray(y_train), dtype=torch.float32)
    generator = torch.Generator().manual_seed(seed)
    best = {"auc": -1.0, "epoch": 0, "state": None}
    since = 0
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(x_train), generator=generator).numpy()
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            if len(idx) < 2:
                continue
            optimizer.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                model(_to_tensor(x_train[idx], device)), labels[idx].to(device))
            loss.backward()
            optimizer.step()
        auc = fast_roc_auc(pixel_logits(model, x_val, device, batch_size), y_val)
        log(f"    epoch {epoch}: validation ROC-AUC {auc:.4f}")
        if auc > best["auc"] + 1e-6:
            best = {"auc": auc, "epoch": epoch, "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
            since = 0
        else:
            since += 1
            if since >= patience:
                break
    if best["state"] is not None:
        model.load_state_dict(best["state"])
    return model, {"validation_auc": best["auc"], "best_epoch": best["epoch"], "seed": seed,
                   "n_params": int(sum(p.numel() for p in model.parameters()))}
