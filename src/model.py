"""The PyTorch network, the training loop, evaluation and the over/under-fitting diagnosis."""
import copy

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, TensorDataset

from src.config import DEFAULT_CFG, SEED

criterion = nn.BCEWithLogitsLoss()      # applies the sigmoid internally, so the model outputs raw logits


class HospitalNet(nn.Module):
    """Linear(n, hidden) -> ReLU -> Dropout -> Linear(hidden, 1)."""

    def __init__(self, n_features: int, hidden: int = 16, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


def make_cfg(**overrides) -> dict:
    """Default settings with overrides. patience=None means early stopping is OFF."""
    cfg = dict(DEFAULT_CFG)
    cfg.update(overrides)
    return cfg


@torch.no_grad()
def loss_on(model, X, y) -> float:
    model.eval()
    return criterion(model(X), y).item()


def train_model(cfg, Xtr, ytr, Xva=None, yva=None, fixed_epochs=None, batch_size=32, progress_cb=None):
    """Train one model. Returns (model, history, best_epoch).

    history = {"train_loss": [...], "val_loss": [...]}, measured in eval mode after every epoch.
    With early stopping, the weights of the best validation epoch are restored at the end.
    progress_cb(epoch, history, max_epochs) is called after every epoch (used by the live dashboard).
    """
    torch.manual_seed(SEED)
    model = HospitalNet(Xtr.shape[1], cfg["hidden"], cfg["dropout"])
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(SEED))
    max_epochs = fixed_epochs or cfg["max_epochs"]
    history = {"train_loss": [], "val_loss": []}
    best_val, best_epoch, best_state, wait = float("inf"), 0, None, 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        history["train_loss"].append(loss_on(model, Xtr, ytr))
        if Xva is not None:
            v = loss_on(model, Xva, yva)
            history["val_loss"].append(v)
            if v < best_val - 1e-6:                              # validation improved
                best_val, best_epoch, wait = v, epoch, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                wait += 1
                if cfg["patience"] is not None and wait >= cfg["patience"]:
                    break                                         # early stopping
        if progress_cb:
            progress_cb(epoch, history, max_epochs)

    if Xva is not None and cfg["patience"] is not None and best_state is not None:
        model.load_state_dict(best_state)                         # go back to the best epoch
    if Xva is None:
        best_epoch = max_epochs
    return model, history, best_epoch


@torch.no_grad()
def predict_proba(model, X) -> np.ndarray:
    """Probability of malignant for each row."""
    model.eval()
    return torch.sigmoid(model(X)).numpy()


@torch.no_grad()
def evaluate_model(model, X, y, threshold: float = 0.5) -> dict:
    """Loss + classification metrics. Malignant (1) is the positive class."""
    model.eval()
    logits = model(X)
    probs = torch.sigmoid(logits).numpy()
    preds = (probs >= threshold).astype(int)
    yt = y.numpy().astype(int)
    tn, fp, fn, tp = confusion_matrix(yt, preds, labels=[0, 1]).ravel()
    return dict(
        loss=criterion(logits, y).item(),
        accuracy=accuracy_score(yt, preds),
        precision=precision_score(yt, preds, zero_division=0),
        recall=recall_score(yt, preds, zero_division=0),
        f1=f1_score(yt, preds, zero_division=0),
        tp=int(tp), fn=int(fn), fp=int(fp), tn=int(tn),
        probs=probs, preds=preds,
    )


def diagnose(history: dict) -> str:
    """Underfitting / Overfitting / Healthy from the loss curves.

    Underfitting : final training loss > 0.30
    Overfitting  : final validation loss > 1.30 x the best validation loss of an EARLIER epoch
                   (the last epoch is excluded from the minimum, otherwise it could never exceed itself)
    """
    tr, va = history["train_loss"], history["val_loss"]
    if tr[-1] > 0.30:
        return "Underfitting"
    if len(va) > 1 and va[-1] > 1.30 * min(va[:-1]):
        return "Overfitting"
    return "Healthy"
