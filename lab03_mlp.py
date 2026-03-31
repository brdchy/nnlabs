"""
Лабораторная 3: предобработка, MLP (PyTorch), график потерь, метрики.
Датасет: Iris (классификация).
Запуск: python lab03_mlp.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, num_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.long, device=device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32, device=device)
    y_test_t = torch.tensor(y_test, dtype=torch.long, device=device)

    model = MLP(in_dim=4, hidden=32, num_classes=3).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = nn.CrossEntropyLoss()

    epochs = 200
    losses: list[float] = []
    model.train()
    for ep in range(epochs):
        opt.zero_grad()
        logits = model(X_train_t)
        loss = loss_fn(logits, y_train_t)
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))

    model.eval()
    with torch.no_grad():
        logits_test = model(X_test_t)
        pred = logits_test.argmax(dim=1).cpu().numpy()
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, average="macro")

    print(f"Точность на тесте: {acc:.4f}")
    print(f"F1 (macro):         {f1:.4f}")

    # График функции потерь
    plt.figure(figsize=(8, 4))
    plt.plot(losses, color="steelblue")
    plt.xlabel("Эпоха")
    plt.ylabel("Loss (CrossEntropy)")
    plt.title("Лаб. 3 — функция потерь при обучении MLP")
    plt.grid(True, alpha=0.3)
    out = __file__.replace(".py", "_loss.png")
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    print(f"График сохранён: {out}")

    # Confusion matrix
    cm = confusion_matrix(y_test, pred)
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xlabel("Предсказанный класс")
    ax.set_ylabel("Истинный класс")
    ax.set_title("Лаб. 3 — confusion matrix")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8)
    plt.tight_layout()
    out_cm = __file__.replace(".py", "_confusion.png")
    plt.savefig(out_cm, dpi=120)
    print(f"Confusion matrix сохранена: {out_cm}")


if __name__ == "__main__":
    main()
