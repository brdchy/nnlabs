"""
Лабораторная 4: MNIST, нормализация, MLP и CNN, графики потерь, сравнение метрик.
Запуск: python lab04_mnist.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class MLPNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CNNNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        return self.fc(x)


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    return (logits.argmax(dim=1) == y).float().mean().item()


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int = 5,
) -> tuple[list[float], float]:
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    losses: list[float] = []
    model.train()
    for _ in range(epochs):
        epoch_loss = 0.0
        n = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * xb.size(0)
            n += xb.size(0)
        losses.append(epoch_loss / n)

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            correct += (logits.argmax(1) == yb).sum().item()
            total += yb.size(0)
    return losses, correct / total


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tfm = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=tfm)
    test_ds = datasets.MNIST(root="./data", train=False, download=True, transform=tfm)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=256)

    mlp = MLPNet().to(device)
    cnn = CNNNet().to(device)

    print("Обучение MLP...")
    loss_mlp, acc_mlp = train_model(mlp, train_loader, test_loader, device, epochs=5)
    print(f"  Точность MLP на тесте: {acc_mlp:.4f}")

    print("Обучение CNN...")
    loss_cnn, acc_cnn = train_model(cnn, train_loader, test_loader, device, epochs=5)
    print(f"  Точность CNN на тесте: {acc_cnn:.4f}")

    # График потерь
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(loss_mlp, label="MLP (loss по эпохам)", marker="o", markersize=3)
    ax.plot(loss_cnn, label="CNN (loss по эпохам)", marker="s", markersize=3)
    ax.set_xlabel("Эпоха")
    ax.set_ylabel("Средний CrossEntropy")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title("Лаб. 4 — MNIST: потери MLP vs CNN")
    plt.tight_layout()
    out = __file__.replace(".py", "_loss.png")
    plt.savefig(out, dpi=120)
    print(f"График сохранён: {out}")

    # Бар‑чарт точности моделей
    fig2, ax2 = plt.subplots(figsize=(4, 4))
    models = ["MLP", "CNN"]
    accs = [acc_mlp, acc_cnn]
    ax2.bar(models, accs, color=["tab:orange", "tab:blue"])
    ax2.set_ylim(0.0, 1.05)
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Лаб. 4 — точность MLP vs CNN")
    for i, v in enumerate(accs):
        ax2.text(i, v + 0.01, f"{v:.3f}", ha="center", va="bottom")
    ax2.grid(True, axis="y", alpha=0.2)
    plt.tight_layout()
    out2 = __file__.replace(".py", "_accuracy.png")
    plt.savefig(out2, dpi=120)
    print(f"Бар-чарт точности сохранён: {out2}")


if __name__ == "__main__":
    main()
