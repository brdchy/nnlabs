"""
Лабораторная 5: последовательности (IMDB), токенизация, эмбеддинги, RNN vs LSTM.
Запуск: python lab05_imdb_rnn.py  (первый раз скачает датасет)
"""

from __future__ import annotations

import collections
import re

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm


def tokenize(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    return [w for w in text.split() if w]


def build_vocab(texts: list[str], max_vocab: int) -> dict[str, int]:
    counter: collections.Counter[str] = collections.Counter()
    for t in texts:
        counter.update(tokenize(t))
    most = [w for w, _ in counter.most_common(max_vocab - 2)]
    stoi = {"<pad>": 0, "<unk>": 1}
    for i, w in enumerate(most, start=2):
        stoi[w] = i
    return stoi


def encode_with_length(text: str, stoi: dict[str, int], max_len: int) -> tuple[list[int], int]:
    words = tokenize(text)
    ids = [stoi.get(w, 1) for w in words][:max_len]
    length = max(len(ids), 1)
    if len(ids) < max_len:
        ids = ids + [0] * (max_len - len(ids))
    return ids, min(length, max_len)


class RNNClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        emb_dim: int,
        hidden: int,
        num_classes: int = 2,
        num_layers: int = 1,
        dropout: float = 0.4,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        self.hidden = hidden
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        rnn_dropout = dropout if num_layers > 1 else 0.0
        self.rnn = nn.RNN(
            emb_dim,
            hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=rnn_dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden * self.num_directions, num_classes)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        e = self.dropout(self.emb(x))
        packed = nn.utils.rnn.pack_padded_sequence(
            e, lengths.detach().cpu(), batch_first=True, enforce_sorted=False
        )
        _, h = self.rnn(packed)
        # h: (num_layers * num_directions, batch, hidden)
        if self.bidirectional:
            h = torch.cat((h[-2], h[-1]), dim=1)
        else:
            h = h[-1]
        return self.fc(self.dropout(h))


class LSTMClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        emb_dim: int,
        hidden: int,
        num_classes: int = 2,
        num_layers: int = 1,
        dropout: float = 0.4,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        self.hidden = hidden
        self.num_directions = 2 if bidirectional else 1
        self.bidirectional = bidirectional
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            emb_dim,
            hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=lstm_dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden * self.num_directions, num_classes)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        e = self.dropout(self.emb(x))
        packed = nn.utils.rnn.pack_padded_sequence(
            e, lengths.detach().cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        if self.bidirectional:
            h = torch.cat((h_n[-2], h_n[-1]), dim=1)
        else:
            h = h_n[-1]
        return self.fc(self.dropout(h))


def train_eval(
    model: nn.Module,
    X_train: np.ndarray,
    len_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    len_test: np.ndarray,
    y_test: np.ndarray,
    device: torch.device,
    epochs: int,
    batch_size: int,
) -> tuple[list[float], float]:
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss()
    losses: list[float] = []

    n = X_train.shape[0]
    Xt = torch.tensor(X_train, dtype=torch.long, device=device)
    Lt = torch.tensor(len_train, dtype=torch.long, device=device)
    yt = torch.tensor(y_train, dtype=torch.long, device=device)

    model.train()
    for _ in tqdm(range(epochs), desc=f"{model.__class__.__name__} epochs", leave=False):
        perm = torch.randperm(n, device=device)
        epoch_loss = 0.0
        steps = 0
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            xb, lb, yb = Xt[idx], Lt[idx], yt[idx]
            opt.zero_grad()
            logits = model(xb, lb)
            loss = loss_fn(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            epoch_loss += float(loss.item())
            steps += 1
        losses.append(epoch_loss / max(steps, 1))
        sched.step()

    model.eval()
    Xte = torch.tensor(X_test, dtype=torch.long, device=device)
    Lte = torch.tensor(len_test, dtype=torch.long, device=device)
    yte = torch.tensor(y_test, dtype=torch.long, device=device)
    correct = 0
    total = 0
    with torch.no_grad():
        for i in range(0, len(y_test), batch_size):
            xb = Xte[i : i + batch_size]
            lb = Lte[i : i + batch_size]
            yb = yte[i : i + batch_size]
            pred = model(xb, lb).argmax(dim=1)
            correct += (pred == yb).sum().item()
            total += yb.size(0)
    return losses, correct / total


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Устройство: {device}", flush=True)

    print("Загрузка IMDB...", flush=True)
    ds = load_dataset("imdb", split="train")
    texts = [ds[i]["text"] for i in range(len(ds))]
    labels = [ds[i]["label"] for i in range(len(ds))]

    X_tr, X_te, y_tr, y_te = train_test_split(
        texts, labels, test_size=0.15, random_state=42, stratify=labels
    )
    y_tr = np.array(y_tr, dtype=np.int64)
    y_te = np.array(y_te, dtype=np.int64)

    max_vocab = 20000
    max_len = 256
    print("Словарь и кодирование текстов...", flush=True)
    stoi = build_vocab(X_tr, max_vocab=max_vocab)

    train_ids: list[list[int]] = []
    train_lens: list[int] = []
    for t in X_tr:
        ids, ln = encode_with_length(t, stoi, max_len)
        train_ids.append(ids)
        train_lens.append(ln)
    test_ids: list[list[int]] = []
    test_lens: list[int] = []
    for t in X_te:
        ids, ln = encode_with_length(t, stoi, max_len)
        test_ids.append(ids)
        test_lens.append(ln)

    X_train = np.array(train_ids, dtype=np.int64)
    len_train = np.array(train_lens, dtype=np.int64)
    X_test = np.array(test_ids, dtype=np.int64)
    len_test = np.array(test_lens, dtype=np.int64)

    vocab_size = max(stoi.values()) + 1
    emb_dim = 128
    hidden = 128
    epochs = 10
    batch_size = 128

    print("Обучение RNN + Embedding (двунаправленный, pack_padded)...", flush=True)
    rnn = RNNClassifier(vocab_size, emb_dim, hidden, num_layers=1, dropout=0.4, bidirectional=True)
    loss_rnn, acc_rnn = train_eval(
        rnn, X_train, len_train, y_tr, X_test, len_test, y_te, device, epochs, batch_size
    )
    print(f"  Точность RNN на тесте: {acc_rnn:.4f}", flush=True)

    print("Обучение LSTM + Embedding...", flush=True)
    lstm = LSTMClassifier(vocab_size, emb_dim, hidden, num_layers=1, dropout=0.4, bidirectional=True)
    loss_lstm, acc_lstm = train_eval(
        lstm, X_train, len_train, y_tr, X_test, len_test, y_te, device, epochs, batch_size
    )
    print(f"  Точность LSTM на тесте: {acc_lstm:.4f}", flush=True)

    # График потерь
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(loss_rnn, label="RNN", marker="o", markersize=3)
    ax.plot(loss_lstm, label="LSTM", marker="s", markersize=3)
    ax.set_xlabel("Эпоха")
    ax.set_ylabel("Средний loss (батчи)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title("Лаб. 5 — IMDB: потери RNN vs LSTM")
    plt.tight_layout()
    out = __file__.replace(".py", "_loss.png")
    plt.savefig(out, dpi=120)
    print(f"График сохранён: {out}", flush=True)

    # Бар‑чарт точности
    fig2, ax2 = plt.subplots(figsize=(4, 4))
    models = ["RNN", "LSTM"]
    accs = [acc_rnn, acc_lstm]
    ax2.bar(models, accs, color=["tab:green", "tab:purple"])
    ax2.set_ylim(0.0, 1.0)
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Лаб. 5 — точность RNN vs LSTM")
    for i, v in enumerate(accs):
        ax2.text(i, v + 0.01, f"{v:.3f}", ha="center", va="bottom")
    ax2.grid(True, axis="y", alpha=0.2)
    plt.tight_layout()
    out2 = __file__.replace(".py", "_accuracy.png")
    plt.savefig(out2, dpi=120)
    print(f"Бар‑чарт точности сохранён: {out2}", flush=True)


if __name__ == "__main__":
    main()
