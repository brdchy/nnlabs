"""
Лабораторная 1: функции активации и нейрон (сумматор + активация).
Запуск: python lab01_neuron.py
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def act_identity(x: np.ndarray) -> np.ndarray:
    return x


def act_step(x: np.ndarray) -> np.ndarray:
    return (x >= 0).astype(np.float64)


def act_sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def act_tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def act_arctan(x: np.ndarray) -> np.ndarray:
    return np.arctan(x)


def act_softsign(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.abs(x))


ACTIVATIONS = {
    "identity": act_identity,
    "step": act_step,
    "sigmoid": act_sigmoid,
    "tanh": act_tanh,
    "arctan": act_arctan,
    "softsign": act_softsign,
}


class Neuron:
    """Один нейрон: z = w·x + b, затем f(z)."""

    def __init__(self, weights: np.ndarray, bias: float, activation: str = "step") -> None:
        if activation not in ACTIVATIONS:
            raise ValueError(f"Неизвестная активация: {activation}. Допустимо: {list(ACTIVATIONS)}")
        self.w = np.asarray(weights, dtype=np.float64)
        self.b = float(bias)
        self.activation_name = activation
        self._f = ACTIVATIONS[activation]

    def forward(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=np.float64)
        z = float(np.dot(self.w, x) + self.b)
        return float(self._f(np.array([z]))[0])


def demo_pdf_example() -> None:
    """Пример из методички: w=(0.6,-0.4), b=0.3, x=(1,0), ступенька -> 1."""
    n = Neuron(weights=[0.6, -0.4], bias=0.3, activation="step")
    y = n.forward([1.0, 0.0])
    print("Пример из методички (пороговая активация):")
    print(f"  z = 0.6*1 + (-0.4)*0 + 0.3 = 0.9  ->  f(z) = {y}")


def main() -> None:
    x = np.linspace(-2, 2, 5)
    print("Проверка активаций на векторе x =", x)
    for name, fn in ACTIVATIONS.items():
        print(f"  {name:10s}: {np.round(fn(x), 4)}")
    print()
    demo_pdf_example()

    # Графики функций активации для отчёта
    xs = np.linspace(-5, 5, 400)
    plt.figure(figsize=(10, 6))
    for name, fn in ACTIVATIONS.items():
        ys = fn(xs)
        plt.plot(xs, ys, label=name)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("Лаб. 1 — функции активации")
    plt.grid(True, alpha=0.3)
    plt.legend()
    out = __file__.replace(".py", "_activations.png")
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    print(f"График активаций сохранён: {out}")


if __name__ == "__main__":
    main()
