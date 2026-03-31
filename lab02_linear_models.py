"""
Лабораторная 2: датасет, линейная регрессия, метрики, предобработка, снова метрики.
California Housing: сначала простая модель на исходных признаках,
затем StandardScaler + полиномиальные признаки + Ridge — типичная предобработка.
Запуск: python lab02_linear_models.py
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


def metrics_report(y_true: np.ndarray, y_pred: np.ndarray, title: str) -> tuple[float, float, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)
    print(f"\n{title}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R2:   {r2:.4f}")
    return mae, rmse, r2


def main() -> None:
    X, y = fetch_california_housing(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Шаги 1–4: обучение и метрики без сложной предобработки
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    pred_raw = lr.predict(X_test)
    mae_raw, rmse_raw, r2_raw = metrics_report(
        y_test, pred_raw, "До предобработки: LinearRegression на исходных признаках"
    )

    # Шаги 5–6: предобработка (масштаб + нелинейные признаки), снова обучение и метрики
    prep = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("ridge", Ridge(alpha=5.0, random_state=42)),
        ]
    )
    prep.fit(X_train, y_train)
    pred_prep = prep.predict(X_test)
    mae_p, rmse_p, r2_p = metrics_report(
        y_test, pred_prep, "После предобработки: Scaler + Poly(2) + Ridge"
    )

    # Картинка 1: scatter истинных и предсказанных значений
    fig1, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
    lims = [min(y_test.min(), pred_raw.min(), pred_prep.min()), max(y_test.max(), pred_raw.max(), pred_prep.max())]
    for ax, pred, title in zip(
        axes,
        [pred_raw, pred_prep],
        ["До предобработки", "После предобработки"],
    ):
        ax.scatter(y_test, pred, s=8, alpha=0.4)
        ax.plot(lims, lims, "r--", linewidth=1)
        ax.set_title(title)
        ax.set_xlabel("y (истинное)")
        ax.set_ylabel("ŷ (предсказание)")
        ax.grid(True, alpha=0.2)
    plt.tight_layout()
    out1 = __file__.replace(".py", "_scatter.png")
    plt.savefig(out1, dpi=120)
    print(f"Scatter-графики сохранены: {out1}")

    # Картинка 2: бар‑чарт метрик
    labels = ["MAE", "RMSE", "R2"]
    before = [mae_raw, rmse_raw, r2_raw]
    after = [mae_p, rmse_p, r2_p]

    x = np.arange(len(labels))
    width = 0.35
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.bar(x - width / 2, before, width, label="До")
    ax2.bar(x + width / 2, after, width, label="После")
    ax2.set_xticks(x, labels)
    ax2.set_title("Лаб. 2 — метрики до/после предобработки")
    ax2.grid(True, axis="y", alpha=0.2)
    ax2.legend()
    plt.tight_layout()
    out2 = __file__.replace(".py", "_metrics.png")
    plt.savefig(out2, dpi=120)
    print(f"Бар-чарт метрик сохранён: {out2}")


if __name__ == "__main__":
    main()
