"""Atividade 2: perceptron binário e algoritmo pocket, sem modelo pronto.

Execução, a partir da raiz do repositório:
    python docs/exercises/perceptron/code/perceptron_exercise.py

A única instância de RNG é criada em main() e passada a toda operação aleatória.
Os dados são visitados na ordem em que foram gerados (classe 0, depois classe 1).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
OUTPUTS = ROOT / "outputs"
COLORS = ("#2563eb", "#ea580c")


@dataclass
class TrainingResult:
    weights: np.ndarray
    bias: float
    accuracy_by_epoch: list[float]
    updates_by_epoch: list[int]
    pocket_weights: np.ndarray | None
    pocket_bias: float | None
    pocket_accuracy: float | None
    pocket_epoch: int | None
    pocket_update: int | None
    pocket_accuracy_by_epoch: list[float]

    @property
    def epochs(self) -> int:
        return len(self.accuracy_by_epoch)

    @property
    def final_accuracy(self) -> float:
        return self.accuracy_by_epoch[-1]


def make_gaussians(
    rng: np.random.Generator,
    mean_0: tuple[float, float],
    mean_1: tuple[float, float],
    covariance: np.ndarray,
    samples_per_class: int = 1000,
) -> tuple[np.ndarray, np.ndarray]:
    """Gera as duas classes em ordem fixa, com a covariância especificada."""
    x_0 = rng.multivariate_normal(mean_0, covariance, size=samples_per_class)
    x_1 = rng.multivariate_normal(mean_1, covariance, size=samples_per_class)
    x = np.vstack((x_0, x_1))
    y = np.concatenate(
        (np.zeros(samples_per_class, dtype=np.int8),
         np.ones(samples_per_class, dtype=np.int8))
    )
    return x, y


def step(z: np.ndarray | float) -> np.ndarray | np.int8:
    """Ativação: 1 quando z >= 0; 0 nos demais casos."""
    return np.int8(np.asarray(z) >= 0)


def predict(x: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray | np.int8:
    """Predição implementada diretamente como step(w · x + b)."""
    return step(x @ weights + bias)


def accuracy(x: np.ndarray, y: np.ndarray, weights: np.ndarray, bias: float) -> float:
    """Fração de acertos sobre todas as amostras, sem biblioteca de métricas."""
    return float(np.mean(predict(x, weights, bias) == y))


def overlap_certificate(x: np.ndarray, y: np.ndarray) -> dict:
    """Encontra um ponto da classe 1 dentro de um triângulo da classe 0.

    Um separador linear que classifique os três vértices como 0 também teria
    de classificar toda combinação convexa deles como 0. O ponto testemunha
    de classe 1 torna a separação perfeita matematicamente impossível.
    """
    x_0, x_1 = x[y == 0], x[y == 1]
    triangulation = mtri.Triangulation(x_0[:, 0], x_0[:, 1])
    triangle_for_point = triangulation.get_trifinder()(x_1[:, 0], x_1[:, 1])
    candidates = np.flatnonzero(triangle_for_point >= 0)
    if candidates.size == 0:
        raise ValueError("Nenhum ponto da classe 1 no fecho convexo da classe 0.")
    index_1 = int(candidates[0])
    indices_0 = triangulation.triangles[triangle_for_point[index_1]]
    vertices = x_0[indices_0]
    first_two = np.linalg.solve(
        np.column_stack((vertices[0] - vertices[2], vertices[1] - vertices[2])),
        x_1[index_1] - vertices[2],
    )
    coefficients = np.array((first_two[0], first_two[1], 1 - first_two.sum()))
    if not np.all(coefficients > 0):
        raise AssertionError("O ponto escolhido deveria estar no interior do triângulo.")
    return {
        "class_1_index": index_1,
        "class_1_point": x_1[index_1].tolist(),
        "class_0_indices": indices_0.tolist(),
        "class_0_vertices": vertices.tolist(),
        "convex_coefficients": coefficients.tolist(),
    }


def train_perceptron(
    x: np.ndarray,
    y: np.ndarray,
    initial_weights: np.ndarray,
    *,
    learning_rate: float = 0.01,
    initial_bias: float = 0.0,
    max_epochs: int = 100,
    keep_pocket: bool = False,
) -> TrainingResult:
    """Treina o mesmo perceptron nos dois exercícios.

    O pocket só guarda uma cópia do melhor estado. Ele é comparado a cada
    atualização, inclusive dentro da época, sem alterar a regra de treino.
    Empates conservam o primeiro estado que alcançou aquela acurácia.
    """
    weights = np.array(initial_weights, dtype=float, copy=True)
    bias = float(initial_bias)
    if weights.shape != (2,) or x.shape[1] != 2 or x.shape[0] != y.size:
        raise ValueError("Esperados X com duas features e um rótulo por linha.")
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Os rótulos precisam ser 0 e 1.")

    pocket_weights = weights.copy() if keep_pocket else None
    pocket_bias = bias if keep_pocket else None
    pocket_accuracy = accuracy(x, y, weights, bias) if keep_pocket else None
    pocket_epoch = 0 if keep_pocket else None
    pocket_update = 0 if keep_pocket else None
    accuracy_by_epoch: list[float] = []
    updates_by_epoch: list[int] = []
    pocket_accuracy_by_epoch: list[float] = []
    total_updates = 0

    for epoch in range(1, max_epochs + 1):
        updates = 0
        for sample, label in zip(x, y):
            prediction = int(predict(sample, weights, bias))
            error = int(label) - prediction  # 0, +1 ou -1 para rótulos 0/1.
            if error == 0:
                continue

            weights += learning_rate * error * sample
            bias += learning_rate * error
            updates += 1
            total_updates += 1

            if keep_pocket:
                candidate_accuracy = accuracy(x, y, weights, bias)
                assert pocket_accuracy is not None
                if candidate_accuracy > pocket_accuracy:
                    pocket_accuracy = candidate_accuracy
                    pocket_weights = weights.copy()  # Evita aliasing com o estado atual.
                    pocket_bias = bias
                    pocket_epoch = epoch
                    pocket_update = total_updates

        # Acurácia medida após cada passagem completa, inclusive a última.
        accuracy_by_epoch.append(accuracy(x, y, weights, bias))
        updates_by_epoch.append(updates)
        if keep_pocket:
            assert pocket_accuracy is not None
            pocket_accuracy_by_epoch.append(pocket_accuracy)
        if updates == 0:
            break

    return TrainingResult(
        weights=weights,
        bias=bias,
        accuracy_by_epoch=accuracy_by_epoch,
        updates_by_epoch=updates_by_epoch,
        pocket_weights=pocket_weights,
        pocket_bias=pocket_bias,
        pocket_accuracy=pocket_accuracy,
        pocket_epoch=pocket_epoch,
        pocket_update=pocket_update,
        pocket_accuracy_by_epoch=pocket_accuracy_by_epoch,
    )


def draw_classes(ax: plt.Axes, x: np.ndarray, y: np.ndarray) -> None:
    """Nuvens e legenda de classes usadas em todas as figuras de dispersão."""
    for label, color in enumerate(COLORS):
        mask = y == label
        ax.scatter(
            x[mask, 0], x[mask, 1], s=13, alpha=0.42,
            color=color, label=f"Classe {label}", rasterized=True,
        )
    ax.set_xlabel("Atributo $x_1$")
    ax.set_ylabel("Atributo $x_2$")
    ax.grid(alpha=0.18)


def draw_boundary(
    ax: plt.Axes,
    weights: np.ndarray,
    bias: float,
    x_limits: tuple[float, float],
    *,
    label: str,
    color: str,
    linestyle: str = "-",
) -> None:
    """Traça w1*x1 + w2*x2 + b = 0, inclusive se a reta for vertical."""
    if abs(weights[1]) > 1e-12:
        x_line = np.array(x_limits)
        y_line = -(weights[0] * x_line + bias) / weights[1]
        ax.plot(x_line, y_line, color=color, linestyle=linestyle,
                linewidth=2.3, label=label, zorder=4)
    elif abs(weights[0]) > 1e-12:
        ax.axvline(-bias / weights[0], color=color, linestyle=linestyle,
                   linewidth=2.3, label=label, zorder=4)


def save_figure(name: str) -> None:
    plt.tight_layout()
    plt.savefig(FIGURES / name, dpi=180, bbox_inches="tight")
    plt.close()


def make_figures(
    x_sep: np.ndarray,
    y_sep: np.ndarray,
    sep: TrainingResult,
    x_ov: np.ndarray,
    y_ov: np.ndarray,
    ov: TrainingResult,
) -> None:
    """Cria exatamente as Figuras 1 a 6 solicitadas no enunciado."""
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12})

    fig, ax = plt.subplots(figsize=(7, 5.4))
    draw_classes(ax, x_sep, y_sep)
    ax.set_title("Figura 1 — Classes gaussianas quase separáveis")
    ax.legend(title="Classes")
    save_figure("figure_1_separable_data.png")

    fig, ax = plt.subplots(figsize=(7, 5.4))
    draw_classes(ax, x_sep, y_sep)
    miss = predict(x_sep, sep.weights, sep.bias) != y_sep
    if np.any(miss):
        ax.scatter(x_sep[miss, 0], x_sep[miss, 1], s=70, facecolors="none",
                   edgecolors="black", linewidths=1.3,
                   label=f"Erros ({int(miss.sum())})", zorder=5)
    else:
        ax.text(0.98, 0.03, "Erros: 0", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=10,
                bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"})
    x_limits = tuple(ax.get_xlim())
    y_limits = tuple(ax.get_ylim())
    draw_boundary(ax, sep.weights, sep.bias, x_limits,
                  label="Fronteira $\\eta=0{,}01$", color="#111827")
    ax.set_xlim(x_limits)
    ax.set_ylim(y_limits)
    ax.set_title("Figura 2 — Fronteira final sobre os dados separáveis")
    ax.legend(title="Classes e fronteira")
    save_figure("figure_2_separable_boundary.png")

    epochs = np.arange(1, sep.epochs + 1)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(epochs, sep.accuracy_by_epoch, color="#111827", marker="o",
            label="Total (classes 0 e 1)")
    ax.set_title("Figura 3 — Acurácia após cada época, dados separáveis")
    ax.set_xlabel("Época")
    ax.set_ylabel("Acurácia (classes 0 e 1)")
    ax.set_ylim(0.45, 1.01)
    ax.set_xticks(epochs)
    ax.grid(alpha=0.22)
    ax.legend(title="Classes avaliadas")
    save_figure("figure_3_separable_accuracy.png")

    fig, ax = plt.subplots(figsize=(7, 5.4))
    draw_classes(ax, x_ov, y_ov)
    ax.set_title("Figura 4 — Classes gaussianas sobrepostas")
    ax.legend(title="Classes")
    save_figure("figure_4_overlapping_data.png")

    assert ov.pocket_weights is not None and ov.pocket_bias is not None
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.5), sharex=True, sharey=True)
    # Os painéis distinguem quais erros pertencem a cada conjunto de pesos.
    for ax, weights, bias, title in (
        (axes[0], ov.weights, ov.bias, "Iteração final"),
        (axes[1], ov.pocket_weights, ov.pocket_bias, "Pocket"),
    ):
        draw_classes(ax, x_ov, y_ov)
        miss = predict(x_ov, weights, bias) != y_ov
        ax.scatter(x_ov[miss, 0], x_ov[miss, 1], s=37,
                   facecolors="none", edgecolors="#111827",
                   linewidths=0.55, label=f"Erros deste modelo ({miss.sum()})",
                   zorder=3, rasterized=True)
        x_limits = tuple(ax.get_xlim())
        y_limits = tuple(ax.get_ylim())
        draw_boundary(ax, ov.weights, ov.bias, x_limits,
                      label="Final", color="#991b1b")
        draw_boundary(ax, ov.pocket_weights, ov.pocket_bias, x_limits,
                      label="Pocket", color="#047857", linestyle="--")
        ax.set_xlim(x_limits)
        ax.set_ylim(y_limits)
        ax.set_title(title)
        ax.legend(title="Classes, erros e fronteiras", fontsize=8)
    fig.suptitle("Figura 5 — Fronteiras final e pocket, com seus erros", fontsize=13)
    save_figure("figure_5_final_and_pocket.png")

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    epochs = np.arange(1, ov.epochs + 1)
    ax.plot(epochs, ov.accuracy_by_epoch, color="#991b1b", alpha=0.85,
            label="Pesos atuais (classes 0 e 1)")
    ax.plot(epochs, ov.pocket_accuracy_by_epoch, color="#047857", linewidth=2,
            label="Pocket (classes 0 e 1)")
    ax.set_title("Figura 6 — Acurácia atual e melhor acurácia até a época")
    ax.set_xlabel("Época")
    ax.set_ylabel("Acurácia")
    ax.set_xlim(1, ov.epochs)
    ax.set_ylim(0.45, 0.8)
    ax.grid(alpha=0.22)
    ax.legend(title="Classes avaliadas: 0 e 1")
    save_figure("figure_6_accuracy_and_pocket.png")


def as_record(result: TrainingResult) -> dict:
    """Valores em JSON para auditar as afirmações do relatório."""
    return {
        "weights": result.weights.tolist(),
        "bias": result.bias,
        "epochs": result.epochs,
        "final_accuracy": result.final_accuracy,
        "accuracy_by_epoch": result.accuracy_by_epoch,
        "updates_by_epoch": result.updates_by_epoch,
        "pocket_weights": (
            result.pocket_weights.tolist() if result.pocket_weights is not None else None
        ),
        "pocket_bias": result.pocket_bias,
        "pocket_accuracy": result.pocket_accuracy,
        "pocket_epoch": result.pocket_epoch,
        "pocket_update": result.pocket_update,
        "pocket_accuracy_by_epoch": result.pocket_accuracy_by_epoch,
    }


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)

    x_sep, y_sep = make_gaussians(
        rng, (1.5, 1.5), (5.0, 5.0), np.diag([0.5, 0.5])
    )
    initial_sep = rng.normal(0, 0.01, size=2)
    sep = train_perceptron(x_sep, y_sep, initial_sep, learning_rate=0.01)
    # Mesmo vetor inicial e mesma ordem: só eta muda na comparação.
    sep_eta_1 = train_perceptron(x_sep, y_sep, initial_sep, learning_rate=1.0)

    x_ov, y_ov = make_gaussians(
        rng, (3.0, 3.0), (4.0, 4.0), np.diag([1.5, 1.5])
    )
    initial_ov = rng.normal(0, 0.01, size=2)
    ov = train_perceptron(x_ov, y_ov, initial_ov,
                          learning_rate=0.01, keep_pocket=True)

    make_figures(x_sep, y_sep, sep, x_ov, y_ov, ov)
    results = {
        "seed": 42,
        "samples_per_class": 1000,
        "sample_order": "class_0_then_class_1_without_shuffling",
        "exercise_1": {
            "initial_weights": initial_sep.tolist(),
            "eta_0_01": as_record(sep),
            "eta_1_0": as_record(sep_eta_1),
            "direction_eta_0_01": (sep.weights / np.linalg.norm(sep.weights)).tolist(),
            "direction_eta_1_0": (
                sep_eta_1.weights / np.linalg.norm(sep_eta_1.weights)
            ).tolist(),
        },
        "exercise_2": {
            "initial_weights": initial_ov.tolist(),
            "eta_0_01": as_record(ov),
            "nonseparability_certificate": overlap_certificate(x_ov, y_ov),
            "class_0_mean": x_ov[y_ov == 0].mean(axis=0).tolist(),
            "class_1_mean": x_ov[y_ov == 1].mean(axis=0).tolist(),
            "mean_sample_norm": float(np.mean(np.linalg.norm(x_ov, axis=1))),
        },
    }
    output_path = OUTPUTS / "results.json"
    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
    print(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"Resultados e figuras salvos em: {ROOT}")


if __name__ == "__main__":
    main()
