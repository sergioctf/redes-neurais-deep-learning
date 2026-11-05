"""Atividade 3: experimentos e seis figuras com o MLP de NumPy.

Executar da raiz do repositório:
    python docs/exercises/mlp/code/mlp_exercise.py

Scikit-learn é usado apenas para os geradores/divisões exigidos, as duas
linhas de base de regressão logística e a matriz de confusão autorizada.
O modelo treinado é sempre a classe MLP de mlp_from_scratch.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from sklearn.datasets import make_blobs, make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from mlp_from_scratch import MLP


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
OUTPUTS = ROOT / "outputs"
CLASS_COLORS = ("#2563eb", "#ea580c", "#059669")
SHALLOW_ARCH = (4, 16, 3)
DEEP_ARCH = (4, 16, 16, 3)
BINARY_ARCH = (2, 12, 1)
BINARY_LR = 0.25
BINARY_EPOCHS = 1200
MULTI_LR = 0.12
MULTI_EPOCHS = 1400


def standardize_from_train(
    x_train: np.ndarray, x_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Estatísticas vêm só do treino: nenhuma informação do teste vaza."""
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    return (x_train - mean) / std, (x_test - mean) / std, mean, std


def save_figure(filename: str) -> None:
    plt.tight_layout()
    plt.savefig(FIGURES / filename, dpi=180, bbox_inches="tight")
    plt.close()


def plot_scatter(x: np.ndarray, y: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    for class_id in (0, 1):
        mask = y == class_id
        ax.scatter(
            x[mask, 0], x[mask, 1], s=20, alpha=0.62,
            color=CLASS_COLORS[class_id], label=f"Classe {class_id}",
            rasterized=True,
        )
    ax.set(
        title="Figura 1 — Um grupo central e dois grupos da classe 1",
        xlabel="Atributo $x_1$", ylabel="Atributo $x_2$",
    )
    ax.grid(alpha=0.15)
    ax.legend(title="Classes")
    save_figure("figure_1_binary_data.png")


def plot_loss(
    losses: list[float], filename: str, figure_number: int, classes: str
) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.plot(
        np.arange(1, len(losses) + 1), losses, color="#1e3a8a",
        linewidth=2, label=f"Treino — classes {classes}",
    )
    ax.set(
        title=f"Figura {figure_number} — Entropia cruzada no treino",
        xlabel="Época", ylabel="Perda média",
    )
    ax.grid(alpha=0.2)
    ax.legend(title="Classes avaliadas")
    save_figure(filename)


def plot_boundary(
    model: MLP,
    x_test_raw: np.ndarray,
    y_test: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> None:
    margin = 0.7
    x_axis = np.linspace(x_test_raw[:, 0].min() - margin,
                         x_test_raw[:, 0].max() + margin, 280)
    y_axis = np.linspace(x_test_raw[:, 1].min() - margin,
                         x_test_raw[:, 1].max() + margin, 280)
    xx, yy = np.meshgrid(x_axis, y_axis)
    grid = np.column_stack((xx.ravel(), yy.ravel()))
    region = model.predict((grid - mean) / std).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    ax.contourf(
        xx, yy, region, levels=[-0.5, 0.5, 1.5],
        cmap=ListedColormap(("#dbeafe", "#ffedd5")), alpha=0.8,
    )
    ax.contour(xx, yy, region, levels=[0.5], colors=["#334155"], linewidths=1.3)
    for class_id, marker in ((0, "o"), (1, "^")):
        mask = y_test == class_id
        ax.scatter(
            x_test_raw[mask, 0], x_test_raw[mask, 1],
            s=34, color=CLASS_COLORS[class_id], marker=marker,
            edgecolors="white", linewidths=0.45,
            label=f"Classe {class_id} (teste)", zorder=3,
        )
    ax.set(
        title="Figura 3 — Regiões previstas pelo MLP e pontos de teste",
        xlabel="Atributo $x_1$", ylabel="Atributo $x_2$",
    )
    ax.legend(title="Classes")
    save_figure("figure_3_binary_boundary.png")


def plot_confusion(matrix: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.4))
    image = ax.imshow(matrix, cmap="Blues", vmin=0)
    ax.set_xticks(range(3), labels=["Classe 0", "Classe 1", "Classe 2"])
    ax.set_yticks(range(3), labels=["Classe 0", "Classe 1", "Classe 2"])
    ax.set(
        title="Figura 5 — Matriz de confusão no teste (MLP raso)",
        xlabel="Classe prevista", ylabel="Classe real",
    )
    threshold = matrix.max() / 2
    for row in range(3):
        for col in range(3):
            ax.text(
                col, row, str(matrix[row, col]), ha="center", va="center",
                fontsize=12, color="white" if matrix[row, col] > threshold else "#111827",
            )
    fig.colorbar(image, ax=ax, label="Número de amostras")
    save_figure("figure_5_confusion_matrix.png")


def plot_depth_comparison(shallow: list[float], deep: list[float]) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.9))
    epochs = np.arange(1, len(shallow) + 1)
    ax.plot(epochs, shallow, color="#2563eb", linewidth=2,
            label="1 camada oculta (classes 0, 1, 2)")
    ax.plot(epochs, deep, color="#ea580c", linewidth=2,
            label="2 camadas ocultas (classes 0, 1, 2)")
    ax.set(
        title="Figura 6 — Mesmos dados e orçamento, profundidades diferentes",
        xlabel="Época", ylabel="Perda média no treino",
    )
    ax.grid(alpha=0.2)
    ax.legend(title="Modelos e classes avaliadas")
    save_figure("figure_6_depth_loss.png")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12})

    # Exercício 2: parâmetros do gerador e partição exatamente como no enunciado.
    centers = np.array([[0., 0.], [3., 3.], [-3., -3.]])
    x_binary, clusters = make_blobs(
        n_samples=[500, 250, 250], centers=centers,
        cluster_std=1.2, random_state=42,
    )
    y_binary = (clusters > 0).astype(int)
    xb_train_raw, xb_test_raw, yb_train, yb_test = train_test_split(
        x_binary, y_binary, test_size=0.2, random_state=42, stratify=y_binary,
    )
    xb_train, xb_test, b_mean, b_std = standardize_from_train(
        xb_train_raw, xb_test_raw
    )
    baseline_binary = LogisticRegression(max_iter=1000)
    baseline_binary.fit(xb_train, yb_train)
    binary_baseline_accuracy = float(np.mean(
        baseline_binary.predict(xb_test) == yb_test
    ))
    binary_model = MLP(BINARY_ARCH, "binary", rng)
    binary_losses = binary_model.fit(
        xb_train, yb_train, BINARY_EPOCHS, BINARY_LR
    )
    binary_accuracy = binary_model.accuracy(xb_test, yb_test)

    plot_scatter(x_binary, y_binary)
    plot_loss(binary_losses, "figure_2_binary_loss.png", 2, "0 e 1")
    plot_boundary(binary_model, xb_test_raw, yb_test, b_mean, b_std)

    # Exercícios 3 e 4: o mesmo dataset, split e escala do treino.
    x_multi, y_multi = make_classification(
        n_samples=1500, n_features=4, n_informative=4,
        n_redundant=0, n_repeated=0, n_classes=3,
        n_clusters_per_class=2, class_sep=1.0, random_state=42,
    )
    xm_train_raw, xm_test_raw, ym_train, ym_test = train_test_split(
        x_multi, y_multi, test_size=0.2, random_state=42, stratify=y_multi,
    )
    xm_train, xm_test, m_mean, m_std = standardize_from_train(
        xm_train_raw, xm_test_raw
    )
    baseline_multi = LogisticRegression(max_iter=1000)
    baseline_multi.fit(xm_train, ym_train)
    multi_baseline_accuracy = float(np.mean(
        baseline_multi.predict(xm_test) == ym_test
    ))

    # Três sementes derivadas do RNG mestre. Cada par recebe a mesma semente;
    # a primeira matriz de pesos começa idêntica nos modelos raso e profundo.
    trial_seeds = rng.integers(1, 2**32, size=3, dtype=np.uint32).tolist()
    trials: list[dict] = []
    first_shallow: MLP | None = None
    first_shallow_losses: list[float] | None = None
    first_deep_losses: list[float] | None = None
    for seed in trial_seeds:
        shallow_model = MLP(
            SHALLOW_ARCH, "multiclass", np.random.default_rng(int(seed))
        )
        deep_model = MLP(
            DEEP_ARCH, "multiclass", np.random.default_rng(int(seed))
        )
        shallow_losses = shallow_model.fit(
            xm_train, ym_train, MULTI_EPOCHS, MULTI_LR
        )
        deep_losses = deep_model.fit(
            xm_train, ym_train, MULTI_EPOCHS, MULTI_LR
        )
        trials.append({
            "seed": int(seed),
            "shallow_accuracy": shallow_model.accuracy(xm_test, ym_test),
            "shallow_final_loss": shallow_losses[-1],
            "deep_accuracy": deep_model.accuracy(xm_test, ym_test),
            "deep_final_loss": deep_losses[-1],
        })
        if first_shallow is None:
            first_shallow = shallow_model
            first_shallow_losses = shallow_losses
            first_deep_losses = deep_losses

    assert first_shallow is not None
    assert first_shallow_losses is not None and first_deep_losses is not None
    matrix = confusion_matrix(
        ym_test, first_shallow.predict(xm_test), labels=[0, 1, 2]
    )
    pair_errors = {
        f"{a}-{b}": int(matrix[a, b] + matrix[b, a])
        for a, b in ((0, 1), (0, 2), (1, 2))
    }
    most_confused = max(pair_errors, key=pair_errors.get)
    plot_loss(first_shallow_losses, "figure_4_multiclass_loss.png", 4, "0, 1 e 2")
    plot_confusion(matrix)
    plot_depth_comparison(first_shallow_losses, first_deep_losses)

    shallow_mean = float(np.mean([t["shallow_accuracy"] for t in trials]))
    deep_mean = float(np.mean([t["deep_accuracy"] for t in trials]))
    result = {
        "seed": 42,
        "binary": {
            "samples": len(x_binary),
            "class_counts": np.bincount(y_binary).tolist(),
            "train_samples": len(xb_train),
            "test_samples": len(xb_test),
            "architecture": BINARY_ARCH,
            "learning_rate": BINARY_LR,
            "epochs": BINARY_EPOCHS,
            "linear_baseline_accuracy": binary_baseline_accuracy,
            "mlp_test_accuracy": binary_accuracy,
            "final_train_loss": binary_losses[-1],
            "train_loss_by_epoch": binary_losses,
        },
        "multiclass": {
            "samples": len(x_multi),
            "class_counts": np.bincount(y_multi).tolist(),
            "train_samples": len(xm_train),
            "test_samples": len(xm_test),
            "learning_rate": MULTI_LR,
            "epochs": MULTI_EPOCHS,
            "shallow_architecture": SHALLOW_ARCH,
            "deep_architecture": DEEP_ARCH,
            "shallow_parameters": first_shallow.parameter_count,
            "deep_parameters": sum(
                n_in * n_out + n_out
                for n_in, n_out in zip(DEEP_ARCH[:-1], DEEP_ARCH[1:])
            ),
            "linear_baseline_accuracy": multi_baseline_accuracy,
            "confusion_matrix": matrix.tolist(),
            "pair_errors": pair_errors,
            "most_confused_pair": most_confused,
            "trial_seeds": trial_seeds,
            "trials": trials,
            "shallow_mean_accuracy": shallow_mean,
            "deep_mean_accuracy": deep_mean,
            "deep_minus_shallow_mean_accuracy": deep_mean - shallow_mean,
            "shallow_train_loss_by_epoch_first_seed": first_shallow_losses,
            "deep_train_loss_by_epoch_first_seed": first_deep_losses,
            "train_mean": m_mean.tolist(),
            "train_std": m_std.tolist(),
        },
    }
    (OUTPUTS / "results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    compact = {
        "binary_baseline": binary_baseline_accuracy,
        "binary_mlp": binary_accuracy,
        "multi_baseline": multi_baseline_accuracy,
        "trials": trials,
        "confusion": matrix.tolist(),
        "most_confused_pair": most_confused,
        "shallow_mean": shallow_mean,
        "deep_mean": deep_mean,
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
