"""Gera todos os resultados do exercício individual 1 — Data.

Execute este arquivo a partir da raiz do repositório:

    python docs/exercises/data/code/data_exercise.py

O programa usa uma única instância de ``np.random.Generator`` para todos os
dados sintéticos, salva as seis figuras solicitadas e grava as medidas usadas
no relatório em ``outputs/``.
"""

from __future__ import annotations

import hashlib
import json
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


SEED = 42
SAMPLES_PER_CLOUD = 100
SCALES = (0.5, 1.0, 2.0, 4.0)

CLASS_MEANS = np.array(
    [
        [2.0, 3.0],
        [5.0, 6.0],
        [8.0, 1.0],
        [15.0, 4.0],
    ]
)

CLASS_STDS = np.array(
    [
        [0.8, 2.5],
        [1.2, 1.9],
        [0.9, 0.9],
        [0.5, 2.0],
    ]
)

MU_A = np.zeros(5)
MU_B = np.full(5, 1.5)

SIGMA_A = np.array(
    [
        [1.0, 0.8, 0.1, 0.0, 0.0],
        [0.8, 1.0, 0.3, 0.0, 0.0],
        [0.1, 0.3, 1.0, 0.5, 0.0],
        [0.0, 0.0, 0.5, 1.0, 0.2],
        [0.0, 0.0, 0.0, 0.2, 1.0],
    ]
)

SIGMA_B = np.array(
    [
        [1.5, -0.7, 0.2, 0.0, 0.0],
        [-0.7, 1.5, 0.4, 0.0, 0.0],
        [0.2, 0.4, 1.5, 0.6, 0.0],
        [0.0, 0.0, 0.6, 1.5, 0.3],
        [0.0, 0.0, 0.0, 0.3, 1.5],
    ]
)

SPENDING_COLUMNS = [
    "RoomService",
    "FoodCourt",
    "ShoppingMall",
    "Spa",
    "VRDeck",
]

CATEGORICAL_COLUMNS = ["HomePlanet", "CryoSleep", "Destination", "VIP"]
EXPECTED_DATA_COLUMNS = [
    "PassengerId",
    "HomePlanet",
    "CryoSleep",
    "Cabin",
    "Destination",
    "Age",
    "VIP",
    *SPENDING_COLUMNS,
    "Name",
    "Transported",
]
EXPECTED_DATA_ROWS = 8_693
EXPECTED_DATA_SHA256 = "92515331df73abce3b7c01dba35c33971027268bd84609bc6593ca43b18e649f"

EXERCISE_DIR = Path(__file__).resolve().parent.parent
FIGURES_DIR = EXERCISE_DIR / "figures"
OUTPUTS_DIR = EXERCISE_DIR / "outputs"
DATA_PATH = EXERCISE_DIR / "data" / "train.csv"

CLASS_COLORS = sns.color_palette("colorblind", 4)


def generate_clouds(
    rng: np.random.Generator, scale: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """Gera as quatro nuvens gaussianas bidimensionais do enunciado."""
    points = []
    labels = []

    for class_id, (mean, std) in enumerate(zip(CLASS_MEANS, CLASS_STDS)):
        cloud = rng.normal(
            loc=mean,
            scale=std * scale,
            size=(SAMPLES_PER_CLOUD, 2),
        )
        points.append(cloud)
        labels.append(np.full(SAMPLES_PER_CLOUD, class_id, dtype=int))

    return np.vstack(points), np.concatenate(labels)


def shared_limits(datasets: dict[float, tuple[np.ndarray, np.ndarray]]) -> tuple:
    """Calcula limites comuns para comparar todos os fatores de escala."""
    all_points = np.vstack([points for points, _ in datasets.values()])
    minimum = all_points.min(axis=0)
    maximum = all_points.max(axis=0)
    padding = 0.05 * (maximum - minimum)
    return (
        (minimum[0] - padding[0], maximum[0] + padding[0]),
        (minimum[1] - padding[1], maximum[1] + padding[1]),
    )


def plot_class_points(
    ax: plt.Axes,
    points: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
    mark_centers: bool = False,
) -> None:
    """Desenha pontos por classe e, opcionalmente, os centros teóricos."""
    for class_id, class_name in enumerate(class_names):
        selected = labels == class_id
        ax.scatter(
            points[selected, 0],
            points[selected, 1],
            s=24,
            alpha=0.72,
            color=CLASS_COLORS[class_id],
            label=class_name,
        )

    if mark_centers:
        ax.scatter(
            CLASS_MEANS[:, 0],
            CLASS_MEANS[:, 1],
            s=155,
            marker="X",
            color="white",
            edgecolor="black",
            linewidth=1.5,
            zorder=5,
            label="Centros teóricos",
        )


def add_nearest_center_boundaries(
    ax: plt.Axes,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
) -> None:
    """Adiciona um esboço reprodutível das fronteiras do centro mais próximo."""
    x_grid = np.linspace(*x_limits, 500)
    y_grid = np.linspace(*y_limits, 500)
    xx, yy = np.meshgrid(x_grid, y_grid)
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    distances = np.linalg.norm(grid[:, None, :] - CLASS_MEANS[None, :, :], axis=2)
    nearest = np.argmin(distances, axis=1).reshape(xx.shape)
    ax.contour(
        xx,
        yy,
        nearest,
        levels=[0.5, 1.5, 2.5],
        colors="black",
        linewidths=1.3,
        linestyles="--",
        alpha=0.8,
    )


def create_figure_1(points: np.ndarray, labels: np.ndarray) -> None:
    """Figura 1: nuvens originais, centros e esboço das fronteiras."""
    x_min, y_min = points.min(axis=0)
    x_max, y_max = points.max(axis=0)
    x_limits = (x_min - 1.0, x_max + 1.0)
    y_limits = (y_min - 1.0, y_max + 1.0)

    fig, ax = plt.subplots(figsize=(10, 6.5))
    plot_class_points(
        ax,
        points,
        labels,
        ["Classe 0", "Classe 1", "Classe 2", "Classe 3"],
        mark_centers=True,
    )
    add_nearest_center_boundaries(ax, x_limits, y_limits)

    handles, legend_labels = ax.get_legend_handles_labels()
    handles.append(Line2D([0], [0], color="black", linestyle="--"))
    legend_labels.append("Esboço: centro mais próximo")
    ax.legend(handles, legend_labels, loc="upper left", frameon=True)
    ax.set(
        title="Figura 1 — Nuvens gaussianas originais (s = 1)",
        xlabel="Feature x",
        ylabel="Feature y",
        xlim=x_limits,
        ylim=y_limits,
    )
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "figure_1_clouds.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def create_figure_2(datasets: dict[float, tuple[np.ndarray, np.ndarray]]) -> None:
    """Figura 2: quatro fatores de escala com os mesmos limites de eixos."""
    x_limits, y_limits = shared_limits(datasets)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True, sharey=True)

    for ax, scale in zip(axes.flat, SCALES):
        points, labels = datasets[scale]
        plot_class_points(
            ax,
            points,
            labels,
            ["Classe 0", "Classe 1", "Classe 2", "Classe 3"],
        )
        ax.scatter(
            CLASS_MEANS[:, 0],
            CLASS_MEANS[:, 1],
            s=90,
            marker="X",
            color="white",
            edgecolor="black",
            linewidth=1.2,
            zorder=5,
        )
        ax.set_title(f"Escala s = {scale:.1f}")
        ax.set_xlim(x_limits)
        ax.set_ylim(y_limits)
        ax.grid(alpha=0.2)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    handles.append(
        Line2D(
            [0],
            [0],
            marker="X",
            color="white",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=9,
            linestyle="None",
        )
    )
    labels.append("Centros teóricos")
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.012),
        ncol=5,
        frameon=True,
    )
    fig.suptitle("Figura 2 — Efeito do fator de escala sobre as quatro classes", fontsize=15)
    fig.supxlabel("Feature x", y=0.072)
    fig.supylabel("Feature y")
    fig.tight_layout(rect=(0.02, 0.11, 1.0, 0.96))
    fig.savefig(FIGURES_DIR / "figure_2_scales.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def separation_ratios() -> pd.DataFrame:
    """Calcula os seis r_ij usando os parâmetros teóricos em s = 1."""
    mean_stds = CLASS_STDS.mean(axis=1)
    rows = []

    for class_i, class_j in combinations(range(len(CLASS_MEANS)), 2):
        center_distance = np.linalg.norm(CLASS_MEANS[class_i] - CLASS_MEANS[class_j])
        ratio = center_distance / (mean_stds[class_i] + mean_stds[class_j])
        rows.append(
            {
                "pair": f"({class_i}, {class_j})",
                "center_distance": center_distance,
                "r_ij_s_1": ratio,
            }
        )

    return pd.DataFrame(rows)


def mixing_rate(
    points: np.ndarray, labels: np.ndarray
) -> tuple[float, dict[int, float]]:
    """Mede erros do critério geométrico do centro de classe mais próximo."""
    distances = np.linalg.norm(points[:, None, :] - CLASS_MEANS[None, :, :], axis=2)
    nearest_center = np.argmin(distances, axis=1)
    mixed = nearest_center != labels
    per_class = {
        class_id: float(mixed[labels == class_id].mean())
        for class_id in range(len(CLASS_MEANS))
    }
    return float(mixed.mean()), per_class


def create_figure_3(
    overall_rates: dict[float, float],
    class_rates: dict[float, dict[int, float]],
) -> None:
    """Figura 3: mixing rate total e por classe em função da escala."""
    scales = np.array(SCALES)
    fig, ax = plt.subplots(figsize=(9, 5.8))

    for class_id in range(len(CLASS_MEANS)):
        values = [class_rates[scale][class_id] for scale in SCALES]
        ax.plot(
            scales,
            values,
            marker="o",
            linestyle="--",
            linewidth=1.2,
            color=CLASS_COLORS[class_id],
            alpha=0.8,
            label=f"Classe {class_id}",
        )

    ax.plot(
        scales,
        [overall_rates[scale] for scale in SCALES],
        marker="o",
        color="black",
        linewidth=2.6,
        label="Total",
    )
    ax.set(
        title="Figura 3 — Mixing rate por fator de escala",
        xlabel="Fator de escala s",
        ylabel="Mixing rate",
        xticks=scales,
        ylim=(-0.015, max(overall_rates.values()) * 1.35 + 0.02),
    )
    ax.grid(alpha=0.25)
    ax.legend(ncol=3, frameon=True)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "figure_3_mixing_rate.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def generate_shifted_gaussians(
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Gera o Dataset I, com duas gaussianas multivariadas em cinco dimensões."""
    class_a = rng.multivariate_normal(MU_A, SIGMA_A, size=500, check_valid="raise")
    class_b = rng.multivariate_normal(MU_B, SIGMA_B, size=500, check_valid="raise")
    points = np.vstack([class_a, class_b])
    labels = np.concatenate([np.zeros(500, dtype=int), np.ones(500, dtype=int)])
    return points, labels


def sample_sphere_directions(rng: np.random.Generator, size: int) -> np.ndarray:
    """Amostra direções uniformes na esfera unitária de R^5."""
    vectors = rng.normal(size=(size, 5))
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def generate_concentric_shells(
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Gera o Dataset II, formado por um núcleo e uma casca concêntricos."""
    directions_c = sample_sphere_directions(rng, 500)
    radii_c = rng.normal(loc=2.0, scale=0.4, size=500)
    class_c = radii_c[:, None] * directions_c

    directions_d = sample_sphere_directions(rng, 500)
    radii_d = rng.normal(loc=5.0, scale=0.4, size=500)
    class_d = radii_d[:, None] * directions_d

    points = np.vstack([class_c, class_d])
    labels = np.concatenate([np.zeros(500, dtype=int), np.ones(500, dtype=int)])
    return points, labels


def class_center_distance(points: np.ndarray, labels: np.ndarray) -> float:
    """Calcula a distância euclidiana entre os centros amostrais em 5D."""
    first_center = points[labels == 0].mean(axis=0)
    second_center = points[labels == 1].mean(axis=0)
    return float(np.linalg.norm(first_center - second_center))


def create_figure_4(
    dataset_i: tuple[np.ndarray, np.ndarray],
    dataset_ii: tuple[np.ndarray, np.ndarray],
) -> tuple[float, float]:
    """Figura 4: projeções PCA e variância explicada dos dois datasets."""
    datasets = [dataset_i, dataset_ii]
    titles = ["Dataset I — Gaussianas deslocadas", "Dataset II — Cascas concêntricas"]
    class_names = [["Classe A", "Classe B"], ["Classe C", "Classe D"]]
    explained_variances = []

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.7))
    for ax, (points, labels), title, names in zip(
        axes, datasets, titles, class_names
    ):
        pca = PCA(n_components=2)
        projected = pca.fit_transform(points)
        explained_variances.append(float(pca.explained_variance_ratio_.sum()))

        for class_id, class_name in enumerate(names):
            selected = labels == class_id
            ax.scatter(
                projected[selected, 0],
                projected[selected, 1],
                s=22,
                alpha=0.62,
                color=CLASS_COLORS[class_id],
                label=class_name,
            )

        ax.set(
            title=f"{title}\nPC1 + PC2 = {explained_variances[-1]:.2%}",
            xlabel="Componente principal 1",
            ylabel="Componente principal 2",
        )
        ax.grid(alpha=0.2)
        ax.legend(frameon=True)

    fig.suptitle("Figura 4 — Projeções PCA em duas dimensões", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIGURES_DIR / "figure_4_pca.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    return explained_variances[0], explained_variances[1]


def create_figure_5(
    dataset_i: tuple[np.ndarray, np.ndarray],
    dataset_ii: tuple[np.ndarray, np.ndarray],
) -> None:
    """Figura 5: histogramas dos raios, calculados no espaço original 5D."""
    datasets = [dataset_i, dataset_ii]
    titles = ["Dataset I — Gaussianas deslocadas", "Dataset II — Cascas concêntricas"]
    class_names = [["Classe A", "Classe B"], ["Classe C", "Classe D"]]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, (points, labels), title, names in zip(
        axes, datasets, titles, class_names
    ):
        radii = np.linalg.norm(points, axis=1)
        common_bins = np.linspace(radii.min(), radii.max(), 28)
        for class_id, class_name in enumerate(names):
            ax.hist(
                radii[labels == class_id],
                bins=common_bins,
                density=True,
                alpha=0.58,
                color=CLASS_COLORS[class_id],
                label=class_name,
            )
        ax.set(
            title=title,
            xlabel=r"Raio em 5D: $\Vert x \Vert$",
            ylabel="Densidade",
        )
        ax.grid(axis="y", alpha=0.2)
        ax.legend(frameon=True)

    fig.suptitle("Figura 5 — Distribuição dos raios no espaço original", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIGURES_DIR / "figure_5_radii.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def validate_spaceship_file(path: Path) -> pd.DataFrame:
    """Valida identidade e estrutura do train.csv antes de usá-lo."""
    if not path.exists():
        raise FileNotFoundError(
            "train.csv não encontrado. Siga docs/exercises/data/data/README.md."
        )

    file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    data = pd.read_csv(path)

    if data.columns.tolist() != EXPECTED_DATA_COLUMNS:
        raise ValueError("O cabeçalho do train.csv não corresponde ao dataset esperado.")
    if len(data) != EXPECTED_DATA_ROWS:
        raise ValueError(f"Esperadas {EXPECTED_DATA_ROWS} linhas; encontradas {len(data)}.")
    if file_hash != EXPECTED_DATA_SHA256:
        raise ValueError("O SHA-256 do train.csv não corresponde ao arquivo oficial esperado.")

    return data


def preprocess_spaceship(
    x_train_raw: pd.DataFrame,
    x_test_raw: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Ajusta todas as transformações no treino e apenas as aplica ao teste."""
    base_numerical = ["Age", *SPENDING_COLUMNS]

    numerical_imputer = SimpleImputer(strategy="median")
    train_numerical = pd.DataFrame(
        numerical_imputer.fit_transform(x_train_raw[base_numerical]),
        columns=base_numerical,
    )
    test_numerical = pd.DataFrame(
        numerical_imputer.transform(x_test_raw[base_numerical]),
        columns=base_numerical,
    )

    for frame in (train_numerical, test_numerical):
        frame["TotalSpend"] = frame[SPENDING_COLUMNS].sum(axis=1)

    monetary_columns = [*SPENDING_COLUMNS, "TotalSpend"]
    train_numerical[monetary_columns] = np.log1p(
        train_numerical[monetary_columns]
    )
    test_numerical[monetary_columns] = np.log1p(test_numerical[monetary_columns])

    numerical_columns = ["Age", *monetary_columns]
    scaler = MinMaxScaler(feature_range=(-1, 1), clip=True)
    train_scaled = pd.DataFrame(
        scaler.fit_transform(train_numerical[numerical_columns]),
        columns=numerical_columns,
    )
    test_scaled = pd.DataFrame(
        scaler.transform(test_numerical[numerical_columns]),
        columns=numerical_columns,
    )

    categorical_imputer = SimpleImputer(strategy="most_frequent")
    train_categorical = categorical_imputer.fit_transform(
        x_train_raw[CATEGORICAL_COLUMNS]
    )
    test_categorical = categorical_imputer.transform(x_test_raw[CATEGORICAL_COLUMNS])

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    train_encoded = encoder.fit_transform(train_categorical)
    test_encoded = encoder.transform(test_categorical)
    encoded_columns = encoder.get_feature_names_out(CATEGORICAL_COLUMNS).tolist()

    train_encoded_frame = pd.DataFrame(train_encoded, columns=encoded_columns)
    test_encoded_frame = pd.DataFrame(test_encoded, columns=encoded_columns)
    x_train = pd.concat([train_scaled, train_encoded_frame], axis=1)
    x_test = pd.concat([test_scaled, test_encoded_frame], axis=1)

    preprocessing_details = {
        "numerical_imputation_medians": {
            column: float(value)
            for column, value in zip(base_numerical, numerical_imputer.statistics_)
        },
        "categorical_imputation_modes": {
            column: str(value)
            for column, value in zip(CATEGORICAL_COLUMNS, categorical_imputer.statistics_)
        },
        "encoded_columns": encoded_columns,
        "all_feature_names": x_train.columns.tolist(),
    }
    return x_train, x_test, preprocessing_details


def create_figure_6(
    x_train_raw: pd.DataFrame,
    x_train_processed: pd.DataFrame,
    y_train: pd.Series,
) -> None:
    """Figura 6: FoodCourt antes e depois de imputação, log1p e scaling."""
    raw_frame = x_train_raw[["FoodCourt"]].copy()
    raw_frame["Transported"] = y_train
    processed_frame = pd.DataFrame(
        {
            "FoodCourt": x_train_processed["FoodCourt"],
            "Transported": y_train.reset_index(drop=True),
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
    for label, color in zip((False, True), CLASS_COLORS[:2]):
        raw_values = raw_frame.loc[
            raw_frame["Transported"] == label, "FoodCourt"
        ].dropna()
        axes[0].hist(
            raw_values,
            bins=45,
            density=True,
            alpha=0.55,
            color=color,
            label=f"Transported = {label}",
        )

        processed_values = processed_frame.loc[
            processed_frame["Transported"] == label, "FoodCourt"
        ]
        axes[1].hist(
            processed_values,
            bins=35,
            density=True,
            alpha=0.55,
            color=color,
            label=f"Transported = {label}",
        )

    axes[0].set(
        title="Antes: valores monetários brutos",
        xlabel="FoodCourt",
        ylabel="Densidade",
    )
    axes[1].set(
        title=r"Depois: imputação, $\log(1+x)$ e escala $[-1,1]$",
        xlabel="FoodCourt transformado",
        ylabel="Densidade",
    )
    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
        ax.legend(frameon=True)

    fig.suptitle("Figura 6 — Efeito do pré-processamento em FoodCourt", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(
        FIGURES_DIR / "figure_6_foodcourt_before_after.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


def stratified_split_with_rng(
    x: pd.DataFrame,
    y: pd.Series,
    rng: np.random.Generator,
    test_fraction: float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Separa treino e teste por classe usando a mesma instância de ``rng``."""
    labels, counts = np.unique(y.to_numpy(), return_counts=True)
    exact_test_counts = counts * test_fraction
    test_counts = np.floor(exact_test_counts).astype(int)
    remaining = round(len(y) * test_fraction) - int(test_counts.sum())

    if remaining > 0:
        fractional_parts = exact_test_counts - test_counts
        for position in np.argsort(fractional_parts)[::-1][:remaining]:
            test_counts[position] += 1

    train_indices = []
    test_indices = []
    y_values = y.to_numpy()
    for label, class_test_count in zip(labels, test_counts):
        class_indices = np.flatnonzero(y_values == label)
        shuffled = rng.permutation(class_indices)
        test_indices.extend(shuffled[:class_test_count])
        train_indices.extend(shuffled[class_test_count:])

    train_indices = rng.permutation(np.asarray(train_indices))
    test_indices = rng.permutation(np.asarray(test_indices))
    return (
        x.iloc[train_indices].copy(),
        x.iloc[test_indices].copy(),
        y.iloc[train_indices].copy(),
        y.iloc[test_indices].copy(),
    )


def run_exercise_1(rng: np.random.Generator) -> dict:
    """Executa todas as etapas do Exercício 1."""
    original_points, original_labels = generate_clouds(rng, scale=1.0)
    create_figure_1(original_points, original_labels)

    datasets = {scale: generate_clouds(rng, scale=scale) for scale in SCALES}
    create_figure_2(datasets)

    ratios = separation_ratios()
    ratios.to_csv(OUTPUTS_DIR / "separation_ratios.csv", index=False)
    smallest_row = ratios.loc[ratios["r_ij_s_1"].idxmin()]

    overall_rates = {}
    class_rates = {}
    for scale, (points, labels) in datasets.items():
        overall_rates[scale], class_rates[scale] = mixing_rate(points, labels)
    create_figure_3(overall_rates, class_rates)

    return {
        "mixing_rates": {str(scale): rate for scale, rate in overall_rates.items()},
        "mixing_rates_by_class": {
            str(scale): {str(key): value for key, value in rates.items()}
            for scale, rates in class_rates.items()
        },
        "separation_ratios": ratios.to_dict(orient="records"),
        "smallest_ratio_s_1": float(smallest_row["r_ij_s_1"]),
        "smallest_ratio_pair": str(smallest_row["pair"]),
        "smallest_ratio_s_2": float(smallest_row["r_ij_s_1"] / 2),
    }


def run_exercise_2(rng: np.random.Generator) -> dict:
    """Executa todas as etapas do Exercício 2."""
    dataset_i = generate_shifted_gaussians(rng)
    dataset_ii = generate_concentric_shells(rng)
    explained_i, explained_ii = create_figure_4(dataset_i, dataset_ii)
    create_figure_5(dataset_i, dataset_ii)

    points_ii, labels_ii = dataset_ii
    squared_radii_ii = np.sum(points_ii**2, axis=1)
    maximum_core_squared_radius = float(squared_radii_ii[labels_ii == 0].max())
    minimum_shell_squared_radius = float(squared_radii_ii[labels_ii == 1].min())
    squared_radius_threshold = (
        maximum_core_squared_radius + minimum_shell_squared_radius
    ) / 2
    radial_predictions = (squared_radii_ii > squared_radius_threshold).astype(int)

    return {
        "distance_between_centers_dataset_i": class_center_distance(*dataset_i),
        "distance_between_centers_dataset_ii": class_center_distance(*dataset_ii),
        "explained_variance_pc1_pc2_dataset_i": explained_i,
        "explained_variance_pc1_pc2_dataset_ii": explained_ii,
        "maximum_core_squared_radius": maximum_core_squared_radius,
        "minimum_shell_squared_radius": minimum_shell_squared_radius,
        "squared_radius_threshold": squared_radius_threshold,
        "radial_rule_errors": int(np.count_nonzero(radial_predictions != labels_ii)),
        "minimum_eigenvalue_sigma_a": float(np.linalg.eigvalsh(SIGMA_A).min()),
        "minimum_eigenvalue_sigma_b": float(np.linalg.eigvalsh(SIGMA_B).min()),
    }


def run_exercise_3(rng: np.random.Generator) -> dict:
    """Executa descrição, split, pré-processamento e verificações do Exercício 3."""
    data = validate_spaceship_file(DATA_PATH)
    missing = pd.DataFrame(
        {
            "missing_count": data.isna().sum(),
            "missing_percentage": data.isna().mean() * 100,
        }
    )
    missing.to_csv(OUTPUTS_DIR / "missing_values.csv", index_label="column")

    spending_statistics = data[SPENDING_COLUMNS].agg(["mean", "median", "max"]).T
    spending_statistics.to_csv(
        OUTPUTS_DIR / "spending_statistics.csv", index_label="column"
    )

    x = data.drop(columns="Transported")
    y = data["Transported"].astype(bool)
    x_train_raw, x_test_raw, y_train, y_test = stratified_split_with_rng(
        x, y, rng, test_fraction=0.20
    )

    training_foodcourt = {
        "mean": float(x_train_raw["FoodCourt"].mean()),
        "median": float(x_train_raw["FoodCourt"].median()),
    }

    x_train, x_test, preprocessing_details = preprocess_spaceship(
        x_train_raw, x_test_raw
    )
    create_figure_6(x_train_raw, x_train, y_train)

    feature_names = pd.DataFrame({"feature": x_train.columns})
    feature_names.to_csv(OUTPUTS_DIR / "final_feature_names.csv", index=False)

    class_counts = y.value_counts().sort_index()
    class_shares = y.value_counts(normalize=True).sort_index()
    no_nan = not (x_train.isna().any().any() or x_test.isna().any().any())

    return {
        "class_counts": {str(label): int(value) for label, value in class_counts.items()},
        "class_shares": {
            str(label): float(value) for label, value in class_shares.items()
        },
        "positive_class_share": float(y.mean()),
        "missing_values": missing.reset_index(names="column").to_dict(orient="records"),
        "spending_statistics": spending_statistics.reset_index(names="column").to_dict(
            orient="records"
        ),
        "training_foodcourt_before_transform": training_foodcourt,
        "train_shape": list(x_train.shape),
        "test_shape": list(x_test.shape),
        "train_min": float(x_train.to_numpy().min()),
        "train_max": float(x_train.to_numpy().max()),
        "test_min": float(x_test.to_numpy().min()),
        "test_max": float(x_test.to_numpy().max()),
        "no_remaining_nan": no_nan,
        "train_positive_share": float(y_train.mean()),
        "test_positive_share": float(y_test.mean()),
        "preprocessing": preprocessing_details,
    }


def main() -> None:
    """Executa toda a atividade com resultados determinísticos."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")

    rng = np.random.default_rng(SEED)
    results = {
        "seed": SEED,
        "exercise_1": run_exercise_1(rng),
        "exercise_2": run_exercise_2(rng),
        "exercise_3": run_exercise_3(rng),
    }

    output_path = OUTPUTS_DIR / "results.json"
    output_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nResultados salvos em: {output_path}")


if __name__ == "__main__":
    main()
