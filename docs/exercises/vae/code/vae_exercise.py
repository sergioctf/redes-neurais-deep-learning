"""Exercício 4: dois VAEs MNIST treinados e avaliados de modo reproduzível.

Execução na raiz do repositório: python docs/exercises/vae/code/vae_exercise.py
O conjunto de teste só entra após as 20 épocas e a avaliação final de cada modelo.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


ROOT = Path(__file__).resolve().parents[4]
EXERCISE = ROOT / "docs" / "exercises" / "vae"
FIGURES = EXERCISE / "figures"
OUTPUTS = EXERCISE / "outputs"
LATENT_DIMS = (16, 2)
EPOCHS = 20
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
BETA = 1.0


class VAE(nn.Module):
    """MLP 784→512→256→(mu, logvar)→d→256→512→784."""

    def __init__(self, latent_dim: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(784, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU()
        )
        self.mu_head = nn.Linear(256, latent_dim)
        self.logvar_head = nn.Linear(256, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, 784), nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.mu_head(h), self.logvar_head(h)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # eps independe dos parâmetros; a derivada passa por mu e sigma.
        eps = torch.randn_like(mu)
        return mu + torch.exp(0.5 * logvar) * eps

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def loss_terms(
    recon: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """BCE e KL em nats/imagem: soma pixels/dimensões, média imagens."""
    batch_size = x.shape[0]
    reconstruction = F.binary_cross_entropy(recon, x, reduction="sum") / batch_size
    # KL[N(mu,diag(sigma²)) || N(0,I)] em forma fechada.
    kl = 0.5 * torch.sum(mu.square() + logvar.exp() - 1.0 - logvar) / batch_size
    return reconstruction, kl


def run_epoch(
    model: VAE, loader: DataLoader, optimizer: torch.optim.Optimizer | None
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    reconstruction_sum = 0.0
    kl_sum = 0.0
    count = 0
    for images, _ in loader:
        x = images.reshape(images.shape[0], 784)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            recon, mu, logvar = model(x)
            reconstruction, kl = loss_terms(recon, x, mu, logvar)
            if training:
                (reconstruction + BETA * kl).backward()
                optimizer.step()
        n = x.shape[0]
        reconstruction_sum += reconstruction.item() * n
        kl_sum += kl.item() * n
        count += n
    rec = reconstruction_sum / count
    kl = kl_sum / count
    return {"reconstruction": rec, "kl": kl, "negative_elbo": rec + BETA * kl}


def train_model(
    latent_dim: int, train_set, val_set
) -> tuple[VAE, list[dict[str, float]]]:
    # Exigência do enunciado: reiniciar a semente antes de CADA construção.
    torch.manual_seed(42)
    model = VAE(latent_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # Mesma ordem de lotes nos dois modelos, independente da RNG de z.
    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True,
        generator=torch.Generator().manual_seed(42), num_workers=0,
    )
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    history = []
    for epoch in range(1, EPOCHS + 1):
        train_metrics = run_epoch(model, train_loader, optimizer)
        with torch.no_grad():
            val_metrics = run_epoch(model, val_loader, None)
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()},
               **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        print(
            f"d={latent_dim:2d} epoch={epoch:2d} "
            f"train BCE={train_metrics['reconstruction']:.3f} KL={train_metrics['kl']:.3f} "
            f"val BCE={val_metrics['reconstruction']:.3f} KL={val_metrics['kl']:.3f}",
            flush=True,
        )
    return model, history


def evaluate_test(model: VAE, test_set) -> dict[str, float]:
    # Uma única avaliação numérica final; z é sorteado de q(z|x), não trocado por mu.
    torch.manual_seed(42)
    loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    with torch.no_grad():
        return run_epoch(model, loader, None)


def save_history(history: list[dict[str, float]], latent_dim: int) -> None:
    with (OUTPUTS / f"history_d{latent_dim}.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)


def figure_1(history: list[dict[str, float]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, key, title in zip(
        axes, ("reconstruction", "kl"), ("Reconstrução (BCE)", "Regularização (KL)")
    ):
        epochs = [row["epoch"] for row in history]
        ax.plot(epochs, [row[f"train_{key}"] for row in history], label="Treino", color="#2563a6")
        ax.plot(epochs, [row[f"val_{key}"] for row in history], label="Validação", color="#d97937")
        ax.set(title=title, xlabel="Época", ylabel="Nats por imagem")
        ax.legend()
        ax.grid(alpha=0.2)
    fig.suptitle("Figura 1 — VAE d=16: os dois termos da perda")
    fig.tight_layout()
    fig.savefig(FIGURES / "figure_1_loss_terms.png", dpi=170)
    plt.close(fig)


def first_test_digits(test_set) -> tuple[list[int], torch.Tensor]:
    indices = []
    seen = set()
    for idx in range(len(test_set)):
        _, label = test_set[idx]
        if label not in seen:
            indices.append(idx)
            seen.add(label)
        if len(seen) == 10:
            break
    indices.sort(key=lambda i: test_set[i][1])
    x = torch.stack([test_set[i][0].reshape(784) for i in indices])
    return indices, x


def figure_2(model: VAE, test_set) -> list[int]:
    indices, x = first_test_digits(test_set)
    with torch.no_grad():
        mu, _ = model.encode(x)
        recon = model.decode(mu)
    fig, axes = plt.subplots(2, 10, figsize=(13, 3.4))
    for digit in range(10):
        axes[0, digit].imshow(x[digit].reshape(28, 28), cmap="gray", vmin=0, vmax=1)
        axes[1, digit].imshow(recon[digit].reshape(28, 28), cmap="gray", vmin=0, vmax=1)
        axes[0, digit].set_title(f"{digit} · #{indices[digit]}", fontsize=9)
        for row in range(2):
            axes[row, digit].set_xticks([])
            axes[row, digit].set_yticks([])
    fig.suptitle("Figura 2 — Primeiro exemplo de teste por dígito e reconstrução de μ")
    fig.supxlabel("Classe do dígito (índice oficial de teste)")
    fig.supylabel("Original (acima) / reconstrução (abaixo)")
    fig.tight_layout()
    fig.savefig(FIGURES / "figure_2_reconstructions.png", dpi=170)
    plt.close(fig)
    return indices


def figure_3(model: VAE) -> None:
    # Semente exigida especificamente para as 64 amostras do prior.
    torch.manual_seed(0)
    with torch.no_grad():
        z = torch.randn(64, model.latent_dim)
        samples = model.decode(z).reshape(8, 8, 28, 28)
    fig, axes = plt.subplots(8, 8, figsize=(8.5, 8.5))
    for row in range(8):
        for col in range(8):
            axes[row, col].imshow(samples[row, col], cmap="gray", vmin=0, vmax=1)
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
    fig.suptitle("Figura 3 — 64 amostras do prior N(0, I), VAE d=16")
    fig.supxlabel("Coluna da amostra")
    fig.supylabel("Linha da amostra")
    fig.tight_layout()
    fig.savefig(FIGURES / "figure_3_prior_samples.png", dpi=170)
    plt.close(fig)


def figure_4(model: VAE, test_set) -> None:
    # Depois da avaliação final: usar o teste só para visualizar mu e sua classe.
    loader = DataLoader(test_set, batch_size=512, shuffle=False)
    mus, labels = [], []
    with torch.no_grad():
        for images, y in loader:
            mu, _ = model.encode(images.reshape(images.shape[0], 784))
            mus.append(mu.numpy())
            labels.append(y.numpy())
    mu_all = np.concatenate(mus)
    y_all = np.concatenate(labels)
    assert mu_all.shape == (10_000, 2)
    fig, ax = plt.subplots(figsize=(8.5, 7))
    colors = plt.cm.tab10.colors
    for digit in range(10):
        mask = y_all == digit
        ax.scatter(mu_all[mask, 0], mu_all[mask, 1], s=4, alpha=0.45,
                   color=colors[digit], label=str(digit), rasterized=True)
    ax.set(title="Figura 4 — Médias latentes das 10.000 imagens de teste (d=2)",
           xlabel="μ₁", ylabel="μ₂")
    ax.legend(title="Dígito", ncol=2, markerscale=3, fontsize=8)
    ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(FIGURES / "figure_4_test_latents.png", dpi=170)
    plt.close(fig)


def figure_5(model: VAE) -> None:
    q = np.linspace(0.05, 0.95, 15)
    z_axis = norm.ppf(q)
    # Eixo z2 decresce na imagem, para corresponder ao plano cartesiano.
    grid = np.array([[x, y] for y in z_axis[::-1] for x in z_axis], dtype=np.float32)
    with torch.no_grad():
        decoded = model.decode(torch.from_numpy(grid)).reshape(15, 15, 28, 28).numpy()
    canvas = np.concatenate([np.concatenate(decoded[row], axis=1) for row in range(15)], axis=0)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(canvas, cmap="gray", vmin=0, vmax=1, extent=[-0.5, 14.5, 14.5, -0.5])
    ticks = [0, 3, 7, 11, 14]
    ax.set_xticks(ticks, [f"{z_axis[i]:.2f}" for i in ticks])
    ax.set_yticks(ticks, [f"{z_axis[14-i]:.2f}" for i in ticks])
    ax.set(title="Figura 5 — Plano latente decodificado em quantis normais (d=2)",
           xlabel="z₁ = Φ⁻¹(q)", ylabel="z₂ = Φ⁻¹(q)")
    fig.tight_layout()
    fig.savefig(FIGURES / "figure_5_decoded_plane.png", dpi=170)
    plt.close(fig)


def figure_6(model: VAE, test_set, indices: tuple[int, int]) -> tuple[int, int]:
    a, b = indices
    xa, ya = test_set[a]
    xb, yb = test_set[b]
    assert ya != yb
    x = torch.stack([xa.reshape(784), xb.reshape(784)])
    ts = torch.linspace(0, 1, 10).unsqueeze(1)
    pixels = (1 - ts) * x[0] + ts * x[1]
    with torch.no_grad():
        mu, _ = model.encode(x)
        latent = model.decode((1 - ts) * mu[0] + ts * mu[1])
    fig, axes = plt.subplots(2, 10, figsize=(13, 3.4))
    for col, t in enumerate(ts.flatten().tolist()):
        axes[0, col].imshow(pixels[col].reshape(28, 28), cmap="gray", vmin=0, vmax=1)
        axes[1, col].imshow(latent[col].reshape(28, 28), cmap="gray", vmin=0, vmax=1)
        axes[0, col].set_title(f"t={t:.2f}", fontsize=8)
        for row in range(2):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
    fig.suptitle(f"Figura 6 — Interpolação: teste #{a} ({ya}) → #{b} ({yb})")
    fig.supxlabel("t de 0 a 1")
    fig.supylabel("Pixels (acima) / latente d=16 (abaixo)")
    fig.tight_layout()
    fig.savefig(FIGURES / "figure_6_interpolation.png", dpi=170)
    plt.close(fig)
    return ya, yb


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    # Evita excesso de threads pequenas e mantém execução local consistente.
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    transform = transforms.ToTensor()
    train_full = datasets.MNIST("data", train=True, download=True, transform=transform)
    test = datasets.MNIST("data", train=False, download=True, transform=transform)
    train, val = random_split(
        train_full, [50_000, 10_000], generator=torch.Generator().manual_seed(42)
    )
    results = {"configuration": {"epochs": EPOCHS, "batch_size": BATCH_SIZE,
              "learning_rate": LEARNING_RATE, "beta": BETA, "split_seed": 42,
              "model_seed": 42, "test_seed": 42, "prior_sample_seed": 0,
              "train_size": len(train), "val_size": len(val), "test_size": len(test),
              "torch_version": torch.__version__,
              "torchvision_version": __import__("torchvision").__version__}}

    model16, history16 = train_model(16, train, val)
    save_history(history16, 16)
    results["d16"] = {"test": evaluate_test(model16, test),
                      "val_kl_first": history16[0]["val_kl"],
                      "val_kl_last": history16[-1]["val_kl"]}
    print("Final test d=16:", results["d16"]["test"], flush=True)
    figure_1(history16)

    model2, history2 = train_model(2, train, val)
    save_history(history2, 2)
    results["d2"] = {"test": evaluate_test(model2, test),
                     "val_kl_first": history2[0]["val_kl"],
                     "val_kl_last": history2[-1]["val_kl"]}
    print("Final test d=2:", results["d2"]["test"], flush=True)

    # Todas as visualizações do teste vêm depois dos dois treinos e números finais.
    results["d16"]["first_test_indices_by_digit"] = figure_2(model16, test)
    figure_3(model16)
    figure_4(model2, test)
    figure_5(model2)
    results["interpolation_indices"] = [0, 1]
    results["interpolation_labels"] = figure_6(model16, test, (0, 1))
    with (OUTPUTS / "results.json").open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
        fh.write("\n")


if __name__ == "__main__":
    main()
