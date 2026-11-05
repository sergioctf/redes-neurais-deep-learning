"""MLP de classificação implementado apenas com NumPy.

As matrizes usam a convenção de lotes: A[l] tem uma linha por amostra e
W[l] tem formato (unidades_da_camada_anterior, unidades_da_camada_seguinte).
O mesmo forward, backward e update serve para saída binária ou multiclasse.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


class MLP:
    """Camadas ocultas tanh; saída sigmoid/BCE ou softmax/CCE."""

    def __init__(
        self,
        layer_sizes: Sequence[int],
        output_mode: str,
        rng: np.random.Generator,
    ) -> None:
        if len(layer_sizes) < 3 or any(size < 1 for size in layer_sizes):
            raise ValueError("É necessária entrada, camada oculta e saída.")
        if output_mode not in {"binary", "multiclass"}:
            raise ValueError("output_mode deve ser binary ou multiclass.")
        if output_mode == "binary" and layer_sizes[-1] != 1:
            raise ValueError("A saída binária deve ter uma unidade.")
        if output_mode == "multiclass" and layer_sizes[-1] < 2:
            raise ValueError("A saída multiclasse precisa de ao menos duas unidades.")

        self.layer_sizes = tuple(int(size) for size in layer_sizes)
        self.output_mode = output_mode
        # Xavier mantém as ativações tanh em escala razoável no início.
        self.weights = [
            rng.normal(0.0, np.sqrt(2.0 / (n_in + n_out)), (n_in, n_out))
            for n_in, n_out in zip(self.layer_sizes[:-1], self.layer_sizes[1:])
        ]
        self.biases = [
            np.zeros((1, n_out), dtype=float) for n_out in self.layer_sizes[1:]
        ]

    @staticmethod
    def _sigmoid(logits: np.ndarray) -> np.ndarray:
        # O limite evita overflow em exp sem alterar resultados usuais.
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -500.0, 500.0)))

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_shifted = np.exp(shifted)
        return exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)

    def forward(self, x: np.ndarray) -> list[np.ndarray]:
        """Retorna A[0]=X, ativações ocultas e probabilidades de saída."""
        activations = [np.asarray(x, dtype=float)]
        for weights, bias in zip(self.weights[:-1], self.biases[:-1]):
            activations.append(np.tanh(activations[-1] @ weights + bias))
        logits = activations[-1] @ self.weights[-1] + self.biases[-1]
        if self.output_mode == "binary":
            activations.append(self._sigmoid(logits))
        else:
            activations.append(self._softmax(logits))
        return activations

    def loss(self, probabilities: np.ndarray, y: np.ndarray) -> float:
        """Entropia cruzada média, calculada sem biblioteca de modelos."""
        eps = 1e-12
        probabilities = np.clip(probabilities, eps, 1.0 - eps)
        if self.output_mode == "binary":
            target = np.asarray(y).reshape(-1, 1)
            return float(-np.mean(
                target * np.log(probabilities)
                + (1 - target) * np.log(1 - probabilities)
            ))
        rows = np.arange(len(y))
        return float(-np.mean(np.log(probabilities[rows, y.astype(int)])))

    def backward(
        self, activations: list[np.ndarray], y: np.ndarray
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Regra da cadeia para qualquer quantidade de camadas ocultas.

        Sigmoid+BCE e softmax+CCE produzem o mesmo delta terminal (P-Y)/N.
        Para cada camada: dW=A_anterior.T@delta; db=soma(delta).
        Uma camada tanh anterior recebe (delta@W.T)*(1-A_anterior**2).
        """
        n_samples = len(y)
        probabilities = activations[-1]
        if self.output_mode == "binary":
            target = np.asarray(y).reshape(-1, 1)
        else:
            target = np.eye(self.layer_sizes[-1])[y.astype(int)]
        delta = (probabilities - target) / n_samples
        grad_weights = [np.empty_like(w) for w in self.weights]
        grad_biases = [np.empty_like(b) for b in self.biases]

        for layer in range(len(self.weights) - 1, -1, -1):
            grad_weights[layer] = activations[layer].T @ delta
            grad_biases[layer] = np.sum(delta, axis=0, keepdims=True)
            if layer > 0:
                delta = (delta @ self.weights[layer].T) * (
                    1.0 - activations[layer] ** 2
                )
        return grad_weights, grad_biases

    def update(
        self,
        gradients: tuple[list[np.ndarray], list[np.ndarray]],
        learning_rate: float,
    ) -> None:
        """Um passo de gradiente descendente, idêntico em todos os exercícios."""
        grad_weights, grad_biases = gradients
        for layer in range(len(self.weights)):
            self.weights[layer] -= learning_rate * grad_weights[layer]
            self.biases[layer] -= learning_rate * grad_biases[layer]

    def fit(
        self, x: np.ndarray, y: np.ndarray, epochs: int, learning_rate: float
    ) -> list[float]:
        """Treino full-batch determinístico; perda medida depois de cada passo."""
        history: list[float] = []
        for _ in range(epochs):
            activations = self.forward(x)
            gradients = self.backward(activations, y)
            self.update(gradients, learning_rate)
            history.append(self.loss(self.forward(x)[-1], y))
        return history

    def predict(self, x: np.ndarray) -> np.ndarray:
        probabilities = self.forward(x)[-1]
        if self.output_mode == "binary":
            return (probabilities[:, 0] >= 0.5).astype(int)
        return np.argmax(probabilities, axis=1)

    def accuracy(self, x: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(x) == y))

    @property
    def parameter_count(self) -> int:
        return int(sum(w.size + b.size for w, b in zip(self.weights, self.biases)))
