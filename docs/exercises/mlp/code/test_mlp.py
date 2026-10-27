"""Verificações independentes dos gradientes analíticos do MLP.

Executar da raiz: python docs/exercises/mlp/code/test_mlp.py
Não usa bibliotecas de modelos nem diferenciação automática.
"""

import unittest

import numpy as np

from mlp_from_scratch import MLP


class GradientTests(unittest.TestCase):
    def check_gradients(self, architecture: tuple[int, ...], mode: str) -> None:
        rng = np.random.default_rng(42)
        model = MLP(architecture, mode, rng)
        x = rng.normal(size=(4, architecture[0]))
        y = (
            np.array([0, 1, 0, 1])
            if mode == "binary"
            else np.array([0, 1, 2, 1])
        )
        activations = model.forward(x)
        gradients = model.backward(activations, y)
        epsilon = 1e-6

        # Diferenças centrais: derivada estimada sem reutilizar o backward.
        for parameters, expected_gradients in zip(
            (model.weights, model.biases), gradients
        ):
            for parameter, expected in zip(parameters, expected_gradients):
                for index in np.ndindex(parameter.shape):
                    original = parameter[index]
                    parameter[index] = original + epsilon
                    loss_plus = model.loss(model.forward(x)[-1], y)
                    parameter[index] = original - epsilon
                    loss_minus = model.loss(model.forward(x)[-1], y)
                    parameter[index] = original
                    numerical = (loss_plus - loss_minus) / (2 * epsilon)
                    self.assertAlmostEqual(
                        numerical, expected[index], delta=1e-7,
                        msg=f"{mode}: gradiente em {index}",
                    )

        before = model.loss(model.forward(x)[-1], y)
        model.update(gradients, learning_rate=0.05)
        after = model.loss(model.forward(x)[-1], y)
        self.assertLess(after, before)

    def test_binary_one_hidden_layer(self) -> None:
        self.check_gradients((2, 3, 1), "binary")

    def test_multiclass_one_hidden_layer(self) -> None:
        self.check_gradients((2, 3, 3), "multiclass")

    def test_multiclass_two_hidden_layers(self) -> None:
        self.check_gradients((2, 3, 4, 3), "multiclass")


if __name__ == "__main__":
    unittest.main()
