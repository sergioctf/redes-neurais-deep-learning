"""Verificações pequenas das fórmulas e dimensões; não acessa MNIST."""

import math
import unittest

import torch

from vae_exercise import VAE, loss_terms


class VaeTests(unittest.TestCase):
    def test_shapes_and_reparameterization(self):
        torch.manual_seed(42)
        model = VAE(16)
        x = torch.zeros(3, 784)
        recon, mu, logvar = model(x)
        self.assertEqual(recon.shape, (3, 784))
        self.assertEqual(mu.shape, (3, 16))
        self.assertEqual(logvar.shape, (3, 16))
        self.assertTrue(bool(((recon >= 0) & (recon <= 1)).all()))
        torch.manual_seed(1)
        first = model.reparameterize(mu, logvar)
        torch.manual_seed(1)
        second = model.reparameterize(mu, logvar)
        self.assertTrue(torch.equal(first, second))

    def test_loss_is_sum_per_image_and_prior_kl_is_zero(self):
        x = torch.zeros(2, 784)
        recon = torch.full_like(x, 0.5)
        mu = torch.zeros(2, 16)
        logvar = torch.zeros(2, 16)
        bce, kl = loss_terms(recon, x, mu, logvar)
        self.assertAlmostEqual(bce.item(), 784 * math.log(2), places=3)
        self.assertEqual(kl.item(), 0.0)


if __name__ == "__main__":
    unittest.main()
