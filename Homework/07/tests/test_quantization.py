import torch
from scripts.quantization import (
    absmax_quantization,
    absmax_dequantization,
    zeropoint_quantization,
    zeropoint_dequantization
)
from unittest import TestCase


class TestQuantization(TestCase):
    def test_absmax_quantization(self):
        x = torch.tensor([[1.5, 0, -1.1], [-2, 0.5, -0.1], [1.2, 1.7, -0.5]])
        s, x_q = absmax_quantization(x)
        self.assertEqual(63.5, round(s, 1))
        self.assertTrue(torch.equal(
            torch.tensor([[95, 0, -70], [-127, 32, -6], [76, 108, -32]], dtype=torch.int8),
            x_q
        ))

    def test_absmax_dequantization(self):
        s = 63.5
        x_q = torch.tensor([[95,   0, -70], [127,  32,  -6], [76, 108, -32]], dtype=torch.int8)
        x = absmax_dequantization(s, x_q)
        self.assertTrue(torch.equal(
            torch.tensor([[1.496, 0, -1.102], [2, 0.504, -0.094], [1.197, 1.701, -0.504]]),
            torch.round(x, decimals=3)
        ))

    def test_zeropoint_quantization(self):
        x = torch.tensor([[1.5, 0, -1.1], [2, 0.5, -0.1], [1.2, 1.7, -0.5]])
        s, z, x_q = zeropoint_quantization(x)
        self.assertEqual(82.258, round(s, 3))
        self.assertEqual(-38, z)
        self.assertTrue(torch.equal(
            torch.tensor([[85, -38, -128], [127, 3, -46], [61, 102, -79]], dtype=torch.int8),
            x_q
        ))

    def test_zeropoint_dequantization(self):
        s = 82.25806451612902
        z = -38
        x_q = torch.tensor([[85, -38, -128], [127, 3, -46], [61, 102, -79]], dtype=torch.int8)
        x = zeropoint_dequantization(s, z, x_q)
        self.assertTrue(torch.equal(
            torch.tensor([[1.495, 0, -1.094], [2.006, 0.498, -0.097], [1.204, 1.702, -0.498]]),
            torch.round(x, decimals=3)
        ))
