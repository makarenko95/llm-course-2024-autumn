import torch
from unittest import TestCase
from scripts.model import Model


class TestModel(TestCase):
    def test_model(self):
        model = Model(
            vocab_size=3,
            emb_size=8,
            hidden_size=16,
            num_layers=2,
            dropout=0.05
        )
        inputs = torch.tensor([[0, 0, 0, 0, 0], [0, 1, 2, 1, 1], [0, 2, 1, 2, 2], [1, 0, 2, 0, 0]])
        outputs, (h_n, c_n) = model(inputs)
        self.assertEqual(outputs.shape, (4, 5, 3))
        self.assertEqual(h_n.shape, (2, 4, 16))
        self.assertEqual(c_n.shape, (2, 4, 16))

        hx = None
        for _ in range(10):
            outputs, hx = model(inputs, hx=hx)
        h_n, c_n = hx

        self.assertEqual(h_n.shape, (2, 4, 16))
        self.assertEqual(c_n.shape, (2, 4, 16))
