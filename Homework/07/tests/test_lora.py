import torch
from unittest import TestCase
from torch.nn import Linear
from scripts.lora import LoraLayer, merge


class TestLora(TestCase):
    def test_lora_layer_init(self):
        layer = LoraLayer(4, 3, 2)
        self.assertTrue(torch.equal(
            torch.zeros(3, 2),
            layer.B.weight.data
        ))
        self.assertFalse(torch.equal(
            torch.zeros(2, 4),
            layer.A.weight.data
        ))

    def test_lora_layer_inference(self):
        layer = LoraLayer(4, 3, 2)
        a_weights = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=torch.float)
        b_weights = torch.tensor([[2, -1], [0, 1], [0, 0]], dtype=torch.float)

        self.assertTrue(torch.equal(
            torch.tensor([0, 0, 0], dtype=torch.float),
            layer(torch.tensor([2, -1, 3, 7], dtype=torch.float))
        ))

        layer.load(a_weights, b_weights)

        self.assertTrue(torch.equal(
            torch.tensor([5, -1, 0], dtype=torch.float),
            layer(torch.tensor([2, -1, 3, 7], dtype=torch.float)).round()
        ))
        self.assertTrue(torch.equal(
            torch.tensor([5, -1, 0], dtype=torch.float),
            layer(torch.tensor([2, -1, -0.5, 0.7], dtype=torch.float)).round()
        ))
        self.assertTrue(torch.equal(
            torch.tensor([1, 1, 0], dtype=torch.float),
            layer(torch.tensor([1, 1, 30, 2], dtype=torch.float)).round()
        ))

    def test_merge(self):
        linear_layer = Linear(3, 5, bias=False)
        lora_layer = LoraLayer(3, 5, 2)
        lora_layer.load(torch.rand(lora_layer.A.weight.shape), torch.rand(lora_layer.B.weight.shape))
        x = torch.rand((10, 3))
        expected = torch.round(linear_layer(x) + lora_layer(x), decimals=3)
        merge(linear_layer, lora_layer)
        actual = torch.round(linear_layer(x), decimals=3)
        self.assertTrue(torch.equal(expected, actual))
