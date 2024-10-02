import torch
from unittest import TestCase
from scripts.collator import Collator


class TestCollator(TestCase):
    def test_collator(self):
        data = [[1, 2], [3, 4, 5], [1]]
        collator = Collator(0)

        expected = torch.tensor([[1, 2, 0], [3, 4, 5], [1, 0, 0]], dtype=torch.long)
        actual = collator(data)
        self.assertEqual(expected.shape, actual.shape)
        self.assertTrue(torch.all(expected == actual))
