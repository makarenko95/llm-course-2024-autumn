from unittest import TestCase
from scripts.tokenizer import ByteTokenizer
from scripts.dataset import MyDataset


class TestDataset(TestCase):
    def test_dataset(self):
        data = ['aaaaa', 'abababc']
        tokenizer = ByteTokenizer()

        dataset = MyDataset(data, tokenizer)
        self.assertEqual(
            len(dataset),
            2
        )
        self.assertEqual(
            dataset[0],
            [257, 97, 97, 97, 97, 97, 258]
        )
        self.assertEqual(
            dataset[1],
            [257, 97, 98, 97, 98, 97, 98, 99, 258]
        )

        dataset = MyDataset(data, tokenizer, max_length=3)
        self.assertEqual(
            len(dataset),
            2
        )
        self.assertEqual(
            dataset[0],
            [257, 97, 97]
        )
        self.assertEqual(
            dataset[1],
            [257, 97, 98]
        )
