from unittest import TestCase
from scripts.tokenizer import count_pairs, merge, BpeTokenizer


class TestTokenizer(TestCase):
    def test_count_pairs(self):
        data = [[0, 0, 1, 2, 2], [2, 2, 3, 4, 0, 10]]
        self.assertEqual(
            count_pairs(data),
            {
                (0, 0): 1,
                (0, 1): 1,
                (1, 2): 1,
                (2, 2): 2,
                (2, 3): 1,
                (3, 4): 1,
                (4, 0): 1,
                (0, 10): 1
            }
        )

    def test_merge(self):
        self.assertEqual(
            merge([0, 1, 1, 0, 0, 1, 0, 1, 2], (0, 1), 3),
            [3, 1, 0, 3, 3, 2]
        )
        self.assertEqual(
            merge([0, 0, 0, 1, 0, 0, 1, 0], (0, 0), 2),
            [2, 0, 1, 2, 1, 0]
        )

    def test_encode(self):
        data = ['aaaaa', 'abababc']
        tokenizer = BpeTokenizer()

        tokenizer.train(data, max_vocab=259)
        self.assertEqual(
            tokenizer.encode('aaaaababcd'),
            [97, 97, 97, 97, 97, 98, 97, 98, 99, 100]
        )

        tokenizer.train(data, max_vocab=260)
        self.assertEqual(
            tokenizer.encode('aaaaababcd'),
            [259, 259, 97, 98, 97, 98, 99, 100]
        )

        tokenizer.train(data, max_vocab=261)
        self.assertEqual(
            tokenizer.encode('aaaaababcd'),
            [259, 259, 260, 260, 99, 100]
        )

        tokenizer.train(data, max_vocab=262)
        self.assertEqual(
            tokenizer.encode('aaaaababcd'),
            [259, 259, 261, 99, 100]
        )

        tokenizer.train(data, max_vocab=512)
        self.assertEqual(
            tokenizer.encode('aaaaababcd'),
            [259, 259, 261, 99, 100]
        )
