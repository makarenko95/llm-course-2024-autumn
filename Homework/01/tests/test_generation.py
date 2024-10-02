from unittest import TestCase
from scripts.model import Model
from scripts.tokenizer import ByteTokenizer
from scripts.generation import generate


class TestGeneration(TestCase):
    def test_generate(self):
        tokenizer = ByteTokenizer()
        model = Model(tokenizer.get_vocab_size(), emb_size=8, hidden_size=32)

        greedy_gens = [generate(model, tokenizer, temperature=0, max_length=32) for _ in range(10)]
        self.assertEqual(len(set(greedy_gens)), 1)

        random_gens = [generate(model, tokenizer, temperature=50, max_length=32) for _ in range(10)]
        self.assertTrue(len(set(random_gens)) > 1)
