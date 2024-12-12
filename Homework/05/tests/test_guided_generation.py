from unittest import TestCase
import scripts.guided_generation as guided_generation
import warnings
warnings.filterwarnings("ignore")

import torch
import numpy as np
from transformers.tokenization_utils_base import BatchEncoding


class MockModel:
    def generate(self, input_ids, attention_mask=None, max_new_tokens=50, do_sample=True, *args, **kwargs):
        num_samples = input_ids.shape[0]
        return [torch.tensor([np.random.uniform(-1, 1)]) for _ in range(num_samples)]

class MockTokenizer:
    def __call__(self, texts, return_tensors=None):
        # Simulate tokenization by returning a BatchEncoding with 'input_ids' and 'attention_mask'
        num_samples = len(texts)
        seq_length = 2
        input_ids = torch.zeros((num_samples, seq_length), dtype=torch.long)
        attention_mask = torch.ones((num_samples, seq_length), dtype=torch.long)  # Typically ones
        return BatchEncoding({'input_ids': input_ids, 'attention_mask': attention_mask})

    def decode(self, tokens):
        return str(tokens[0])


class TestGuidedGeneration(TestCase):

    def setUp(self):
        self.mock_model = MockModel()
        self.mock_tokenizer = MockTokenizer()

    def test_random_reward(self):
        def compute_reward(reward_model, reward_tokenizer, texts, device='cpu'):
            def inner_reward(text):
                return float(text)

            return torch.tensor([inner_reward(text) for text in texts])

        # Use the function defined earlier
        guided_generation.compute_reward = compute_reward
        best_sample = guided_generation.generate_with_reward_guidance(
            self.mock_model, self.mock_tokenizer,
            None, None,
            N = 1000,
        )
        
        self.assertGreater(float(best_sample), 0.98)