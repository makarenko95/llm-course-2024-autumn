from unittest import TestCase
from scripts.compute_reward import compute_reward
import warnings
warnings.filterwarnings("ignore")

import torch
from datasets import load_dataset
from transformers.tokenization_utils_base import BatchEncoding
from transformers.modeling_outputs import SequenceClassifierOutput


class MockRewardModel:
    def __call__(self, input_ids, attention_mask=None, max_new_tokens=50, do_sample=True, *args, **kwargs):
        return SequenceClassifierOutput(
            logits=torch.tensor(
                [[-0.2009,  0.0244],
                 [-0.2187,  0.0313]]
                ))

class MockTokenizer:
    def __call__(self, texts, return_tensors=None, *args, **kwargs):
        # Simulate tokenization by returning a BatchEncoding with 'input_ids' and 'attention_mask'
        num_samples = len(texts)
        seq_length = 2
        input_ids = torch.zeros((num_samples, seq_length), dtype=torch.long)
        attention_mask = torch.ones((num_samples, seq_length), dtype=torch.long)  # Typically ones
        return BatchEncoding({'input_ids': input_ids, 'attention_mask': attention_mask})

    def decode(self, tokens):
        return None


class TestComputeReward(TestCase):

    def setUp(self):    
        self.reward_model = MockRewardModel()
        self.reward_tokenizer = MockTokenizer()
        self.imdb = load_dataset("imdb", split='train')

    def test_mock(self):
        rewards = compute_reward(
            self.reward_model,
            self.reward_tokenizer,
            [self.imdb[45]['text'], self.imdb[16000]['text']],
        )

        self.assertEqual(len(rewards), 2)
        self.assertEqual(rewards[0], -0.2009)
        self.assertEqual(rewards[1], -0.2187)