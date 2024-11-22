from unittest import TestCase
# from scripts.eval_reward_model import eval_reward_model
import scripts.eval_reward_model as eval_reward_model
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from datasets import load_dataset
from torch import tensor
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data_list):
        """
        Args:
            data_list (list of dicts): List where each dict contains features and labels.
        """
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]
        # data_point = self.data_list[idx]
        # return data_point['text'], data_point['label']


class TestEvalRewardModel(TestCase):

    def test_custom_dataset(self):
        
        def compute_reward(reward_model, reward_tokenizer, texts):
            def inner_reward(text):
                return int(text)

            return tensor([inner_reward(text) for text in texts])

        # from scripts.eval_reward_model import eval_reward_model

        direct_test = [
            {'text': '1', 'label': 1},
            {'text': '2', 'label': 1},
            {'text': '3', 'label': 1},
            {'text': '4', 'label': 1},
            {'text': '5', 'label': 1},
            {'text': '-1', 'label': 0},
            {'text': '-2', 'label': 0},
            {'text': '-3', 'label': 0},
            {'text': '-4', 'label': 0},
            {'text': '6', 'label': 0},
        ]

        eval_reward_model.compute_reward = compute_reward
        test_accuracy = eval_reward_model.eval_reward_model(
            None,
            None,
            CustomDataset(direct_test),
            target_label=1,
        )

        self.assertEqual(test_accuracy, 0.8)


    def test_imdb_heuristics(self):
        def compute_reward(reward_model, reward_tokenizer, texts):
            def simple_reward(text):

                positive_keywords = [
                    "amazing",
                    "awesome",
                    "great",
                    "fantastic",
                    "enjoyable",
                    "fun",
                    "exciting",
                    "wonderful",
                    "excellent",
                    "entertaining"
                ]

                negative_keywords = [
                    "bad",
                    "boring",
                    "dull",
                    "terrible",
                    "poor",
                    "uninteresting",
                    "slow",
                    "annoying",
                    "forgettable",
                    "mediocre"
                ]

                reward = sum([text.count(word) for word in negative_keywords]) - sum([text.count(word) for word in positive_keywords])
                if any(word in text for word in negative_keywords):
                    # return 1 + np.random.rand()
                    return reward + np.random.rand()
                elif any(word in text for word in positive_keywords):
                    # return -1 - np.random.rand()
                    # return -sum([text.count(word) for word in positive_keywords]) + np.random.rand()
                    return reward - np.random.rand()
                else:
                    return np.random.rand()

            return tensor([simple_reward(text) for text in texts])



        imdb_test = load_dataset("imdb", split='test')

        eval_reward_model.compute_reward = compute_reward
        test_accuracy = eval_reward_model.eval_reward_model(
            None,
            None,
            imdb_test,
            target_label=0,
        )

        self.assertGreater(test_accuracy, 0.75)