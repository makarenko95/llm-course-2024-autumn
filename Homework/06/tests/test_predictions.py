import sys
sys.path.append("Homework/06")
import unittest
from unittest.mock import MagicMock
import re
import torch

from get_predictions import predict_by_token_id, get_choice_log_probs


class TestGetChoiceLogProbs(unittest.TestCase):
    def test_example_case_1(self):
        # Define the toy inputs
        logits = torch.tensor([[
            [2.0, 1.0, 0.1, 0.5, 0.3],  # Logits for the first token (start token)
            [0.1, 2.0, 0.1, 0.5, 0.3],  # Logits for the second token
            [0.2, 0.1, 2.0, 0.5, 0.3]   # Logits for the third token
        ]])

        input_ids = torch.tensor([[0, 1, 2]])  # Start token, then token 1, then token 2

        # Expected result
        expected_mean_log_prob = -2.0437793731689453

        # Calculate the mean log probability using the function
        mean_log_prob = get_choice_log_probs(logits, input_ids)

        # Assert that the calculated mean log probability matches the expected result
        self.assertAlmostEqual(mean_log_prob, expected_mean_log_prob, places=1)

    def test_example_case_2(self):
        logits = torch.tensor([[
            [0.1, 0.5, 2.0],  # Logits for the first token (start token)
            [0.5, 2.0, 0.1],  # Logits for the second token
            [2.0, 0.1, 0.5],  # Logits for the third token
            [0.1, 0.5, 2.0]  # Logits for the fourth token
        ]])

        # Toy input_ids tensor
        input_ids = torch.tensor([[0, 1, 0, 2]])  # Start token, then token 1, then token 0, then token 2

        # Expected result
        expected_mean_log_prob = -1.8167786598205566

        # Calculate the mean log probability using the function
        mean_log_prob = get_choice_log_probs(logits, input_ids)

        # Assert that the calculated mean log probability matches the expected result
        self.assertAlmostEqual(mean_log_prob, expected_mean_log_prob, places=1)

    def test_predict_choice_1(self):
        # Mock tokenizer
        tokenizer = MagicMock()
        tokenizer.encode = MagicMock(
            side_effect=lambda x, add_special_tokens: {'A': [0], 'B': [1], 'C': [2], 'D': [3]}[x])

        # Define toy logits tensor
        logits = torch.tensor([[
            [0.1, 0.2, 0.3, 0.4],  # Logits for some token
            [0.4, 0.3, 0.2, 0.1],  # Logits for another token
            [0.1, 0.5, 0.2, 0.3]  # Logits for the last token (answer token)
        ]])

        # Expected result: The index of the predicted choice
        expected_choice = 1  # 'B' has the highest logit (0.5)

        # Calculate the predicted choice using the function
        predicted_choice = predict_by_token_id(logits, tokenizer)

        # Assert that the predicted choice matches the expected result
        self.assertEqual(predicted_choice, expected_choice)

    def test_predict_choice_2(self):
        # Mock tokenizer
        tokenizer = MagicMock()
        tokenizer.encode = MagicMock(
            side_effect=lambda x, add_special_tokens: {'A': [0], 'B': [1], 'C': [2], 'D': [3]}[x])

        # Define toy logits tensor
        logits = torch.tensor([[
            [0.1, 0.5, 0.2, 10.3]  # Logits for the last token (answer token)
        ]])

        # Expected result: The index of the predicted choice
        expected_choice = 3  # 'B' has the highest logit (0.5)

        # Calculate the predicted choice using the function
        predicted_choice = predict_by_token_id(logits, tokenizer)

        # Assert that the predicted choice matches the expected result
        self.assertEqual(predicted_choice, expected_choice)