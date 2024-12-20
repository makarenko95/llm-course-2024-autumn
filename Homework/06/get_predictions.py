import torch
import torch.nn.functional as F
from transformers import AutoTokenizer


def predict_by_token_id(logits: torch.Tensor, tokenizer: AutoTokenizer) -> int:
    """
    Determines the predicted choice based on the logits of the model's output.

    Args:
        logits (torch.Tensor): The logits output from the model, typically of shape (1, sequence_length, vocab_size).
        tokenizer (AutoTokenizer): The tokenizer used to encode the input prompt.

    Returns:
        int: The index of the predicted choice (0 for 'A', 1 for 'B', 2 for 'C', 3 for 'D').
    """
    <ВАШ КОД ЗДЕСЬ>

    return ...


def get_choice_log_probs(logits: torch.Tensor, input_ids: torch.Tensor) -> float:
    """
    Calculates the average log probabilities of predicted tokens for a given sequence.


    Args:
        logits (torch.Tensor): A tensor of logits generated by the model, with shape (batch_size, sequence_length, vocab_size).
        input_ids (torch.Tensor): A tensor of input token IDs, with shape (batch_size, sequence_length).

    Returns:
         float: The average log probability of the predicted tokens.
    """
    <ВАШ КОД ЗДЕСЬ>

    return ...
