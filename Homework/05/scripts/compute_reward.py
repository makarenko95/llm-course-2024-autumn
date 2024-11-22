from torch import Tensor, no_grad

def compute_reward(reward_model, reward_tokenizer, texts: list[str], device='cpu') -> Tensor:
    """
    Compute the reward scores for a list of texts using a specified reward model and tokenizer.

    Parameters:
    reward_model: The model used to compute the reward scores
    reward_tokenizer: The tokenizer for reward_model
    texts (list[str]): A list of text strings for which the reward scores are to be computed.
    device (str, optional): The device on which the computation should be performed. Default is 'cpu'.

    Returns:
    torch.Tensor: A tensor containing the reward scores for each input text. The scores are extracted
                  from the logits of the reward model.

    Example:
    >>> compute_reward(my_reward_model, my_reward_tokenizer, ["text1", "text2"])
    tensor([ 5.1836, -4.8438], device='cpu')
    """
    raise NotImplementedError

    # <YOUR CODE HERE>
    with no_grad():
        # <YOUR CODE HERE>