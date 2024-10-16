import pytest
import torch

from alibi import compute_alibi


def _assert_sequence_equal(expected, actual):
    assert len(expected) == len(actual), f"Unmatched sequence lengths: {len(expected)} vs. {len(actual)}"
    for i, (elem1, elem2) in enumerate(zip(expected, actual)):
        assert elem1 == elem2, f"Difference at position {i}: {elem1} vs {elem2}"


@pytest.mark.parametrize("num_heads", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("seq_len", [4, 8, 16, 128])
def test_output_shape(num_heads, seq_len):
    biases = compute_alibi(num_heads, seq_len)
    _assert_sequence_equal(biases.size(), (num_heads, seq_len, seq_len))


@pytest.mark.parametrize("num_heads", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("seq_len", [4, 8, 16, 128])
def test_bias_symmetric(num_heads, seq_len):
    biases = compute_alibi(num_heads, seq_len)
    for head in range(num_heads):
        torch.testing.assert_close(
            biases[head], -biases[head].T, msg=f"Biases asymmetrical about the diagonal for head {head}"
        )


def test_slope_computation():
    num_head = 8
    seq_len = 16
    slopes = [1 / (2**i) for i in range(1, num_head + 1)]
    biases = compute_alibi(num_head, seq_len)
    for head in range(num_head):
        for position in range(1, seq_len):
            assert biases[head, 0, position] / position == slopes[head]


def test_alibi_computation():
    relative_positions = torch.tensor([[0, 1, 2, 3], [-1, 0, 1, 2], [-2, -1, 0, 1], [-3, -2, -1, 0]])
    slopes = torch.tensor([2**-2, 2**-4, 2**-6, 2**-8]).view(-1, 1, 1)
    expected_bias = relative_positions * slopes

    num_head = 4
    seq_len = 4
    actual_bias = compute_alibi(num_head, seq_len)

    torch.testing.assert_close(expected_bias, actual_bias)
