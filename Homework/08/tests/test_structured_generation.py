import pytest

from fsm import build_odd_zeros_fsm
from structured_generation import get_valid_tokens, random_generation
from tests.test_fsm import build_aboba_fsm


@pytest.mark.parametrize(
    "prefix,expected_tokens",
    [("", [1, 2, 3, 4, 5, 6]), ("1010", [1, 2, 3, 4, 5, 6]), ("101010", [0, 1, 2, 3, 4, 5, 6])],
)
def test_get_valid_tokens_odd_zeros(prefix, expected_tokens):
    fsm, start = build_odd_zeros_fsm()
    vocab = {0: "[EOS]", 1: "0", 2: "00", 3: "000", 4: "1", 5: "10", 6: "01"}
    eos_token_id = 0

    state = fsm.move(prefix, start)
    actual_tokens = get_valid_tokens(vocab, eos_token_id, fsm, state)
    assert set(expected_tokens) == set(actual_tokens)


@pytest.mark.parametrize("prefix,expected_tokens", [("", [1, 4]), ("a", [2]), ("ab", [3]), ("aboba", [0, 1, 4])])
def test_get_valid_tokens_aboba(prefix, expected_tokens):
    fsm = build_aboba_fsm()
    vocab = {0: "[EOS]", 1: "a", 2: "b", 3: "o", 4: "aboba"}
    eos_token_id = 0

    state = fsm.move(prefix)
    actual_tokens = get_valid_tokens(vocab, eos_token_id, fsm, state)
    assert set(expected_tokens) == set(actual_tokens)


def test_random_generation(subtests, n_tries=100):
    for i in range(n_tries):
        with subtests.test(i=i):
            generated_tokens = random_generation()
            unique_tokens = set(generated_tokens)
            for token in unique_tokens:
                assert token in {"0", "1"}
            n_zeros = generated_tokens.count("0")
            assert n_zeros % 2 == 1
