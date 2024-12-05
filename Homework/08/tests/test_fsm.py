import random

import pytest

from fsm import FSM, State, build_odd_zeros_fsm


def build_aboba_fsm():
    """
    FSM that accepts 1 or more "aboba"

    (q0) --a--> (q1) --b--> (q2) --o--> (q3) --b--> (q4) --a--> ((q5))
    ^                                                             |
    |____________________________a________________________________|
    """
    q0 = State(is_terminal=False)
    q1 = State(is_terminal=False)
    q2 = State(is_terminal=False)
    q3 = State(is_terminal=False)
    q4 = State(is_terminal=False)
    q5 = State(is_terminal=True)

    q0.add_transition("a", q1)
    q1.add_transition("b", q2)
    q2.add_transition("o", q3)
    q3.add_transition("b", q4)
    q4.add_transition("a", q5)
    q5.add_transition("a", q1)

    return FSM([q0, q1, q2, q3, q4, q5], initial=0)


@pytest.mark.parametrize("candidate", ["abobaboba", "test", "", "ab", "ababa", "abobaa"])
def test_aboba_fsm_no_accept(candidate):
    fsm = build_aboba_fsm()
    assert not fsm.accept(candidate)


@pytest.mark.parametrize("num_repeats", range(1, 10))
def test_aboba_fsm_accept(num_repeats):
    fsm = build_aboba_fsm()
    line = "aboba" * num_repeats
    assert fsm.accept(line)


@pytest.mark.parametrize("prefix,continuation", [("a", "b"), ("abobaab", "o"), ("aboba", "a")])
def test_aboba_move_validate_correct(prefix, continuation):
    fsm = build_aboba_fsm()
    inter_state = fsm.move(prefix)
    assert fsm.validate_continuation(inter_state, continuation)


@pytest.mark.parametrize("prefix,continuation", [("a", "b"), ("abobaab", "o"), ("aboba", "a")])
def test_aboba_move_validate_correct(prefix, continuation):
    fsm = build_aboba_fsm()
    inter_state = fsm.move(prefix)
    assert fsm.validate_continuation(inter_state, continuation)


@pytest.mark.parametrize("prefix,continuation", [("a", "a"), ("abobaab", "a"), ("aboba", "b")])
def test_aboba_move_validate_incorrect(prefix, continuation):
    fsm = build_aboba_fsm()
    inter_state = fsm.move(prefix)
    assert not fsm.validate_continuation(inter_state, continuation)


def test_odd_zeros_fsm(subtests, n_tries=100):
    fsm, _ = build_odd_zeros_fsm()
    for i in range(n_tries):
        with subtests.test(i=i):
            num_len = random.randint(1, 100)
            test_num = "".join(random.choice("01") for _ in range(num_len))
            n_zeros = test_num.count("0")
            is_odd_zeros = n_zeros % 2 == 1
            assert fsm.accept(test_num) == is_odd_zeros
