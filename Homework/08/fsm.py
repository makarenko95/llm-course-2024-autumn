from dataclasses import dataclass, field
from typing import Optional


@dataclass
class State:
    is_terminal: bool
    transitions: dict[str, "State"] = field(default_factory=dict)

    def add_transition(self, char, state):
        self.transitions[char] = state


class FSM:
    def __init__(self, states: list[State], initial: int):
        self.states = states
        self.initial = initial

    def is_terminal(self, state_id):
        return self.states[state_id].is_terminal

    def move(self, line: str, start: Optional[int] = None) -> Optional[int]:
        """Iterate over the FSM from the given state using symbols from the line.
        If no possible transition is found during iteration, return None.
        If no given state start from initial.
        
        Args:
            line (str): line to iterate via FSM
            start (optional int): if passed, using as start start
        Returns:
            end (optional int): end state if possible, None otherwise
        """
        raise NotImplementedError

    def accept(self, candidate: str) -> bool:
        """Check if the candidate is accepted by the FSM.

        Args:
            candidate (str): line to check
        Returns:
            is_accept (bool): result of checking
        """
        raise NotImplementedError

    def validate_continuation(self, state_id: int, continuation: str) -> bool:
        """Check if the continuation can be achieved from the given state.

        Args:
            state_id (int): state to iterate from
            continuation (str): continuation to check
        Returns:
            is_possible (bool): result of checking
        """
        raise NotImplementedError


def build_odd_zeros_fsm() -> tuple[FSM, int]:
    """FSM that accepts binary numbers with odd number of zeros

    For example,
    - correct words: 0, 01, 10, 101010
    - incorrect words: 1, 1010

    Args:
    Returns:
        fsm (FSM): FSM
        start_state (int): index of initial state
    """
    raise NotImplementedError



if __name__ == "__main__":
    _fsm, _ = build_odd_zeros_fsm()
    print("101010 -- ", _fsm.accept("101010"))
    print("10101 -- ", _fsm.accept("10101"))
