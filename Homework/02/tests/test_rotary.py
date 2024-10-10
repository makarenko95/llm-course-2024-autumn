import sys
sys.path.append("Homework/02")
import unittest

import torch
from torchtune.modules import RotaryPositionalEmbeddings

from solution import compute_rotary_embeddings


N_HEADS = 3
DIM_PER_HEAD = 64
HIDDEN_DIM = N_HEADS * DIM_PER_HEAD
BATCH_SIZE = 32
SEQ_LENGTH = 128

x = torch.rand(BATCH_SIZE, SEQ_LENGTH, N_HEADS, DIM_PER_HEAD)


class TestRotary(unittest.TestCase):

    def test_rotary_embeddings(self):
        custom_rope = compute_rotary_embeddings(x)

        rpe = RotaryPositionalEmbeddings(dim=DIM_PER_HEAD, max_seq_len=SEQ_LENGTH)
        default_rope = rpe(x)

        self.assertTrue(
            torch.allclose(default_rope, custom_rope, atol=1e-5)
        )
