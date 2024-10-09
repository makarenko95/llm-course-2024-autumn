import sys
sys.path.append("Homework/02")
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

from solution import compute_attention, compute_multihead_attention


N_HEADS = 3
DIM_PER_HEAD = 64
HIDDEN_DIM = N_HEADS * DIM_PER_HEAD
BATCH_SIZE = 32
SEQ_LENGTH = 128

x = torch.rand(BATCH_SIZE, SEQ_LENGTH, HIDDEN_DIM)

Q = nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=False)
K = nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=False)
V = nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=False)

q = Q(x)
k = K(x)
v = V(x)


class TestAttention(unittest.TestCase):

    def test_one_head(self):
        custom_attention_out = compute_attention(queries=q, keys=k, values=v)
        default_attention_out = F.scaled_dot_product_attention(query=q, key=k, value=v)
        self.assertTrue(
            torch.allclose(custom_attention_out, default_attention_out, atol=1e-7)
        )

    def test_multihead(self):
        mha = nn.MultiheadAttention(
            embed_dim=HIDDEN_DIM, num_heads=N_HEADS, batch_first=True, dropout=0, bias=False
        )
        mha.eval()
        default_mha, _ = mha(query=q, key=k, value=v, need_weights=False)

        w_q, w_k, w_v = mha.in_proj_weight.chunk(3)
        q_proj = (q.transpose(0, 1) @ w_q.T).view(SEQ_LENGTH, BATCH_SIZE * N_HEADS, DIM_PER_HEAD).transpose(0, 1)
        k_proj = (k.transpose(0, 1) @ w_k.T).view(SEQ_LENGTH, BATCH_SIZE * N_HEADS, DIM_PER_HEAD).transpose(0, 1)
        v_proj = (v.transpose(0, 1) @ w_v.T).view(SEQ_LENGTH, BATCH_SIZE * N_HEADS, DIM_PER_HEAD).transpose(0, 1)

        q_ = q_proj.view(BATCH_SIZE, N_HEADS, SEQ_LENGTH, DIM_PER_HEAD)
        k_ = k_proj.view(BATCH_SIZE, N_HEADS, SEQ_LENGTH, DIM_PER_HEAD)
        v_ = v_proj.view(BATCH_SIZE, N_HEADS, SEQ_LENGTH, DIM_PER_HEAD)
    
        custom_mha = compute_multihead_attention(
            queries=q_, keys=k_, values=v_, projection_matrix=mha.out_proj.weight
        )

        self.assertTrue(
            torch.allclose(default_mha, custom_mha, atol=1e-7)
        )
