import pytest
import torch

from gqa import scaled_dot_product_gqa


@pytest.mark.parametrize("embed_dim", [64])
@pytest.mark.parametrize("num_heads", [2, 4, 8, 16])
@pytest.mark.parametrize("kv_heads", [4, 8])
@pytest.mark.parametrize("is_causal", [True, False])
@pytest.mark.parametrize("seq_len", [8, 16])
@pytest.mark.parametrize("kv_seq_len", [8, 16])
def test_grouped_scaled_dot_product_attention(embed_dim, num_heads, kv_heads, is_causal, seq_len, kv_seq_len):
    x = torch.randn(1, seq_len, num_heads, embed_dim)
    kv = torch.randn(1, kv_seq_len, kv_heads, embed_dim)

    if kv_heads > num_heads:
        with pytest.raises(ValueError):
            scaled_dot_product_gqa(x, kv, kv, is_causal=is_causal)
        return

    out, attn_weights = scaled_dot_product_gqa(x, kv, kv, is_causal=is_causal, need_weights=True)
    assert out.size(0) == 1
    assert out.size(1) == seq_len
    assert out.size(2) == num_heads
    assert out.size(3) == embed_dim
    assert attn_weights.size(0) == 1
    assert attn_weights.size(1) == num_heads
    assert attn_weights.size(2) == seq_len
    assert attn_weights.size(3) == kv_seq_len

    # Test that grouped SDPA is equivalent to SDPA if we duplicate the KV heads.
    kv = kv.repeat_interleave(num_heads // kv_heads, dim=2)
    kv = kv.permute(0, 2, 1, 3)
    x = x.permute(0, 2, 1, 3)
    out_vanilla = torch.nn.functional.scaled_dot_product_attention(x, kv, kv, is_causal=is_causal)
    out_vanilla = out_vanilla.permute(0, 2, 1, 3)
    torch.testing.assert_close(out, out_vanilla)
