import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SelfAttention(nn.Module):

    def __init__(self, n_heads: int, d_embed: int, in_proj_bias=True, out_proj_bias=True):
        # d_embed == token num of a sentence == feature num (in image domain)
        super().__init__()

        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias) # projection of input; input -> (Q, K, V)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads # split features

    def forward(self, x: torch.Tensor, casual_mask=False) -> torch.Tensor:
        input_shape = x.shape
        batch, seq_len, d_embed = input_shape

        interm_shape = (batch, seq_len, self.n_heads, self.d_head)

        # split features in Q, K,  V in order
        q, k, v = self.in_proj(x).chunk(3, dim=-1) # split features

        # (B, SeqLen, Dim) -> (B, SeqLen, H, Dim/H) -> (B, H, SeqLen, Dim/H)
        q = q.view(interm_shape).transpose(1,2)
        k = k.view(interm_shape).transpose(1,2)
        v = v.view(interm_shape).transpose(1,2)

        weight = q @ k.transpose(-1, -2)

        if casual_mask:
            # Masking upper triangle -> 1
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            weight.masked_fill_(mask, float('-inf'))

        weight /= math.sqrt(self.d_head)

        weight = F.softmax(weight, dim=-1)
        
        # (B, H, SeqLen, SeqLen) @ (B, H, SeqLen, Dim/H) -> (B, H, SeqLen, Dim/H)
        out = weight @ v

        out = out.transpose(1, 2)
        out = out.reshape(input_shape)

        # (batch, seq_len, d_embed)
        return self.out_proj(out)
