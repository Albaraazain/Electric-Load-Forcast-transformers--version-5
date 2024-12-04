"""
Custom layers for the Informer model implementation.
Reference: https://github.com/zhouhaoyi/Informer2020

Dependencies:
- torch>=2.0.1
- numpy>=1.24.3
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import sqrt
from typing import Optional, Tuple

class ProbAttention(nn.Module):
    """
    Probability Attention mechanism with complexity reduction from O(L*L) to O(L*ln(L)).
    """
    def __init__(
            self,
            mask_flag: bool = True,
            factor: int = 5,
            scale: Optional[float] = None,
            attention_dropout: float = 0.1,
            output_attention: bool = False
    ):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q: torch.Tensor, K: torch.Tensor, sample_k: int, n_top: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate the sparse attention using sampling with fixed memory handling"""
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # Sample K for each query instead of expanding K
        K_expand = []
        for q_idx in range(L_Q):
            # Sample random indices for this query
            index_sample = torch.randint(L_K, (sample_k,), device=K.device)
            # Get sampled keys
            K_sample = K[:, :, index_sample, :]
            K_expand.append(K_sample)

        # Stack sampled keys
        K_expand = torch.stack(K_expand, dim=2)  # B x H x L_Q x sample_k x E

        # Calculate Q_K_sample more efficiently
        Q_expanded = Q.unsqueeze(-2)  # B x H x L_Q x 1 x E
        Q_K_sample = torch.matmul(Q_expanded, K_expand.transpose(-2, -1))  # B x H x L_Q x 1 x sample_k
        Q_K_sample = Q_K_sample.squeeze(-2)  # B x H x L_Q x sample_k

        # Find Top_k queries with sparsity measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # Use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                    torch.arange(H)[None, :, None],
                    M_top, :]
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))

        # Clean up to free memory
        del K_expand, Q_expanded, Q_K_sample, M
        torch.cuda.empty_cache()

        return Q_K, M_top

    def _get_initial_context(self, V: torch.Tensor, L_Q: int) -> torch.Tensor:
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # More memory efficient mean calculation
            V_sum = torch.mean(V, dim=-2, keepdim=True)
            context = V_sum.expand(B, H, L_Q, D)
        else:
            # More memory efficient cumsum
            context = torch.cumsum(V, dim=-2)
            context = torch.div(context, torch.arange(1, L_V + 1, device=V.device).view(1, 1, -1, 1))
        return context

    def forward(self, queries, keys, values, attn_mask):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)

        # Add scale factor
        scale = self.scale or 1./sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale

        # Get attention and context
        context = self._get_initial_context(values, L_Q)

        # Update context with selected top_k queries
        scores_top = self.dropout(F.softmax(scores_top, dim=-1))
        
        # Clone context before updating to avoid memory overlap
        context_clone = context.clone()
        context_clone[torch.arange(B)[:, None, None],
                     torch.arange(H)[None, :, None],
                     index, :] = torch.matmul(scores_top, values)

        return context_clone.transpose(2,1).contiguous(), scores_top if self.output_attention else None

class AttentionLayer(nn.Module):
    """
    Attention Layer with ProbAttention mechanism.
    """
    def __init__(
            self,
            attention: nn.Module,
            d_model: int,
            n_heads: int,
            d_keys: Optional[int] = None,
            d_values: Optional[int] = None
    ):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(
            self,
            queries: torch.Tensor,
            keys: torch.Tensor,
            values: torch.Tensor,
            attn_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )

        out = out.view(B, L, -1)
        return self.out_projection(out), attn