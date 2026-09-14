import torch
import torch.nn as nn
from typing import Tuple, Optional

class KVCache:
    def __init__(self):
        self.cache_k: Optional[torch.Tensor] = None  # (batch, seq_len, model_dim)
        self.cache_v: Optional[torch.Tensor] = None

    def update(self, new_k: torch.Tensor, new_v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Append new_k and new_v to the cache along the sequence dimension (dim=1).
        # On the first call, initialize the cache with the given tensors.
        if self.cache_k is None:
            self.cache_k = new_k
            self.cache_v = new_v
        else:
            self.cache_k = torch.cat((self.cache_k, new_k), dim=1)
            self.cache_v = torch.cat((self.cache_v, new_v), dim=1)
        
        # Return the full (cached) K and V tensors.
        return (self.cache_k, self.cache_v)

    def clear(self):
        self.cache_k = None
        self.cache_v = None

class CachedAttention(nn.Module):
    def __init__(self, model_dim: int):
        super().__init__()
        torch.manual_seed(0)
        self.q_proj = nn.Linear(model_dim, model_dim, bias=False)
        self.k_proj = nn.Linear(model_dim, model_dim, bias=False)
        self.v_proj = nn.Linear(model_dim, model_dim, bias=False)

    def forward(self, x: torch.Tensor, kv_cache: Optional[KVCache] = None) -> Tuple[torch.Tensor, KVCache]:
        # 1. Project x into Q, K, V using the linear layers
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 2. If kv_cache is None, create a new KVCache
        if kv_cache is None:
            kv_cache = KVCache()
        # 3. Update the cache with the new K and V
        full_k, full_v = kv_cache.update(k, v)
        print(full_v.shape)
        # print(q.shape)
        # 4. Compute scaled dot-product attention using Q and the full cached K, V
        scores = (q @ full_k.transpose(-2, -1)) * (full_k.shape[-1] ** -0.5)
        print(scores.shape)
        seq_len = scores.size(1)
        low_tri = torch.tril(torch.ones(seq_len, seq_len))
        mask = (low_tri == 0)
        scores = scores.masked_fill(mask, float('-inf'))
        
        output = torch.softmax(scores, dim=-1) @ full_v

        # 5. Apply a causal mask offset by the number of previously cached tokens
        # 6. Return (rounded output, kv_cache)
        return torch.round(output, decimals=4), kv_cache
