import torch
import torch.nn as nn
from torchtyping import TensorType

class GroupedQueryAttention(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, num_kv_heads: int):
        super().__init__()
        torch.manual_seed(0)
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = model_dim // num_heads

        self.q_proj = nn.Linear(model_dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(model_dim, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(model_dim, num_kv_heads * self.head_dim, bias=False)
        self.output_proj = nn.Linear(num_heads * self.head_dim, model_dim, bias=False)

    def forward(self, x: TensorType[float]) -> TensorType[float]:
        B, T, D = x.shape
        # print(x.shape)
        # 1. Project x into Q, K, V using the projection layers
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        # print(q)
        # print(num_heads, num_kv_heads)
        # 2. Reshape into heads: Q has num_heads, K and V have num_kv_heads
        q_heads = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        # print(q_heads.shape)
        k_heads = k.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v_heads = v.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        # print(k_heads.shape, v_heads.shape)
        
        # 3. Expand K, V by repeating each KV head (num_heads // num_kv_heads) times
        repeats = self.num_heads // self.num_kv_heads
        # print(repeats) 
        k_heads = k_heads.repeat_interleave(repeats, dim=1)
        v_heads = v_heads.repeat_interleave(repeats, dim=1)

        # 4. Compute scaled dot-product attention with causal mask
        scores = (q_heads @ k_heads.transpose(-2, -1)) * (self.head_dim ** -0.5)
        mask =  torch.tril(torch.ones(T, T))
        scores = scores.masked_fill(mask == 0, float('-inf'))
        # 5. Concatenate heads and apply output projection
        out = torch.softmax(scores, dim=-1) @ v_heads
        # print(out.shape) 
        output = out.transpose(1, 2).reshape(B, T, -1)
        # print(out.shape) 
        # 6. Return rounded output (decimals=4)
        return torch.round(self.output_proj(output), decimals=4)
