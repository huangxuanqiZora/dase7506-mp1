"""MP1 student implementation.

A compute-matched, modernized decoder-only transformer. Every design choice is
a config flag so the same code path can produce the baseline-style control and
the full model, which keeps the ablations clean:

    config['pos_emb']  'rope' (default) | 'learned'
    config['mlp']      'swiglu' (default) | 'gelu'
    config['norm']     'rmsnorm' (default) | 'layernorm'
    config['qk_norm']  bool (default False)
    config['tied']     bool (default True)

The model exposes the two interfaces required by the classroom pipeline:
``forward(ids)`` returns unnormalized logits and ``predict_log_probs(ids)``
returns finite, normalized natural-log probabilities. Both are strictly causal
and keep no state across calls, so independent 256-token windows start fresh.

``build_model(config)`` is the factory used by ``train.py`` / ``evaluate.py``.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    """Root-mean-square normalization (no mean subtraction, no bias)."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        scale = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x * scale


class LayerNorm(nn.Module):
    """Standard LayerNorm without bias (used for the RMSNorm ablation)."""

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return F.layer_norm(x, (x.shape[-1],), self.weight, None, self.eps)


def make_norm(kind, dim):
    return RMSNorm(dim) if kind == 'rmsnorm' else LayerNorm(dim)


def build_rope_cache(head_dim, length, base=10000.0):
    """Precompute cos/sin tables of shape [length, head_dim]."""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    positions = torch.arange(length).float()
    freqs = torch.outer(positions, inv_freq)          # [length, head_dim/2]
    emb = torch.cat((freqs, freqs), dim=-1)           # [length, head_dim]
    return emb.cos(), emb.sin()


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x, cos, sin):
    """x: [batch, heads, length, head_dim]; cos/sin: [length, head_dim]."""
    return x * cos + rotate_half(x) * sin


class Attention(nn.Module):
    def __init__(self, config, width, heads):
        super().__init__()
        self.heads = heads
        self.head_dim = width // heads
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.proj = nn.Linear(width, width, bias=False)
        self.qk_norm = bool(config.get('qk_norm', False))
        self.q_norm = RMSNorm(self.head_dim) if self.qk_norm else None
        self.k_norm = RMSNorm(self.head_dim) if self.qk_norm else None
        self.rope = config.get('pos_emb', 'rope') == 'rope'
        self.dropout = float(config.get('dropout', 0.0))

    def forward(self, x, cos, sin):
        batch, length, width = x.shape
        q, k, v = self.qkv(x).view(batch, length, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)
        if self.rope:
            cos, sin = cos[:length], sin[:length]
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                                  dropout_p=self.dropout if self.training else 0.0)
        return self.proj(attended.transpose(1, 2).reshape(batch, length, width))


class SwiGLU(nn.Module):
    def __init__(self, width, hidden, dropout):
        super().__init__()
        self.up = nn.Linear(width, 2 * hidden, bias=False)
        self.down = nn.Linear(hidden, width, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        a, b = self.up(x).chunk(2, dim=-1)
        return self.dropout(self.down(F.silu(a) * b))


class GELUMLP(nn.Module):
    def __init__(self, width, hidden, dropout):
        super().__init__()
        self.fc = nn.Linear(width, hidden, bias=False)
        self.proj = nn.Linear(hidden, width, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    def __init__(self, config, width, heads, hidden):
        super().__init__()
        norm = config.get('norm', 'rmsnorm')
        dropout = float(config.get('dropout', 0.0))
        self.norm1, self.norm2 = make_norm(norm, width), make_norm(norm, width)
        self.attn = Attention(config, width, heads)
        mlp = config.get('mlp', 'swiglu')
        self.mlp = SwiGLU(width, hidden, dropout) if mlp == 'swiglu' else GELUMLP(width, hidden, dropout)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        return x + self.mlp(self.norm2(x))


class ModernGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        self.context = int(config['context'])
        width = int(config['width'])
        heads = int(config['heads'])
        depth = int(config['depth'])
        vocab = int(config['vocab'])
        dropout = float(config.get('dropout', 0.0))
        hidden = int(config.get('mlp_hidden', 4 * width))
        base = float(config.get('rope_base', 10000.0))
        self.rope = config.get('pos_emb', 'rope') == 'rope'

        self.token = nn.Embedding(vocab, width)
        if not self.rope:
            self.pos = nn.Embedding(self.context, width)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(config, width, heads, hidden) for _ in range(depth)])
        self.norm = make_norm(config.get('norm', 'rmsnorm'), width)
        self.head = nn.Linear(width, vocab, bias=False)

        head_dim = width // heads
        cos, sin = build_rope_cache(head_dim, self.context, base)
        self.register_buffer('rope_cos', cos, persistent=False)
        self.register_buffer('rope_sin', sin, persistent=False)

        self.apply(self._init_weights)
        for name, param in self.named_parameters():
            if name.endswith('proj.weight') or name.endswith('down.weight'):
                nn.init.normal_(param, std=0.02 / math.sqrt(2 * depth))
        if config.get('tied', True):
            self.head.weight = self.token.weight

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def features(self, ids):
        x = self.token(ids)
        if not self.rope:
            positions = torch.arange(ids.shape[1], device=ids.device)
            x = x + self.pos(positions)
        x = self.drop(x)
        cos, sin = self.rope_cos.to(ids.device), self.rope_sin.to(ids.device)
        for block in self.blocks:
            x = block(x, cos, sin)
        return self.norm(x)

    def forward(self, ids):
        """Training interface: unnormalized next-token logits [batch, time, vocab]."""
        return self.head(self.features(ids))

    def predict_log_probs(self, ids):
        """Evaluation interface: normalized log probabilities [batch, time, vocab]."""
        return F.log_softmax(self(ids).float(), dim=-1)


def build_model(config):
    return ModernGPT(config)
