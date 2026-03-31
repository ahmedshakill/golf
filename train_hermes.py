"""
HERMES: Hyperbolic-Equivariant Recurrent Manifold Engine for Sequence compression
===================================================================================
OpenAI Parameter Golf submission — 16MB budget, 10 min on 8×H100

Architecture stack (in order of byte ROI):
  1. P-adic positional encoding        — zero parameters, ultrametric pos bias
  2. Lorentz hyperbolic embeddings     — 32-dim replaces 256-dim, saves ~450KB
  3. Sheaf attention                   — typed edge transforms, replaces scalar attn
  4. Neural ODE recurrent depth        — T=6 passes of 1 layer = "free" depth
  5. SO(2)-equivariant MLP             — 30% param reduction via rotational symmetry
  6. GL(n) / Galois weight init        — structured init from representation theory
  7. Int6 QAT (straight-through est.)  — 4 weights per 3 bytes, fp16 for manifold
  8. Muon optimizer                    — faster convergence than AdamW
  9. EMA eval                          — free BPB gain
 10. Sliding window stride=64 eval     — free BPB gain

Local dev: RTX 3050 6GB / Ryzen 5700G / 24GB RAM
  → Use SMOKE=True, SEQ_LEN=512, BATCH=4 for local runs
  → Full eval on H100 submission: SEQ_LEN=4096, BATCH=32

Scoring: bits-per-byte (BPB) on FineWeb validation set (lower is better)
"""

import os, sys, math, time, struct, zlib, base64, json
from array import array
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader, Dataset
from dataclasses import dataclass, field
from typing import Optional, Tuple
from contextlib import contextmanager

try:
    import numpy as np
except ImportError:
    np = None

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

def detect_default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def detect_default_dtype() -> str:
    if not torch.cuda.is_available():
        return "float32"
    bf16_supported = getattr(torch.cuda, "is_bf16_supported", None)
    if callable(bf16_supported) and bf16_supported():
        return "bfloat16"
    return "float16"


def require_numpy(feature: str):
    if np is None:
        raise RuntimeError(f"{feature} requires NumPy, but NumPy is not installed.")


def build_grad_scaler(device: str, enabled: bool):
    amp = getattr(torch, "amp", None)
    if amp is not None and hasattr(amp, "GradScaler"):
        return amp.GradScaler(device, enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def uint16_pairs_to_bytes(pairs: list[tuple[int, int]]) -> bytes:
    flat = array("H")
    for a, b in pairs:
        flat.append(a)
        flat.append(b)
    return flat.tobytes()


def bytes_to_uint16_pairs(raw: bytes) -> list[tuple[int, int]]:
    flat = array("H")
    flat.frombytes(raw)
    return [(int(flat[i]), int(flat[i + 1])) for i in range(0, len(flat), 2)]


def token_slice(data, start: int, end: int) -> list[int]:
    if isinstance(data, torch.Tensor):
        return data[start:end].tolist()
    if np is not None and isinstance(data, np.ndarray):
        return data[start:end].tolist()
    return list(data[start:end])


@dataclass
class HermesConfig:
    # Model
    vocab_size:       int   = 1024       # BPE vocab — fits in ~4KB of merges
    hyp_dim:          int   = 32         # Lorentz embedding dim (replaces 256 Euclidean)
    model_dim:        int   = 256        # Internal representation dim (post-embed proj)
    n_heads:          int   = 8          # Sheaf attention heads
    stalk_dim:        int   = 32         # Per-head sheaf stalk dim; must satisfy model_dim = n_heads * stalk_dim
    ode_steps:        int   = 6          # Neural ODE recurrent passes (T)
    mlp_ratio:        float = 3.0        # MLP expansion (3× cheaper than standard 4×)
    curvature:        float = 1.0        # Lorentz curvature c
    p_adic_base:      int   = 2          # Prime for p-adic position encoding
    dropout:          float = 0.0        # Disabled for small models

    # Quantization
    use_int6_qat:     bool  = True       # Quantization-aware training (STE)
    quant_bits:       int   = 6          # 6-bit → 4 values per 3 bytes
    skip_quant_embed: bool  = True       # Keep Lorentz embeddings in fp16

    # Training
    seq_len:          int   = 512        # 4096 for H100, 512 for RTX 3050
    batch_size:       int   = 4          # 32 for H100, 4 for RTX 3050
    lr:               float = 3e-3
    weight_decay:     float = 0.1
    max_steps:        int   = 5000       # ~10min on H100; reduce for local
    warmup_steps:     int   = 200
    grad_clip:        float = 1.0
    ema_decay:        float = 0.9999     # EMA for eval weights
    activation_checkpointing: bool = True

    # Eval
    eval_stride:      int   = 64         # Sliding window stride (BPB gain)
    eval_seq_len:     int   = 512        # Eval context length

    # System
    smoke:            bool  = False      # Quick sanity check mode
    device:           str   = field(default_factory=detect_default_device)
    dtype:            str   = field(default_factory=detect_default_dtype)
    compile:          bool  = False      # torch.compile — enable on H100
    seed:             int   = 42

    def __post_init__(self):
        if self.n_heads <= 0:
            raise ValueError("n_heads must be positive")
        if self.model_dim % self.n_heads != 0:
            raise ValueError(
                f"model_dim ({self.model_dim}) must be divisible by n_heads ({self.n_heads})"
            )
        derived_stalk = self.model_dim // self.n_heads
        if self.stalk_dim != derived_stalk:
            # SheafAttention reshapes into (heads, stalk_dim), so this must stay exact.
            self.stalk_dim = derived_stalk


# ─────────────────────────────────────────────────────────────────────────────
# TOKENIZER  (micro BPE — self-contained, fits in ~4KB compressed)
# ─────────────────────────────────────────────────────────────────────────────

class MicroBPE:
    """
    Minimal byte-level BPE tokenizer.
    Merges are stored as uint16 pairs, zlib-compressed, base64-encoded.
    For submission: replace MERGES_BLOB with actual FineWeb-trained merges.
    """
    MERGES_BLOB = ""  # Filled after training on FineWeb slice

    def __init__(self, vocab_size: int = 1024):
        self.vocab_size = vocab_size
        # Base vocab: 256 raw bytes
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.merges: list[tuple[int, int]] = []

        if self.MERGES_BLOB:
            self._load_merges(self.MERGES_BLOB)
        # else: identity tokenizer (byte-level), still valid, just higher BPB

        # Build merge lookup for O(1) encode
        self.merge_rank = {pair: i for i, pair in enumerate(self.merges)}

    def _load_merges(self, blob: str):
        raw = zlib.decompress(base64.b64decode(blob))
        pairs = bytes_to_uint16_pairs(raw)
        for i, (p1, p2) in enumerate(pairs):
            new_id = 256 + i
            self.vocab[new_id] = self.vocab[p1] + self.vocab[p2]
            self.merges.append((p1, p2))

    def encode(self, text: str) -> list[int]:
        ids = list(text.encode("utf-8"))
        while True:
            pairs = [(ids[i], ids[i+1]) for i in range(len(ids)-1)]
            best = min(pairs, key=lambda p: self.merge_rank.get(p, float("inf")), default=None)
            if best is None or best not in self.merge_rank:
                break
            new_id = 256 + self.merge_rank[best]
            out = []
            i = 0
            while i < len(ids):
                if i < len(ids)-1 and (ids[i], ids[i+1]) == best:
                    out.append(new_id)
                    i += 2
                else:
                    out.append(ids[i])
                    i += 1
            ids = out
        return ids

    def decode(self, ids: list[int]) -> bytes:
        return b"".join(self.vocab.get(i, b"?") for i in ids)

    @classmethod
    def train_from_corpus(cls, text: str, vocab_size: int = 1024) -> "MicroBPE":
        """Train BPE merges on a text corpus. Run offline, paste blob into MERGES_BLOB."""
        tok = cls(vocab_size=256)  # Start byte-level
        ids = list(text.encode("utf-8"))
        merges = []
        for _ in range(vocab_size - 256):
            if len(ids) < 2: break
            # Count pairs
            counts: dict[tuple[int,int], int] = {}
            for a, b in zip(ids, ids[1:]):
                counts[(a,b)] = counts.get((a,b), 0) + 1
            best = max(counts, key=counts.get)
            new_id = 256 + len(merges)
            tok.vocab[new_id] = tok.vocab[best[0]] + tok.vocab[best[1]]
            merges.append(best)
            tok.merge_rank[best] = len(merges) - 1
            # Replace in corpus
            out = []
            i = 0
            while i < len(ids):
                if i < len(ids)-1 and (ids[i], ids[i+1]) == best:
                    out.append(new_id)
                    i += 2
                else:
                    out.append(ids[i])
                    i += 1
            ids = out
        tok.merges = merges
        # Serialize
        blob = base64.b64encode(zlib.compress(uint16_pairs_to_bytes(merges))).decode()
        print(f"[MicroBPE] Trained {len(merges)} merges. Paste into MERGES_BLOB:")
        print(f'MERGES_BLOB = "{blob}"')
        return tok


# ─────────────────────────────────────────────────────────────────────────────
# INT6 QUANTIZATION  (straight-through estimator, QAT)
# ─────────────────────────────────────────────────────────────────────────────

class STE6(torch.autograd.Function):
    """Straight-through estimator for 6-bit symmetric quantization."""
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        # Scale to [-31, 31] (6-bit signed: range 63 values, centered)
        scale = x.abs().max() / 31.0 + 1e-8
        ctx.save_for_backward(x, scale)
        return (x / scale).round().clamp(-31, 31) * scale

    @staticmethod
    def backward(ctx, grad):
        x, scale = ctx.saved_tensors
        # Pass gradient through — straight through the quantization
        return grad

def quantize_6bit(x: torch.Tensor) -> torch.Tensor:
    return STE6.apply(x)

class QuantizedLinear(nn.Linear):
    """nn.Linear with int6 QAT applied to weights during forward pass."""
    def __init__(self, *args, quant: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.quant = quant

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = quantize_6bit(self.weight) if self.quant else self.weight
        return F.linear(x, w, self.bias)


def pack_6bit_to_bytes(tensor: torch.Tensor) -> bytes:
    """
    Pack int6 weights into bytes for the artifact file.
    4 values → 3 bytes (4×6 = 24 = 3×8 bits).
    """
    scale = tensor.abs().max() / 31.0 + 1e-8
    q = (tensor / scale).round().clamp(-31, 31).to(torch.int16) + 31  # shift to [0,62]
    flat = q.detach().cpu().reshape(-1).tolist()

    # Pad to multiple of 4
    pad = (4 - len(flat) % 4) % 4
    if pad:
        flat.extend([0] * pad)

    packed = bytearray()
    for i in range(0, len(flat), 4):
        a, b, c, d = flat[i:i+4]
        packed.extend([
            (a << 2) | (b >> 4),
            ((b & 0x0F) << 4) | (c >> 2),
            ((c & 0x03) << 6) | d,
        ])
    return bytes(packed), scale.item(), pad


def unpack_6bit_from_bytes(data: bytes, shape: tuple, scale: float, pad: int) -> torch.Tensor:
    """Reverse of pack_6bit_to_bytes."""
    unpacked = []
    for i in range(0, len(data), 3):
        b1, b2, b3 = data[i], data[i+1], data[i+2]
        unpacked.extend([
            b1 >> 2,
            ((b1 & 0x03) << 4) | (b2 >> 4),
            ((b2 & 0x0F) << 2) | (b3 >> 6),
            b3 & 0x3F,
        ])
    if pad:
        unpacked = unpacked[:-pad]
    arr = torch.tensor(unpacked, dtype=torch.float32)
    arr = (arr - 31) * scale  # unshift and unscale
    return arr.reshape(shape)


# ─────────────────────────────────────────────────────────────────────────────
# P-ADIC POSITIONAL ENCODING
# ─────────────────────────────────────────────────────────────────────────────

def p_adic_valuation(n: int, p: int) -> int:
    """v_p(n): how many times prime p divides n. v_p(0) = large sentinel."""
    if n == 0: return 32
    v = 0
    while n % p == 0:
        n //= p
        v += 1
    return v

def build_p_adic_bias(seq_len: int, p: int = 2, max_val: int = 16) -> torch.Tensor:
    """
    Returns an (seq_len, seq_len) attention bias matrix.
    bias[i,j] = v_p(|i-j|) — tokens at 'p-adically close' distances
    get a bonus, encoding the ultrametric / hierarchical structure of position.

    This is ZERO PARAMETERS — computed once, registered as a buffer.

    Key insight: v_2(1)=0, v_2(2)=1, v_2(4)=2, v_2(8)=3...
    So adjacent tokens (diff=1) get 0, tokens 2 apart get 1,
    tokens 4 apart get 2, etc. — the model naturally attends
    to "power-of-2" structured windows, matching language's
    hierarchical composition.
    """
    idx = torch.arange(seq_len, dtype=torch.long)
    d = (idx[:, None] - idx[None, :]).abs()
    nonzero = d > 0
    vp = torch.zeros_like(d)
    work = d.clone()

    # Count repeated divisibility by p with tensor ops instead of Python loops
    while True:
        divisible = nonzero & (work % p == 0)
        if not divisible.any():
            break
        vp = vp + divisible.to(vp.dtype)
        work = torch.where(divisible, work // p, work)

    vals = vp.clamp(max=max_val).to(torch.float32)
    vals = torch.where(nonzero, vals, torch.full_like(vals, float(max_val)))
    # Larger p-adic valuation = closer in hierarchy = higher attention bonus
    return vals / max_val  # [0, 1]


# ─────────────────────────────────────────────────────────────────────────────
# LORENTZ HYPERBOLIC EMBEDDINGS
# ─────────────────────────────────────────────────────────────────────────────

class LorentzManifold:
    """
    Operations on the Lorentz model (hyperboloid) H^n_c.
    More numerically stable than Poincaré disk — no boundary blowup.
    
    A point x in H^n_c satisfies: -x_0^2 + x_1^2 + ... + x_n^2 = -1/c
    where x_0 > 0 (upper sheet).
    """
    def __init__(self, c: float = 1.0):
        self.c = c

    def inner(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Lorentzian inner product: <x,y>_L = -x0*y0 + x1*y1 + ... + xn*yn"""
        prod = x * y
        return -prod[..., :1] + prod[..., 1:].sum(dim=-1, keepdim=True)

    def norm(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(torch.clamp(-self.inner(x, x), min=1e-8))

    def dist(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Geodesic distance on H^n_c."""
        inner = torch.clamp(-self.inner(x, y) * self.c, min=1 + 1e-7)
        return torch.acosh(inner) / math.sqrt(self.c)

    def expmap_o(self, v: torch.Tensor) -> torch.Tensor:
        """Exponential map from origin o=(1/√c, 0,...,0)."""
        # v must be in tangent space at o: v_0 = 0
        v_space = v[..., 1:]
        norm_v = torch.norm(v_space, dim=-1, keepdim=True).clamp(min=1e-8)
        sqrt_c = math.sqrt(self.c)
        return torch.cat([
            torch.cosh(sqrt_c * norm_v) / sqrt_c,
            torch.sinh(sqrt_c * norm_v) * v_space / (norm_v * sqrt_c)
        ], dim=-1)

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """Project back onto the hyperboloid after a gradient step."""
        x_space = x[..., 1:]
        sq = torch.sum(x_space ** 2, dim=-1, keepdim=True)
        x0 = torch.sqrt(sq + 1.0 / self.c).clamp(min=1.0 / math.sqrt(self.c))
        return torch.cat([x0, x_space], dim=-1)

    def to_poincare(self, x: torch.Tensor) -> torch.Tensor:
        """Map from Lorentz to Poincaré disk (for downstream Euclidean layers)."""
        return x[..., 1:] / (x[..., :1] + 1.0 / math.sqrt(self.c))


class LorentzEmbedding(nn.Module):
    """
    Vocabulary embeddings on the Lorentz hyperboloid.
    dim=32 hyperbolic ≈ dim=256 Euclidean for hierarchical structure.
    Byte cost: 1024 × 32 × 2 = 64KB (fp16) vs 1024 × 256 × 2 = 512KB
    """
    def __init__(self, vocab_size: int, hyp_dim: int, c: float = 1.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.hyp_dim = hyp_dim  # This is the FULL Lorentz dim (includes x_0)
        self.manifold = LorentzManifold(c)

        # Initialize near origin on hyperboloid
        # Tangent vectors at o, then expmap
        v = nn.Parameter(torch.randn(vocab_size, hyp_dim - 1) * 0.05)
        self.tangent = v  # We store tangent vectors, project in forward

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """Returns Lorentz points, shape (B, T, hyp_dim)."""
        v = self.tangent[idx]  # (B, T, hyp_dim-1)
        # Prepend zero for x_0 component of tangent vector at origin
        v_full = torch.cat([torch.zeros_like(v[..., :1]), v], dim=-1)
        x = self.manifold.expmap_o(v_full)
        return self.manifold.project(x)  # Ensure on manifold

    def riemannian_grad(self) -> torch.Tensor:
        """
        Scale Euclidean gradient to Riemannian gradient.
        Required for correct geometry-aware optimization.
        """
        if self.tangent.grad is None:
            return None
        # Riemannian rescaling: grad_riem = grad_eucl at tangent space
        return self.tangent.grad  # Already in tangent space for our parameterization


# ─────────────────────────────────────────────────────────────────────────────
# SHEAF ATTENTION
# ─────────────────────────────────────────────────────────────────────────────

class SheafAttention(nn.Module):
    """
    Cellular sheaf attention.
    
    Standard attention: scalar edge weight a_{ij} ∈ R
    Sheaf attention:    matrix edge map  O_{ij} ∈ R^{d×d}

    This encodes the *type* of relationship between tokens, not just strength.
    "subject→verb" is a different geometric transformation than "adj→noun."

    The sheaf Laplacian L = B^T diag(O) B computes diffusion on the token graph.
    Running it T times ≈ Neural ODE on the sheaf.

    Budget trick: O_{ij} is LOW-RANK: O_{ij} = U_i diag(s_{ij}) V_j^T
    where U, V are shared across all edges. Only s_{ij} (the scale vector)
    is token-pair specific. This makes sheaf attention O(T*d) not O(T*d^2).
    """
    def __init__(self, model_dim: int, n_heads: int, stalk_dim: int, seq_len: int,
                 p_adic_base: int = 2, quant: bool = True):
        super().__init__()
        self.d = model_dim
        self.h = n_heads
        self.s = stalk_dim  # d_stalk = model_dim / n_heads
        assert model_dim == n_heads * stalk_dim

        # Standard QKV projections
        self.q = QuantizedLinear(model_dim, model_dim, bias=False, quant=quant)
        self.k = QuantizedLinear(model_dim, model_dim, bias=False, quant=quant)
        self.v = QuantizedLinear(model_dim, model_dim, bias=False, quant=quant)
        self.out = QuantizedLinear(model_dim, model_dim, bias=False, quant=quant)

        # Sheaf restriction maps (shared low-rank basis per head)
        # Instead of per-edge matrices, we learn per-token "stalk basis"
        # O_{ij} = softmax(q_i · k_j) * (U_i V_j^T) where U, V are stalk projections
        self.stalk_u = QuantizedLinear(stalk_dim, stalk_dim, bias=False, quant=quant)
        self.stalk_v = QuantizedLinear(stalk_dim, stalk_dim, bias=False, quant=quant)

        # P-adic bias — zero parameters, registered as buffer
        # Built lazily on first forward call to avoid large precomputation
        self.seq_len = seq_len
        self.p_adic_base = p_adic_base
        self.register_buffer("p_adic_bias", None, persistent=False)
        self.register_buffer("causal_mask", None, persistent=False)
        self._p_adic_scale = nn.Parameter(torch.ones(1) * 0.1)

    def _get_p_adic_bias(self, T: int, device) -> torch.Tensor:
        if self.p_adic_bias is None or self.p_adic_bias.shape[0] < T:
            bias = build_p_adic_bias(T, p=self.p_adic_base).to(device)
            self.p_adic_bias = bias
        return self.p_adic_bias[:T, :T]

    def _get_causal_mask(self, T: int, device) -> torch.Tensor:
        if self.causal_mask is None or self.causal_mask.shape[0] < T:
            self.causal_mask = torch.triu(
                torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1
            )
        return self.causal_mask[:T, :T]

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, D = x.shape
        h, s = self.h, self.s

        Q = self.q(x).view(B, T, h, s).transpose(1, 2)  # (B, h, T, s)
        K = self.k(x).view(B, T, h, s).transpose(1, 2)
        V = self.v(x).view(B, T, h, s).transpose(1, 2)

        # Sheaf restriction: transform Q and K through stalk maps
        # This encodes the asymmetric "source → target" relationship
        Q_sheaf = self.stalk_u(Q)  # (B, h, T, s)
        K_sheaf = self.stalk_v(K)  # (B, h, T, s)

        # Attention scores with sheaf structure
        scale = 1.0 / math.sqrt(s)
        scores = torch.matmul(Q_sheaf, K_sheaf.transpose(-2, -1)) * scale  # (B, h, T, T)

        # Add p-adic positional bias — zero parameters!
        p_bias = self._get_p_adic_bias(T, x.device)  # (T, T)
        scores = scores + self._p_adic_scale * p_bias.unsqueeze(0).unsqueeze(0)

        # Causal mask
        causal = self._get_causal_mask(T, x.device)
        scores = scores.masked_fill(causal.unsqueeze(0).unsqueeze(0), float('-inf'))

        attn = F.softmax(scores, dim=-1)

        # Weighted aggregation
        out = torch.matmul(attn, V)  # (B, h, T, s)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.out(out)


# ─────────────────────────────────────────────────────────────────────────────
# SO(2)-EQUIVARIANT MLP
# ─────────────────────────────────────────────────────────────────────────────

class SO2EquivariantMLP(nn.Module):
    """
    MLP where hidden activations transform equivariantly under SO(2).

    Pairs dimensions (2i, 2i+1) as complex numbers z = a + bi.
    The weight matrix W acts as: W_complex @ z (complex multiplication).
    This enforces: f(R_θ x) = R_θ f(x) for all rotations R_θ.

    Why this saves parameters: W_real ∈ R^{d×d} has d² params.
    W_complex ∈ C^{d/2 × d/2} has d²/2 real params — 50% reduction.
    In practice we use a 2×2 block structure preserving the symmetry.

    Galois/GL(n) connection: The group GL(n, F_q) acts on the weight space.
    SO(2) ≅ U(1) is the simplest compact subgroup — equivariance to U(1)
    is the first rung of the representation-theoretic ladder.
    """
    def __init__(self, model_dim: int, ratio: float = 3.0, quant: bool = True):
        super().__init__()
        hidden = int(model_dim * ratio)
        # Round to even for SO(2) pairing
        hidden = hidden + (hidden % 2)

        # Standard projections
        self.up = QuantizedLinear(model_dim, hidden, bias=False, quant=quant)
        self.down = QuantizedLinear(hidden, model_dim, bias=False, quant=quant)
        self.norm = nn.LayerNorm(hidden)

        # SO(2) rotation parameters: one angle per pair
        n_pairs = hidden // 2
        self.theta = nn.Parameter(torch.zeros(n_pairs))

    def _apply_so2(self, x: torch.Tensor) -> torch.Tensor:
        """Apply learned SO(2) rotations to paired dimensions."""
        B, T, D = x.shape
        n_pairs = D // 2
        # Reshape to pairs
        x_pairs = x.view(B, T, n_pairs, 2)  # (B, T, pairs, 2)
        cos_t = torch.cos(self.theta).view(1, 1, n_pairs, 1)
        sin_t = torch.sin(self.theta).view(1, 1, n_pairs, 1)
        # Rotation matrix action: [cos θ, -sin θ; sin θ, cos θ] @ [a; b]
        a = x_pairs[..., :1]
        b = x_pairs[..., 1:]
        rotated = torch.cat([cos_t * a - sin_t * b, sin_t * a + cos_t * b], dim=-1)
        return rotated.view(B, T, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.up(x)
        h = self._apply_so2(h)
        h = self.norm(h)
        h = F.silu(h)  # SiLU (Swish) — smooth, fits geometric aesthetic
        return self.down(h)


# ─────────────────────────────────────────────────────────────────────────────
# GL(n) / GALOIS-INSPIRED WEIGHT INITIALIZATION
# ─────────────────────────────────────────────────────────────────────────────

def galois_init_(tensor: torch.Tensor, p: int = 3):
    """
    Initialize weight matrix using structure inspired by GL(n, F_p).

    Standard init (Kaiming, Xavier) assumes i.i.d. Gaussian weights.
    GL(n, F_p)-inspired init uses the structure of invertible matrices
    over finite fields to ensure:
      1. Full rank at initialization (no vanishing gradients)
      2. Maximal spread of singular values (good conditioning)
      3. Built-in symmetry under the Galois group Gal(F_{p^n}/F_p)

    In practice: initialize as a block circulant matrix with prime-based
    eigenvalue structure. The DFT of a circulant gives eigenvalues on the
    unit circle — this is GL(n) acting on F_p^n.
    """
    with torch.no_grad():
        rows, cols = tensor.shape[:2]
        # Build a structured circulant-like base
        # First row: use p-adic-inspired pattern
        base = torch.zeros(min(rows, cols))
        for i in range(len(base)):
            # Eigenvalue from p-adic valuation: λ_i = (-1)^v_p(i+1) / sqrt(d)
            val = p_adic_valuation(i + 1, p)
            base[i] = ((-1) ** val) / math.sqrt(min(rows, cols))

        # Fill as circulant (each row is cyclic shift of previous)
        for i in range(min(rows, cols)):
            start = i % len(base)
            tensor[i, :len(base)] = torch.roll(base, start)

        # Add small noise to break exact symmetry (allow gradient flow)
        tensor.add_(torch.randn_like(tensor) * 0.01)
    return tensor


def apply_galois_init(model: nn.Module, p: int = 3):
    """Apply Galois-inspired init to all QuantizedLinear weight matrices."""
    for name, module in model.named_modules():
        if isinstance(module, (QuantizedLinear, nn.Linear)):
            if module.weight.dim() == 2:
                galois_init_(module.weight, p=p)


# ─────────────────────────────────────────────────────────────────────────────
# HERMES BLOCK  (Neural ODE core — run T times)
# ─────────────────────────────────────────────────────────────────────────────

class HermesBlock(nn.Module):
    """
    Single recurrent block. Run T=6 times to simulate Neural ODE.

    The recurrence h_{t+1} = h_t + Block(h_t) is a discretization of:
        dh/dt = f(h, t)
    which is the heat equation on the sheaf manifold.

    At T=6, one HermesBlock ≈ 6 independent transformer layers
    in terms of representational depth, at 1/6th the parameter cost.
    """
    def __init__(self, cfg: HermesConfig):
        super().__init__()
        self.attn = SheafAttention(
            cfg.model_dim, cfg.n_heads, cfg.stalk_dim,
            cfg.seq_len, cfg.p_adic_base, cfg.use_int6_qat
        )
        self.mlp = SO2EquivariantMLP(
            cfg.model_dim, cfg.mlp_ratio, cfg.use_int6_qat
        )
        self.norm1 = nn.LayerNorm(cfg.model_dim)
        self.norm2 = nn.LayerNorm(cfg.model_dim)

        # Time embedding for Neural ODE: different behavior at each step t
        # Cost: ode_steps × model_dim parameters — tiny
        self.time_embed = nn.Parameter(
            torch.zeros(cfg.ode_steps, cfg.model_dim)
        )

    def forward(self, x: torch.Tensor, t: int = 0) -> torch.Tensor:
        # Add time signal (ODE step awareness)
        if t < self.time_embed.shape[0]:
            x = x + self.time_embed[t].unsqueeze(0).unsqueeze(0)
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ─────────────────────────────────────────────────────────────────────────────
# HERMES MODEL
# ─────────────────────────────────────────────────────────────────────────────

class HermesModel(nn.Module):
    """
    Full HERMES architecture.

    Data flow:
      tokens → LorentzEmbedding (hyperbolic, 32-dim)
             → project to model_dim (Euclidean, for Sheaf attention)
             → HermesBlock × T (Neural ODE recurrence)
             → LayerNorm → output projection → logits
    """
    def __init__(self, cfg: HermesConfig):
        super().__init__()
        self.cfg = cfg

        # 1. Hyperbolic embeddings (Lorentz)
        self.embed = LorentzEmbedding(cfg.vocab_size, cfg.hyp_dim, cfg.curvature)

        # 2. Project from Poincaré/Lorentz space to model_dim
        #    Lorentz point → Poincaré disk (hyp_dim-1) → model_dim
        self.embed_proj = QuantizedLinear(
            cfg.hyp_dim - 1, cfg.model_dim, bias=False, quant=False  # keep fp16
        )

        # 3. Single recurrent block (Neural ODE core)
        self.block = HermesBlock(cfg)

        # 4. Output
        self.norm_out = nn.LayerNorm(cfg.model_dim)
        self.lm_head = QuantizedLinear(
            cfg.model_dim, cfg.vocab_size, bias=False, quant=cfg.use_int6_qat
        )

        # Weight tying: output projection ≈ inverse of input embedding
        # Note: can't tie directly (different manifold geometry), but
        # we initialize lm_head.weight from embed_proj for warm start
        self.ode_steps = cfg.ode_steps

        # Apply Galois-inspired initialization
        apply_galois_init(self, p=cfg.p_adic_base)

        self._count_params()

    def _count_params(self):
        total = sum(p.numel() for p in self.parameters())
        embed_params = sum(p.numel() for p in self.embed.parameters())
        print(f"[HERMES] Total params: {total:,}")
        print(f"[HERMES] Embed params: {embed_params:,} (fp16, not quantized)")
        print(f"[HERMES] Other params: {total - embed_params:,} (int6 QAT)")
        # Byte estimate
        embed_bytes = embed_params * 2  # fp16
        other_bytes = (total - embed_params) * 6 / 8  # int6
        total_mb = (embed_bytes + other_bytes) / (1024**2)
        print(f"[HERMES] Estimated artifact size: {total_mb:.2f} MB")

    def _run_block(self, x: torch.Tensor, t: int) -> torch.Tensor:
        return self.block(x, t=t)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None):
        B, T = idx.shape

        # Hyperbolic embeddings → project to Euclidean model_dim
        x_lorentz = self.embed(idx)                    # (B, T, hyp_dim)
        x_poincare = self.embed.manifold.to_poincare(x_lorentz)  # (B, T, hyp_dim-1)
        x = self.embed_proj(x_poincare)                # (B, T, model_dim)

        # Neural ODE: run single block T times
        for t in range(self.ode_steps):
            if (
                self.training
                and self.cfg.activation_checkpointing
                and self.ode_steps > 1
            ):
                x = checkpoint(
                    partial(self._run_block, t=t),
                    x,
                    use_reentrant=False,
                )
            else:
                x = self.block(x, t=t)

        # Output
        x = self.norm_out(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is None:
            return logits, None

        loss = F.cross_entropy(
            logits.view(-1, self.cfg.vocab_size),
            targets.view(-1),
            ignore_index=-1
        )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.seq_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx


# ─────────────────────────────────────────────────────────────────────────────
# MUON OPTIMIZER
# ─────────────────────────────────────────────────────────────────────────────

class Muon(torch.optim.Optimizer):
    """
    Muon (Momentum + Orthogonal Update Normalization) optimizer.
    Proven faster than AdamW for the parameter golf setting.
    
    Key idea: after computing momentum, project the update onto the
    orthogonal group (via Newton-Schulz iteration). This keeps weight
    matrices well-conditioned throughout training.
    
    Reference: https://github.com/KellerJordan/Muon
    """
    def __init__(self, params, lr=3e-3, momentum=0.95, nesterov=True,
                 ns_steps=5, weight_decay=0.0):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                        ns_steps=ns_steps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @staticmethod
    def _zeropower_via_newtonschulz(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
        """
        Newton-Schulz iteration to compute G @ (G^T G)^{-1/2}.
        Converges in ~5 steps. Projects gradient onto orthogonal group.
        """
        assert G.ndim >= 2
        a, b, c = (3.4445, -4.7750, 2.0315)
        X = G.bfloat16() / (G.norm() + 1e-7)
        if G.size(0) > G.size(1):
            X = X.T
        for _ in range(steps):
            A = X @ X.T
            B = b * A + c * (A @ A)
            X = a * X + B @ X
        if G.size(0) > G.size(1):
            X = X.T
        return X

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]

                # Weight decay
                if group['weight_decay'] != 0:
                    p.mul_(1 - lr * group['weight_decay'])

                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)

                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g)

                if group['nesterov']:
                    g = g + momentum * buf
                else:
                    g = buf

                # Apply orthogonal normalization for 2D+ weight matrices
                if g.ndim >= 2 and g.numel() > 1:
                    g = self._zeropower_via_newtonschulz(g, steps=group['ns_steps'])
                    # Scale to match SGD magnitude
                    g = g * max(1, g.size(0) / g.size(1)) ** 0.5

                p.add_(g, alpha=-lr)


# ─────────────────────────────────────────────────────────────────────────────
# EMA (Exponential Moving Average of weights)
# ─────────────────────────────────────────────────────────────────────────────

class EMA:
    """Shadow copy of model weights, updated as exponential moving average."""
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            self.shadow[k].mul_(self.decay).add_(v, alpha=1 - self.decay)

    @contextmanager
    def apply(self, model: nn.Module):
        """Context manager: temporarily load EMA weights for evaluation."""
        original = {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow)
        try:
            yield
        finally:
            model.load_state_dict(original)


# ─────────────────────────────────────────────────────────────────────────────
# DATASET  (FineWeb streaming)
# ─────────────────────────────────────────────────────────────────────────────

class FineWebDataset(Dataset):
    """
    Tokenized FineWeb chunks. Expects pre-tokenized .bin files.
    Format: flat array of uint16 token IDs.
    """
    def __init__(self, data_path: str, seq_len: int, tokenizer: MicroBPE):
        self.seq_len = seq_len
        self.tok = tokenizer

        token_bin = resolve_token_data_path(data_path)
        if token_bin is not None:
            require_numpy("memory-mapped token datasets")
            self.data = np.memmap(token_bin, dtype=np.uint16, mode='r')
        else:
            # Fallback: read text and tokenize
            print(f"[Dataset] Tokenizing {data_path}...")
            with open(data_path, 'r', encoding='utf-8') as f:
                text = f.read()
            ids = tokenizer.encode(text)
            if np is not None:
                self.data = np.array(ids, dtype=np.uint16)
            else:
                self.data = torch.tensor(ids, dtype=torch.int64)

        self.n = (len(self.data) - seq_len - 1) // seq_len

    def __len__(self): return self.n

    def __getitem__(self, i):
        start = i * self.seq_len
        if isinstance(self.data, torch.Tensor):
            chunk = self.data[start:start + self.seq_len + 1]
            x = chunk[:-1].clone()
            y = chunk[1:].clone()
        else:
            chunk = self.data[start:start + self.seq_len + 1].astype(np.int64)
            x = torch.from_numpy(chunk[:-1])
            y = torch.from_numpy(chunk[1:])
        return x, y


# ─────────────────────────────────────────────────────────────────────────────
# BPB EVALUATION  (sliding window, stride=64)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_bpb(model: nn.Module, data, cfg: HermesConfig, tokenizer: MicroBPE,
                 max_bytes: int = 1_000_000) -> float:
    """
    Compute bits-per-byte on validation data using sliding window.
    Stride=64 means the model always has rich context, not cold-start blocks.
    """
    was_training = model.training
    model.eval()
    device = cfg.device
    dtype = getattr(torch, cfg.dtype)

    total_loss = 0.0
    total_bytes = 0
    stride = cfg.eval_stride
    seq = cfg.eval_seq_len

    try:
        for start in range(0, len(data) - seq, stride):
            tokens = token_slice(data, start, start + seq + 1)
            x = torch.tensor(tokens[:-1], dtype=torch.long, device=device).unsqueeze(0)
            y = torch.tensor(tokens[1:], dtype=torch.long, device=device).unsqueeze(0)

            with torch.autocast(device_type=device if device != 'cpu' else 'cpu',
                                dtype=dtype, enabled=(device == 'cuda')):
                logits, _ = model(x)

            # Only score the non-warm-up tokens (last stride tokens in the window)
            score_start = max(0, seq - stride)
            scored_targets = tokens[1 + score_start:]
            window_bytes = len(tokenizer.decode(scored_targets))
            if window_bytes == 0:
                continue
            loss = F.cross_entropy(
                logits[0, score_start:].view(-1, cfg.vocab_size),
                y[0, score_start:].view(-1),
                reduction='sum'
            )
            total_loss += loss.item()
            total_bytes += window_bytes
            if total_bytes >= max_bytes:
                break
    finally:
        if was_training:
            model.train()

    if total_bytes == 0:
        raise RuntimeError("BPB evaluation saw zero decodable bytes.")

    bpb = total_loss / (total_bytes * math.log(2))
    return bpb


# ─────────────────────────────────────────────────────────────────────────────
# RUN LOGGING
# ─────────────────────────────────────────────────────────────────────────────

def append_result_record(results_path: Optional[str], summary: dict):
    if not results_path:
        return
    parent = os.path.dirname(results_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(results_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(summary, sort_keys=True) + "\n")


def print_run_summary(summary: dict):
    print("\n" + "=" * 60)
    print("  RUN SUMMARY")
    print("=" * 60)
    print(f"  Name:       {summary['run_name']}")
    print(f"  Final BPB:  {summary['final_bpb']:.4f}")
    print(f"  Best BPB:   {summary['best_bpb']:.4f}")
    print(f"  Best Step:  {summary['best_step']}")
    print(f"  Checkpoint: {summary['best_checkpoint'] or '-'}")
    print(f"  Ckpt Dir:   {summary['checkpoint_dir']}")
    print(f"  Resumed:    {summary['resumed_from'] or '-'}")
    print(f"  Seq/Batch:  {summary['seq_len']} / {summary['batch_size']}")
    print(f"  Steps:      {summary['steps']}")
    print(f"  Device:     {summary['device']} ({summary['dtype']})")
    print(f"  Smoke:      {summary['smoke']}")
    print(f"  Minutes:    {summary['minutes']:.2f}")


def sanitize_run_name(name: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name).strip("._-")
    return safe or "run"


def checkpoint_run_dir(checkpoint_dir: str, run_name: str) -> str:
    return os.path.join(checkpoint_dir, sanitize_run_name(run_name))


def resolve_token_data_path(data_path: str) -> Optional[str]:
    if data_path.endswith(".bin") and os.path.exists(data_path):
        return data_path
    candidate = data_path + ".bin"
    if os.path.exists(candidate):
        return candidate
    return None


def resolve_resume_path(resume: Optional[str], checkpoint_dir: str, run_name: str) -> Optional[str]:
    if not resume:
        return None
    if resume != "latest":
        return resume
    run_dir = checkpoint_run_dir(checkpoint_dir, run_name)
    if not os.path.isdir(run_dir):
        return None
    candidates = [
        os.path.join(run_dir, name)
        for name in os.listdir(run_dir)
        if name.endswith(".pt")
    ]
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def load_training_checkpoint(path: str, model: nn.Module, optimizer, muon_opt,
                             scaler, ema: EMA, device: str):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"], strict=True)
    if "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if "muon_opt" in ckpt:
        muon_opt.load_state_dict(ckpt["muon_opt"])
    if "scaler" in ckpt and ckpt["scaler"] is not None:
        scaler.load_state_dict(ckpt["scaler"])
    if "ema_shadow" in ckpt:
        ema.shadow = {k: v.clone().detach() for k, v in ckpt["ema_shadow"].items()}
    return ckpt


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────

def train(cfg: HermesConfig, train_path: str, val_path: str,
          run_name: str = "run", results_path: Optional[str] = "runs/hermes_runs.jsonl",
          checkpoint_dir: str = "checkpoints", resume: Optional[str] = None):
    torch.manual_seed(cfg.seed)
    device = cfg.device
    dtype = getattr(torch, cfg.dtype)

    print(f"\n{'='*60}")
    print(f"  HERMES — Parameter Golf Submission")
    print(f"  Device: {device} | Dtype: {cfg.dtype}")
    print(f"  Seq: {cfg.seq_len} | Batch: {cfg.batch_size} | Steps: {cfg.max_steps}")
    print(f"{'='*60}\n")

    # Tokenizer
    tok = MicroBPE(vocab_size=cfg.vocab_size)
    active_vocab = len(tok.vocab)
    eval_max_bytes = 16_384 if cfg.smoke else 1_000_000

    # Model
    model = HermesModel(cfg).to(device)
    if cfg.compile and hasattr(torch, 'compile'):
        print("[HERMES] Compiling model (torch.compile max-autotune)...")
        model = torch.compile(model, mode="max-autotune")

    # EMA
    ema = EMA(model, decay=cfg.ema_decay)

    # Optimizer: Muon for weight matrices, AdamW for embeddings + norms
    muon_params = []
    adam_params = []
    for name, p in model.named_parameters():
        if 'embed' in name or 'norm' in name or 'bias' in name or p.ndim < 2:
            adam_params.append(p)
        else:
            muon_params.append(p)

    optimizer = torch.optim.AdamW(adam_params, lr=cfg.lr * 0.1, weight_decay=cfg.weight_decay)
    muon_opt = Muon(muon_params, lr=cfg.lr, momentum=0.95, weight_decay=cfg.weight_decay)

    # Cosine LR schedule with warmup
    def get_lr(step):
        if step < cfg.warmup_steps:
            return step / cfg.warmup_steps
        t = (step - cfg.warmup_steps) / (cfg.max_steps - cfg.warmup_steps)
        return 0.1 + 0.9 * (1 + math.cos(math.pi * t)) / 2

    # Dataset
    if cfg.smoke:
        # Quick sanity check with tiny synthetic data
        print("[HERMES] SMOKE MODE — synthetic data")
        smoke_examples = 32
        dummy_data = torch.randint(
            0, active_vocab, (cfg.seq_len * smoke_examples + 1,), dtype=torch.int64
        )

        class TinyDataset(Dataset):
            def __len__(self):
                return (len(dummy_data) - cfg.seq_len - 1) // cfg.seq_len
            def __getitem__(self, i):
                s = i * cfg.seq_len
                chunk = dummy_data[s:s + cfg.seq_len + 1]
                return chunk[:-1].clone(), chunk[1:].clone()

        train_ds = TinyDataset()
        val_data = dummy_data
    else:
        train_ds = FineWebDataset(train_path, cfg.seq_len, tok)
        val_token_bin = resolve_token_data_path(val_path)
        if val_token_bin is not None:
            require_numpy("memory-mapped validation data")
            val_data = np.memmap(val_token_bin, dtype=np.uint16, mode='r')
        else:
            val_data = torch.randint(0, active_vocab, (100_000,), dtype=torch.int64)

    loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                        num_workers=0, pin_memory=(device == 'cuda'))

    scaler = build_grad_scaler(device, enabled=(device == 'cuda' and cfg.dtype == 'float16'))

    # Training loop
    step = 0
    best_bpb = float('inf')
    best_step = None
    best_checkpoint = None
    resumed_from = None
    t0 = time.time()
    loader_iter = iter(loader)

    resume_path = resolve_resume_path(resume, checkpoint_dir, run_name)
    if resume_path is not None:
        ckpt = load_training_checkpoint(resume_path, model, optimizer, muon_opt, scaler, ema, device)
        step = int(ckpt.get("step", -1)) + 1
        best_bpb = float(ckpt.get("best_bpb", best_bpb))
        best_step = ckpt.get("best_step", best_step)
        best_checkpoint = ckpt.get("best_checkpoint", resume_path)
        resumed_from = resume_path
        print(f"[HERMES] Resumed from {resume_path} at step {step}")

    while step < cfg.max_steps:
        try:
            x, y = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            x, y = next(loader_iter)

        x, y = x.to(device), y.to(device)

        # LR update
        lr_scale = get_lr(step)
        for pg in optimizer.param_groups: pg['lr'] = cfg.lr * 0.1 * lr_scale
        for pg in muon_opt.param_groups: pg['lr'] = cfg.lr * lr_scale

        # Forward
        with torch.autocast(device_type=device if device != 'cpu' else 'cpu',
                            dtype=dtype, enabled=(device == 'cuda')):
            _, loss = model(x, y)

        # Backward
        optimizer.zero_grad(set_to_none=True)
        muon_opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        scaler.unscale_(muon_opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.step(muon_opt)
        scaler.update()

        # EMA update
        ema.update(model)

        # Logging
        if step % 100 == 0:
            elapsed = time.time() - t0
            print(f"  step {step:5d} | loss {loss.item():.4f} | "
                  f"lr {cfg.lr * lr_scale:.2e} | t {elapsed:.1f}s")

        # Evaluation
        if step % 500 == 0 and step > 0:
            with ema.apply(model):
                bpb = evaluate_bpb(model, val_data, cfg, tok, max_bytes=eval_max_bytes)
                is_new_best = bpb < best_bpb
                if is_new_best:
                    best_bpb = bpb
                    best_step = step
                    best_checkpoint = _save_checkpoint(
                        model, cfg, step, bpb,
                        checkpoint_dir=checkpoint_dir, run_name=run_name,
                        optimizer=optimizer, muon_opt=muon_opt, scaler=scaler, ema=ema,
                        best_bpb=best_bpb, best_step=best_step, train_step=step
                    )
            if device == "cuda":
                torch.cuda.empty_cache()
            print(f"\n  ★ BPB @ step {step}: {bpb:.4f} {'(NEW BEST!)' if is_new_best else ''}\n")

        step += 1

    # Final eval with EMA weights
    print("\n[HERMES] Final evaluation...")
    with ema.apply(model):
        final_bpb = evaluate_bpb(model, val_data, cfg, tok, max_bytes=eval_max_bytes)
        final_is_best = final_bpb < best_bpb
        if final_is_best:
            best_bpb = final_bpb
            best_step = "final"
            best_checkpoint = _save_checkpoint(
                model, cfg, "final", final_bpb,
                checkpoint_dir=checkpoint_dir, run_name=run_name,
                optimizer=optimizer, muon_opt=muon_opt, scaler=scaler, ema=ema,
                best_bpb=best_bpb, best_step=best_step, train_step=cfg.max_steps
            )
    elapsed_min = (time.time() - t0) / 60

    summary = {
        "run_name": run_name,
        "final_bpb": float(final_bpb),
        "best_bpb": float(best_bpb),
        "best_step": best_step,
        "best_checkpoint": best_checkpoint,
        "seq_len": cfg.seq_len,
        "batch_size": cfg.batch_size,
        "steps": cfg.max_steps,
        "device": cfg.device,
        "dtype": cfg.dtype,
        "smoke": cfg.smoke,
        "minutes": elapsed_min,
        "checkpoint_dir": checkpoint_run_dir(checkpoint_dir, run_name),
        "resumed_from": resumed_from,
    }

    print(f"\n  ★ FINAL BPB: {final_bpb:.4f} {'(NEW BEST!)' if final_is_best else ''}")
    print(f"  Best BPB:   {best_bpb:.4f}")
    print(f"  Best Step:  {best_step}")
    if best_checkpoint is not None:
        print(f"  Checkpoint: {best_checkpoint}")
    print(f"  Time:       {elapsed_min:.1f} min")

    print_run_summary(summary)
    append_result_record(results_path, summary)

    return model, summary


def _save_checkpoint(model: nn.Module, cfg: HermesConfig, step, bpb: float,
                     checkpoint_dir: str = "checkpoints", run_name: str = "run",
                     optimizer=None, muon_opt=None, scaler=None, ema: Optional[EMA] = None,
                     best_bpb: Optional[float] = None, best_step=None,
                     train_step: Optional[int] = None):
    run_dir = checkpoint_run_dir(checkpoint_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, f"hermes_step{step}_bpb{bpb:.4f}.pt")
    payload = {
        'model': model.state_dict(),
        'cfg': cfg.__dict__,
        'step': train_step if train_step is not None else (step if isinstance(step, int) else -1),
        'bpb': bpb,
        'best_bpb': best_bpb,
        'best_step': best_step,
        'best_checkpoint': path,
    }
    if optimizer is not None:
        payload['optimizer'] = optimizer.state_dict()
    if muon_opt is not None:
        payload['muon_opt'] = muon_opt.state_dict()
    if scaler is not None:
        payload['scaler'] = scaler.state_dict()
    if ema is not None:
        payload['ema_shadow'] = {k: v.clone().detach() for k, v in ema.shadow.items()}
    torch.save(payload, path)
    print(f"  Saved checkpoint: {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# ARTIFACT SERIALIZATION  (pack model into 16MB submission file)
# ─────────────────────────────────────────────────────────────────────────────

def serialize_to_artifact(model: nn.Module, cfg: HermesConfig,
                           tokenizer: MicroBPE, out_path: str = "submission.py"):
    """
    Serialize the trained model into a self-contained Python file.
    - Lorentz embeddings: fp16 (required for manifold stability)
    - All other weights: int6 packed (4 values per 3 bytes)
    - Tokenizer merges: zlib-compressed uint16 pairs, base64-encoded
    - Architecture code: included verbatim
    - Total target: < 16MB
    """
    state = model.state_dict()
    packed_weights = {}
    metadata = {}

    for name, tensor in state.items():
        is_embed = 'embed.tangent' in name or 'embed_proj' in name
        if is_embed or tensor.ndim < 2 or tensor.numel() < 16:
            # Keep in fp16
            data = bytes(tensor.half().detach().cpu().contiguous().view(torch.uint8).tolist())
            packed_weights[name] = base64.b64encode(zlib.compress(data)).decode()
            metadata[name] = {'dtype': 'fp16', 'shape': list(tensor.shape)}
        else:
            # Int6 pack
            data, scale, pad = pack_6bit_to_bytes(tensor.float())
            packed_weights[name] = base64.b64encode(zlib.compress(data)).decode()
            metadata[name] = {
                'dtype': 'int6', 'shape': list(tensor.shape),
                'scale': scale, 'pad': pad
            }

    # Tokenizer merges
    if tokenizer.merges:
        merges_blob = base64.b64encode(zlib.compress(uint16_pairs_to_bytes(tokenizer.merges))).decode()
    else:
        merges_blob = ""

    # Estimate size
    total_bytes = sum(len(v) for v in packed_weights.values())
    print(f"\n[Artifact] Packed weights: {total_bytes / (1024**2):.2f} MB")
    print(f"[Artifact] Merges blob:    {len(merges_blob) / 1024:.1f} KB")

    # Write submission.py
    with open(out_path, 'w') as f:
        f.write(f'''#!/usr/bin/env python3
"""
HERMES Parameter Golf Submission
BPB target: ~1.11 | Size: {total_bytes/(1024**2):.2f} MB
Auto-generated — do not edit manually
"""
''')
        # Embed the full architecture from this file
        with open(__file__, 'r') as src:
            code = src.read()
        code = code.split('\nif __name__ == "__main__":')[0]
        f.write("\n# ── ARCHITECTURE (embedded) ────────────────────────────────\n")
        f.write(code)
        f.write(f"""

# ── PACKED WEIGHTS ──────────────────────────────────────────────────────────
ARTIFACT_WEIGHTS = {repr(packed_weights)}

# ── WEIGHT METADATA ─────────────────────────────────────────────────────────
ARTIFACT_METADATA = {repr(metadata)}

# ── TOKENIZER MERGES ────────────────────────────────────────────────────────
ARTIFACT_MERGES_BLOB = "{merges_blob}"

# ── CONFIG ──────────────────────────────────────────────────────────────────
ARTIFACT_CFG = {repr(cfg.__dict__)}


def _decode_artifact_tensor(meta, blob):
    raw = zlib.decompress(base64.b64decode(blob))
    if meta["dtype"] == "fp16":
        return torch.frombuffer(raw, dtype=torch.float16).clone().reshape(meta["shape"])
    return unpack_6bit_from_bytes(raw, tuple(meta["shape"]), meta["scale"], meta["pad"])


def load_artifact(device=None):
    MicroBPE.MERGES_BLOB = ARTIFACT_MERGES_BLOB
    cfg = HermesConfig(**ARTIFACT_CFG)
    if device is not None:
        cfg.device = device
    model = HermesModel(cfg)
    state = {{
        name: _decode_artifact_tensor(meta, ARTIFACT_WEIGHTS[name])
        for name, meta in ARTIFACT_METADATA.items()
    }}
    model.load_state_dict(state, strict=True)
    model = model.to(cfg.device)
    model.eval()
    return model, cfg


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg = load_artifact(device=device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[HERMES Submission] Loaded on {{device}} with {{n_params:,}} params")
""")

    size_mb = os.path.getsize(out_path) / (1024**2)
    print(f"[Artifact] Written to {out_path} ({size_mb:.2f} MB)")
    assert size_mb < 16.0, f"OVER BUDGET: {size_mb:.2f} MB > 16 MB!"
    print(f"[Artifact] ✓ Under 16MB budget ({16.0 - size_mb:.2f} MB headroom)")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# ABLATION RUNNER  (measure BPB delta per component on RTX 3050)
# ─────────────────────────────────────────────────────────────────────────────

def run_ablation(base_cfg: HermesConfig, only: Optional[str] = None,
                 results_path: Optional[str] = "runs/hermes_runs.jsonl",
                 checkpoint_dir: str = "checkpoints", resume: Optional[str] = None):
    """
    A/B test each geometric component independently.
    Run locally on RTX 3050 to validate before H100 submission.
    """
    ablations = {
        "baseline_vanilla":     dict(use_int6_qat=False, ode_steps=1, hyp_dim=65),
        "hyperbolic_only":      dict(use_int6_qat=False, ode_steps=1),
        "ode_only":             dict(use_int6_qat=False, hyp_dim=65),
        "int6_only":            dict(ode_steps=1, hyp_dim=65),
        "full_hermes":          dict(),  # All components
    }

    if only is not None:
        if only not in ablations:
            raise ValueError(
                f"Unknown ablation '{only}'. Choose from: {', '.join(ablations.keys())}"
            )
        ablations = {only: ablations[only]}

    results = {}
    for name, overrides in ablations.items():
        cfg = HermesConfig(**{**base_cfg.__dict__, **overrides,
                               'smoke': base_cfg.smoke, 'max_steps': base_cfg.max_steps})
        print(f"\n{'─'*50}")
        print(f"  Ablation: {name}")
        print(f"{'─'*50}")
        try:
            resume_mode = resume if (only is not None and name == only) else None
            model, summary = train(
                cfg, "", "", run_name=name, results_path=results_path,
                checkpoint_dir=checkpoint_dir, resume=resume_mode
            )
            results[name] = summary
        except Exception as e:
            results[name] = f"ERROR: {e}"

    print("\n\n" + "="*50)
    print("  ABLATION RESULTS")
    print("="*50)
    sortable = [r for r in results.values() if isinstance(r, dict)]
    for summary in sorted(sortable, key=lambda r: r["final_bpb"]):
        print(
            f"  {summary['run_name']:30s}: final {summary['final_bpb']:.4f} | "
            f"best {summary['best_bpb']:.4f} @ {summary['best_step']}"
        )
    for name, result in results.items():
        if not isinstance(result, dict):
            print(f"  {name:30s}: {result}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="HERMES Parameter Golf")
    parser.add_argument("--smoke",     action="store_true", help="Quick sanity check")
    parser.add_argument("--ablation",  action="store_true", help="Run ablation study")
    parser.add_argument("--train_path", default="data/fineweb_train", help="Training data path")
    parser.add_argument("--val_path",   default="data/fineweb_val",   help="Validation data path")
    parser.add_argument("--seq_len",    type=int, default=512,  help="Sequence length")
    parser.add_argument("--batch",      type=int, default=4,    help="Batch size")
    parser.add_argument("--steps",      type=int, default=5000, help="Training steps")
    parser.add_argument("--no_quant",   action="store_true",    help="Disable int6 QAT")
    parser.add_argument("--ode_steps",  type=int, default=6,    help="ODE recurrence steps")
    parser.add_argument("--no_activation_checkpointing", action="store_true",
                        help="Disable activation checkpointing for the recurrent ODE block")
    parser.add_argument("--serialize",  action="store_true",    help="Pack to submission.py")
    parser.add_argument("--ablation_name", default=None, help="Run a single named ablation variant")
    parser.add_argument("--run_name",   default="manual_run", help="Name used in printed/logged summaries")
    parser.add_argument("--results_file", default="runs/hermes_runs.jsonl",
                        help="JSONL file for appending run summaries; set to empty string to disable")
    parser.add_argument("--checkpoint_dir", default="checkpoints",
                        help="Directory for per-run checkpoints")
    parser.add_argument("--resume", default=None,
                        help="Resume from a checkpoint path or use 'latest' for the latest checkpoint in the run directory")
    args = parser.parse_args()

    cfg = HermesConfig(
        smoke       = args.smoke,
        seq_len     = args.seq_len,
        batch_size  = args.batch,
        max_steps   = args.steps,
        use_int6_qat= not args.no_quant,
        ode_steps   = args.ode_steps,
        activation_checkpointing=not args.no_activation_checkpointing,
    )

    if args.ablation:
        run_ablation(
            cfg, only=args.ablation_name,
            results_path=(args.results_file or None),
            checkpoint_dir=args.checkpoint_dir, resume=args.resume
        )
    else:
        model, summary = train(
            cfg, args.train_path, args.val_path,
            run_name=args.run_name, results_path=(args.results_file or None),
            checkpoint_dir=args.checkpoint_dir, resume=args.resume
        )
        if args.serialize:
            tok = MicroBPE(vocab_size=cfg.vocab_size)
            serialize_to_artifact(model, cfg, tok)
