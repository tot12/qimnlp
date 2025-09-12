# qinet_m_nlp.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------- Quantum Interference Modulator 1D -----------------
class QIM1D(nn.Module):
    """
    1D version of r_{alpha->beta}(theta,t) producing M in [B, 1, L]
    theta: normalized position in [0,1] along sequence length L
    tau:   same grid used as 't' (discrete time indices) for interference phase
    """
    def __init__(self, L_max: int):
        super().__init__()
        self.L_max = L_max

        # Learnable scalars
        self.a     = nn.Parameter(torch.tensor(1.0))
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.k     = nn.Parameter(torch.tensor(4.0))
        self.m     = nn.Parameter(torch.tensor(3.0))
        self.delta = nn.Parameter(torch.linspace(0.1, 0.6, 6))   # δ_i
        self.E     = nn.Parameter(torch.linspace(0.05, 0.30, 6)) # E_i

        # Complex vectors u_alpha, u_beta ∈ C^6
        self.u_alpha_re = nn.Parameter(torch.randn(6) * 0.3)
        self.u_alpha_im = nn.Parameter(torch.randn(6) * 0.3)
        self.u_beta_re  = nn.Parameter(torch.randn(6) * 0.3)
        self.u_beta_im  = nn.Parameter(torch.randn(6) * 0.3)

        # buffers for max length; will slice per real length
        theta = torch.linspace(0.0, 1.0, L_max).view(L_max, 1)  # (L,1)
        tau   = torch.linspace(0.0, 1.0, L_max).view(1, L_max)  # (1,L)
        self.register_buffer("theta_grid", theta)
        self.register_buffer("tau_grid", tau)

    def forward(self, B: int, L: int, device=None):
        theta = self.theta_grid[:L].to(device)  # (L,1)
        tau   = self.tau_grid[:, :L].to(device) # (1,L)

        # base envelope: a * exp(-alpha*theta) * sin(k*theta) * cos(m*theta)
        base = self.a * torch.exp(-self.alpha * theta) * torch.sin(self.k * theta) * torch.cos(self.m * theta)  # (L,1)

        # complex unit vectors
        ua = torch.complex(self.u_alpha_re, self.u_alpha_im)
        ub = torch.complex(self.u_beta_re,  self.u_beta_im)
        ua = ua / (ua.abs().pow(2).sum().sqrt() + 1e-8)
        ub = ub / (ub.abs().pow(2).sum().sqrt() + 1e-8)

        coeffs = ub * torch.conj(ua)  # (6,)
        parts = []
        for i in range(6):
            phase = -1j * (2.0 * self.E[i] * tau + self.delta[i] * theta)  # (L,L)
            parts.append(coeffs[i] * torch.exp(phase))
        S = torch.stack(parts, dim=0).sum(dim=0).abs()  # (L,L)

        # collapse second axis by diagonal extraction (position-wise phase)
        # Using diag elements aligns t with theta index (no approximation of formula; a specific slice)
        diag = torch.diagonal(S, dim1=0, dim2=1)  # (L,)
        r = base.squeeze(1) * diag                # (L,)

        # Normalize to [0.5, 1.5] per-batch (shared across tokens)
        r_min = r.min()
        r_max = r.max()
        r_norm = (r - r_min) / (r_max - r_min + 1e-8)
        r_scaled = 0.5 + r_norm  # (L,)

        return r_scaled.view(1, 1, L).expand(B, 1, L)  # [B,1,L]

# ----------------- Tiny building blocks (causal) -----------------
class CausalDWConv1d(nn.Module):
    def __init__(self, ch, kernel=3, dilation=1):
        super().__init__()
        pad = (kernel - 1) * dilation  # left padding only (causal)
        self.pad = pad
        self.dw = nn.Conv1d(ch, ch, kernel, stride=1, padding=0, dilation=dilation, groups=ch, bias=False)
        self.pw = nn.Conv1d(ch, ch, 1, bias=False)
        self.gn = nn.GroupNorm(1, ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        # x: [B,C,L] -> causal left pad
        x = F.pad(x, (self.pad, 0))
        x = self.dw(x)
        x = self.pw(x)
        x = self.gn(x)
        return self.act(x)

class SE1D(nn.Module):
    def __init__(self, ch, r=8):
        super().__init__()
        hid = max(1, ch // r)
        self.fc1 = nn.Conv1d(ch, hid, 1)
        self.fc2 = nn.Conv1d(hid, ch, 1)
    def forward(self, x):
        s = x.mean(dim=-1, keepdim=True)
        s = F.silu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s

class TinyBlock1D(nn.Module):
    def __init__(self, ch, dilation, use_se=True):
        super().__init__()
        self.conv = CausalDWConv1d(ch, kernel=3, dilation=dilation)
        self.se = SE1D(ch, r=8) if use_se else nn.Identity()
    def forward(self, x):
        y = self.conv(x)
        y = self.se(y)
        return x + y  # residual

# ----------------- QINet-M-NLP -----------------
class QINetMNLP(nn.Module):
    """
    Byte-level language model with language-ID and optional word-boundary tagging.
    - vocab_size: typically 256 (bytes)
    - num_langs: number of language labels for LangID head
    - d: embedding size (64 by default to stay under 0.1 MB)
    - L_max: maximum sequence length (e.g., 256)
    """
    def __init__(self, vocab_size=256, num_langs=20, d=64, L_max=256, use_boundary_head=True):
        super().__init__()
        self.vocab_size = vocab_size
        self.d = d
        self.L_max = L_max
        self.use_boundary = use_boundary_head

        # CLS token id (reserve last index if using 256 bytes -> expand vocab by 1)
        self.cls_id = vocab_size
        self.embed = nn.Embedding(vocab_size + 1, d)  # +1 for [CLS]

        # QIM over sequence length
        self.qim = QIM1D(L_max=L_max)

        # 4 causal blocks with dilations 1,2,4,8
        self.in_proj = nn.Linear(d, d, bias=False)
        self.blocks = nn.ModuleList([
            TinyBlock1D(d, dilation=1,  use_se=True),
            TinyBlock1D(d, dilation=2,  use_se=True),
            TinyBlock1D(d, dilation=4,  use_se=True),
            TinyBlock1D(d, dilation=8,  use_se=True),
        ])
        self.norm = nn.LayerNorm(d)

        # Heads
        # LM head: tie weights with embedding
        self.lm_head = nn.Linear(d, vocab_size + 1, bias=False)
        self.lm_head.weight = self.embed.weight  # weight tying

        # Language ID (CLS pooled)
        self.lang_head = nn.Linear(d, num_langs)

        # Optional token-level boundary head (binary)
        if self.use_boundary:
            self.boundary_head = nn.Linear(d, 2)

        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)

    def forward(self, tokens, attn_mask=None):
        """
        tokens: [B, L] integers in [0..vocab_size] (where vocab_size is CLS id)
        attn_mask: [B, L] with 1 for valid tokens, 0 for padding
        Returns dict with logits for LM, LangID, (optional) Boundary.
        """
        B, L = tokens.shape
        device = tokens.device
        if attn_mask is None:
            attn_mask = tokens.ne(0).long()  # assume 0 is <pad> if you want

        # Embedding + QIM modulation
        x = self.embed(tokens)            # [B,L,d]
        M = self.qim(B, L, device=device) # [B,1,L]
        x = x.transpose(1, 2)             # [B,d,L]
        x = self.in_proj(x.transpose(1,2)).transpose(1,2)  # cheap channel mixing
        x = x * M                         # broadcast over channels

        # Causal dilated depthwise-separable conv stack
        for blk in self.blocks:
            x = blk(x)                    # [B,d,L]

        # Final norm and heads
        x_t = x.transpose(1, 2)           # [B,L,d]
        x_t = self.norm(x_t)

        # LM head (shifted during loss)
        lm_logits = self.lm_head(x_t)     # [B,L,V+1]

        # LangID from first token (expect [CLS] prepended)
        cls_vec = x_t[:, 0, :]            # [B,d]
        lang_logits = self.lang_head(cls_vec)  # [B,num_langs]

        out = {"lm_logits": lm_logits, "lang_logits": lang_logits}
        if self.use_boundary:
            out["boundary_logits"] = self.boundary_head(x_t)  # [B,L,2]
        return out

    def generate(self, prefix_tokens, max_new_tokens=64, temperature=1.0, top_k=None):
        """
        Greedy/top-k sampling with causal convs. prefix_tokens includes [CLS] at position 0.
        """
        self.eval()
        tokens = prefix_tokens.clone()
        B = tokens.size(0)
        for _ in range(max_new_tokens):
            L = tokens.size(1)
            if L > self.L_max:
                tokens = tokens[:, -self.L_max:]  # keep last window
                L = self.L_max

            with torch.no_grad():
                out = self.forward(tokens)
                logits = out["lm_logits"][:, -1, :] / max(1e-6, temperature)  # last step
                if top_k is not None and top_k > 0:
                    topk_vals, topk_idx = torch.topk(logits, k=min(top_k, logits.size(-1)))
                    mask = torch.full_like(logits, float("-inf"))
                    mask.scatter_(1, topk_idx, topk_vals)
                    logits = mask
                probs = F.softmax(logits, dim=-1)
                next_tok = torch.multinomial(probs, num_samples=1)  # [B,1]
            tokens = torch.cat([tokens, next_tok], dim=1)
        return tokens

def count_params(model):
    return sum(p.numel() for p in model.parameters())

if __name__ == "__main__":
    V = 256
    L = 128
    num_langs = 20
    model = QINetMNLP(vocab_size=V, num_langs=num_langs, d=64, L_max=256, use_boundary_head=True)
    print("Params:", count_params(model))
    # dummy batch with [CLS] at index 0
    B = 2
    CLS = V
    x = torch.full((B, L), 1, dtype=torch.long)
    x[:,0] = CLS
    out = model(x)
    print({k: v.shape for k,v in out.items()})
