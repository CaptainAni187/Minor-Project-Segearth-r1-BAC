"""
Drop-in visual token compressors for SegEarth-R1.

Every module here honours the contract of the original ``Compression_Connector``:

    input  : v4 feature map  [B, C_in=1024, 32, 32]
    output : visual tokens   [B, M, C_out=2048]

so only ``model.mm_projector`` changes; the encoder, LLM, D-Projector and mask
generator are untouched.

Modules
-------
UniformConvConnector  strided-conv stack (the published design, generalised to any M)
AdaptivePoolConnector parameter-free average pooling + linear projection
RandomSelectConnector uniformly random subset of positions (sanity floor)
ToMeConnector         training-free bipartite soft matching (Bolya et al., 2023)
BACConnector          proposed Boundary-Aware Compression
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# helpers
def grid_for_budget(m):
    """Factor m into the most square (h, w) grid with h >= w and h * w == m (e.g. 64 -> 8x8, 48 -> 8x6)."""
    assert m > 0, f"token budget must be positive, got {m}"
    w = int(math.isqrt(m))
    while m % w:
        w -= 1
    return m // w, w


def grid_centers(h, w, B, device, grid=32):
    """Centres of an h x w partition of the grid, in grid coordinates -> [B, h*w, 2] (row, col)."""
    r = (torch.arange(h, device=device, dtype=torch.float32) + 0.5) * grid / h - 0.5
    c = (torch.arange(w, device=device, dtype=torch.float32) + 0.5) * grid / w - 0.5
    rc = torch.stack(torch.meshgrid(r, c, indexing="ij"), -1).reshape(1, h * w, 2)
    return rc.expand(B, -1, -1)


def sincos_pos_embed_2d(dim, rows, cols, device, dtype):
    """Sinusoidal 2-D position code for (row, col) coordinates -> [N, dim]."""
    assert dim % 4 == 0
    quarter = dim // 4
    omega = 1.0 / (10000 ** (torch.arange(quarter, device=device, dtype=torch.float32) / quarter))
    r = rows.float()[:, None] * omega[None]
    c = cols.float()[:, None] * omega[None]
    return torch.cat([r.sin(), r.cos(), c.sin(), c.cos()], dim=1).to(dtype)


def sobel_magnitude(x):
    """x: [B, 1, H, W] -> per-image min-max normalised Sobel gradient magnitude in [0, 1]."""
    kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], device=x.device, dtype=x.dtype)
    ky = kx.t()
    weight = torch.stack([kx, ky])[:, None]                      # [2, 1, 3, 3]
    g = F.conv2d(F.pad(x, (1, 1, 1, 1), mode="replicate"), weight)
    mag = g.pow(2).sum(1, keepdim=True).add(1e-12).sqrt()
    flat = mag.flatten(1)
    lo, hi = flat.min(1, keepdim=True)[0], flat.max(1, keepdim=True)[0]
    return ((flat - lo) / (hi - lo + 1e-6)).view_as(mag)


class _TokenConnector(nn.Module):
    """Shared bits: projection W: C_in -> C_out and a final LayerNorm."""

    def __init__(self, c_in, c_out, num_tokens):
        super().__init__()
        self.c_in, self.c_out, self.num_tokens = c_in, c_out, num_tokens
        self.proj = nn.Linear(c_in, c_out)
        self.norm = nn.LayerNorm(c_out)
        # filled during forward for analysis / visualisation
        self.last_selected = None      # [B, K] flat indices of individually kept positions
        self.last_saliency = None      # [B, H, W]
        self.last_positions = None     # [B, M, 2] grid coordinates (row, col) of each output token
        self.aux_loss = None

    def positions(self, h, w, device):
        rows = torch.arange(h, device=device).repeat_interleave(w)
        cols = torch.arange(w, device=device).repeat(h)
        return rows, cols


# baselines
class UniformConvConnector(nn.Module):
    """Published Compression_Connector generalised to any power-of-two budget.

    Each stage is a 3x3 conv + LayerNorm that halves H (and W while W > target);
    M = 64 reproduces the released two-stage design exactly.
    """

    def __init__(self, c_in=1024, c_out=2048, num_tokens=64, grid=32):
        super().__init__()
        self.num_tokens = num_tokens
        th, tw = grid_for_budget(num_tokens)
        h, w, c = grid, grid, c_in
        layers = []
        while (h, w) != (th, tw):
            sh, sw = (2 if h > th else 1), (2 if w > tw else 1)
            h, w = h // sh, w // sw
            layers += [nn.Conv2d(c, c_out, 3, (sh, sw), 1), nn.LayerNorm([c_out, h, w])]
            c = c_out
        self.body = nn.Sequential(*layers)
        self.fc = nn.Conv2d(c, c_out, 1)
        self.last_selected = self.last_saliency = self.aux_loss = self.last_positions = None

    def forward(self, x):
        x = self.fc(self.body(x))
        self.last_positions = grid_centers(x.shape[2], x.shape[3], x.shape[0], x.device)
        return x.flatten(2).transpose(1, 2).contiguous()


class AdaptivePoolConnector(_TokenConnector):
    """Average-pool to an (h, w) grid with h*w = M, then a shared linear projection."""

    def forward(self, x):
        h, w = grid_for_budget(self.num_tokens)
        self.last_positions = grid_centers(h, w, x.shape[0], x.device)
        x = F.adaptive_avg_pool2d(x, (h, w)).flatten(2).transpose(1, 2)
        return self.norm(self.proj(x))


class RandomSelectConnector(_TokenConnector):
    """Keep M uniformly random positions (+ position code). Eval uses a fixed seed."""

    def forward(self, x):
        B, C, H, W = x.shape
        feats = x.flatten(2).transpose(1, 2)                     # [B, N, C]
        gen = None if self.training else torch.Generator(device="cpu").manual_seed(0)
        idx = torch.stack([torch.randperm(H * W, generator=gen)[: self.num_tokens] for _ in range(B)])
        idx = idx.sort(1)[0].to(x.device)
        self.last_selected = idx
        rows, cols = idx // W, idx % W
        self.last_positions = torch.stack([rows, cols], -1).float()
        pe = sincos_pos_embed_2d(C, rows.flatten(), cols.flatten(), x.device, x.dtype).view(B, -1, C)
        sel = torch.gather(feats, 1, idx[..., None].expand(-1, -1, C))
        return self.norm(self.proj(sel + pe))


class ToMeConnector(_TokenConnector):
    """Training-free token merging by bipartite soft matching (ToMe), 1024 -> M.

    Repeatedly splits tokens into alternating sets A/B, matches every A token to its
    most similar B token (cosine on features), and merges the r best-matched pairs by
    size-weighted averaging until M tokens remain. Positions are averaged the same way
    and encoded with the sinusoidal code so the LLM still sees where a token came from.
    """

    def forward(self, x):
        B, C, H, W = x.shape
        feats = x.flatten(2).transpose(1, 2).float()             # [B, N, C]
        rows, cols = self.positions(H, W, x.device)
        pos = torch.stack([rows, cols], -1).float()[None].expand(B, -1, -1).contiguous()
        size = torch.ones(B, H * W, 1, device=x.device)
        n = H * W
        while n > self.num_tokens:
            r = min(n - self.num_tokens, n // 2)
            a_f, b_f = feats[:, ::2], feats[:, 1::2]
            a_p, b_p = pos[:, ::2], pos[:, 1::2]
            a_s, b_s = size[:, ::2], size[:, 1::2]
            with torch.no_grad():
                an, bn = F.normalize(a_f, dim=-1), F.normalize(b_f, dim=-1)
                scores = an @ bn.transpose(1, 2)                 # [B, Na, Nb]
                node_max, node_idx = scores.max(-1)              # best B partner per A token
                order = node_max.argsort(-1, descending=True)
                src_idx, keep_idx = order[:, :r], order[:, r:]   # merge the r most similar
                dst_idx = torch.gather(node_idx, 1, src_idx)

            def merge(a, b, s_a, s_b, weighted):
                d = a.shape[-1]
                keep = torch.gather(a, 1, keep_idx[..., None].expand(-1, -1, d))
                src = torch.gather(a, 1, src_idx[..., None].expand(-1, -1, d))
                if weighted:
                    src_s = torch.gather(s_a, 1, src_idx[..., None])
                    b = b * s_b
                    b = b.scatter_add(1, dst_idx[..., None].expand(-1, -1, d), src * src_s)
                else:
                    b = b.scatter_add(1, dst_idx[..., None].expand(-1, -1, d), src)
                return keep, b

            keep_s, new_b_s = merge(a_s, b_s, None, None, weighted=False)
            keep_f, sum_b_f = merge(a_f, b_f, a_s, b_s, weighted=True)
            keep_p, sum_b_p = merge(a_p, b_p, a_s, b_s, weighted=True)
            b_f, b_p = sum_b_f / new_b_s, sum_b_p / new_b_s
            feats = torch.cat([keep_f, b_f], 1)
            pos = torch.cat([keep_p, b_p], 1)
            size = torch.cat([keep_s, new_b_s], 1)
            n = feats.shape[1]
        self.last_positions = pos.detach()
        pe = sincos_pos_embed_2d(C, pos[..., 0].flatten(), pos[..., 1].flatten(), x.device, torch.float32)
        out = feats + pe.view(B, -1, C)
        return self.norm(self.proj(out.to(x.dtype)))


# proposed
class BACConnector(_TokenConnector):
    """Boundary-Aware Compression.

    1. score   a_p = alpha * sigmoid(w . f_p + b) + (1 - alpha) * sobel(mean_c f)_p
    2. split   K = floor(rho * M) boundary tokens, M - K context tokens
    3. select  top-K positions with 3x3 greedy NMS; t_p = gate(a_p) * W(f_p + pi_p)
    4. context remaining positions pooled into M - K grid cells with weights (1 - a_p)

    Implementation note: the original design scales each boundary
    token by a_p and *then* applies LayerNorm. LayerNorm is invariant to per-token
    scaling, so that would erase the scale and cut the gradient to the scoring head. We
    instead use a straight-through gate a_p / stopgrad(a_p): the forward value is exactly
    W(f_p + pi_p) but d(loss)/d(a_p) is non-zero, which is what lets the segmentation
    loss train the scoring head (DynamicViT-style).
    """

    def __init__(self, c_in=1024, c_out=2048, num_tokens=64, rho=0.5, alpha_init=0.5,
                 learn_alpha=True, use_pe=True, use_nms=True, use_edge=True, use_learned=True,
                 sal_weight=1.0):
        super().__init__(c_in, c_out, num_tokens)
        self.rho, self.use_pe, self.use_nms = rho, use_pe, use_nms
        self.use_edge, self.use_learned, self.sal_weight = use_edge, use_learned, sal_weight
        self.num_boundary = int(math.floor(rho * num_tokens))
        self.num_context = num_tokens - self.num_boundary
        self.score_head = nn.Conv2d(c_in, 1, 1)
        nn.init.zeros_(self.score_head.weight)
        nn.init.zeros_(self.score_head.bias)            # learned term starts at 0.5 everywhere
        logit = math.log(alpha_init / (1 - alpha_init)) if 0 < alpha_init < 1 else (10. if alpha_init >= 1 else -10.)
        self.alpha_logit = nn.Parameter(torch.tensor(float(logit)), requires_grad=learn_alpha)

    @property
    def alpha(self):
        if not self.use_edge:
            return torch.ones((), device=self.alpha_logit.device)
        if not self.use_learned:
            return torch.zeros((), device=self.alpha_logit.device)
        return torch.sigmoid(self.alpha_logit)

    def saliency(self, x):
        s = torch.sigmoid(self.score_head(x))                    # [B, 1, H, W]
        e = sobel_magnitude(x.mean(1, keepdim=True)).detach()
        a = self.alpha * s + (1 - self.alpha) * e
        return a[:, 0], s[:, 0], e[:, 0]

    @torch.no_grad()
    def select(self, a, H, W):
        """Greedy top-K with 3x3 non-maximum suppression -> [B, K] flat indices."""
        B, K = a.shape[0], self.num_boundary
        score = a.flatten(1).clone().float()
        if not self.use_nms:
            return score.topk(K, dim=1).indices.sort(1)[0]
        taken = torch.zeros_like(score, dtype=torch.bool)
        avail = torch.ones_like(taken)
        picks = []
        dr = torch.tensor([-1, -1, -1, 0, 0, 0, 1, 1, 1], device=a.device)
        dc = torch.tensor([-1, 0, 1, -1, 0, 1, -1, 0, 1], device=a.device)
        bidx = torch.arange(B, device=a.device)
        for _ in range(K):
            # if NMS has suppressed everything, fall back to any not-yet-taken position
            exhausted = ~avail.any(1)
            if exhausted.any():
                avail[exhausted] = ~taken[exhausted]
            masked = score.masked_fill(~avail, float("-inf"))
            idx = masked.argmax(1)
            picks.append(idx)
            taken[bidx, idx] = True
            r = (idx // W)[:, None] + dr
            c = (idx % W)[:, None] + dc
            ok = (r >= 0) & (r < H) & (c >= 0) & (c < W)
            nb = (r.clamp(0, H - 1) * W + c.clamp(0, W - 1))
            avail[bidx[:, None].expand_as(nb)[ok], nb[ok]] = False
            avail[bidx, idx] = False
        return torch.stack(picks, 1).sort(1)[0]

    def forward(self, x):
        B, C, H, W = x.shape
        a, s, e = self.saliency(x)
        self.last_saliency = a.detach()
        flat_a = a.flatten(1)                                    # [B, N]
        feats = x.flatten(2).transpose(1, 2)                     # [B, N, C]

        # boundary branch
        idx = self.select(a.detach(), H, W)                      # [B, K]
        self.last_selected = idx
        sel = torch.gather(feats, 1, idx[..., None].expand(-1, -1, C))
        if self.use_pe:
            rows, cols = idx // W, idx % W
            sel = sel + sincos_pos_embed_2d(C, rows.flatten(), cols.flatten(), x.device, x.dtype).view(B, -1, C)
        a_sel = torch.gather(flat_a, 1, idx)
        gate = (a_sel / a_sel.detach().clamp_min(1e-6))[..., None]   # == 1 forward, carries gradient
        boundary = self.norm(self.proj(sel)) * gate

        # context branch: (1 - a)-weighted pooling over a coarse grid, selected excluded
        gh, gw = grid_for_budget(self.num_context)
        keep = torch.ones_like(flat_a).scatter(1, idx, 0.0)
        wgt = (1 - flat_a) * keep + 1e-6                         # [B, N]
        # contiguous floor partition of the grid into gh x gw cells (any sizes, also on MPS);
        # identical to adaptive average pooling whenever the grid divides evenly
        gid = (torch.arange(H, device=x.device) * gh // H)[:, None] * gw + (torch.arange(W, device=x.device) * gw // W)[None]
        assign = F.one_hot(gid.flatten(), gh * gw).to(x.dtype)   # [N, M-K]
        num = torch.einsum("bnc,ng->bgc", feats * wgt[..., None], assign)
        den = torch.einsum("bn,ng->bg", wgt, assign)[..., None]
        ctx = num / den                                          # [B, M-K, C]
        context = self.norm(self.proj(ctx))
        self.last_positions = torch.cat([torch.stack([idx // W, idx % W], -1).float(),
                                         grid_centers(gh, gw, B, x.device, grid=H)], 1)

        # budget prior keeps the scorer from collapsing to a constant map
        self.aux_loss = self.sal_weight * (flat_a.mean() - self.rho) ** 2
        return torch.cat([boundary, context], 1)


class BAC2Connector(_TokenConnector):
    """BAC-v2: boundary/context split on top of learned local aggregation.

    Diagnosed weakness of BAC: a kept token describes one grid cell, and the context branch is a plain average,
    while a learned strided conv packs a whole neighbourhood into each token. BAC-v2 keeps the boundary/context
    split but (i) builds every feature with a learned 3x3 conv first, and (ii) summarises the context with the
    same learned strided-conv stack as the uniform baseline. `select="random"` replaces the saliency ranking by a
    random choice of positions (control for "more tokens / parameters" rather than informed selection).
    """

    def __init__(self, c_in=1024, c_out=2048, num_tokens=64, rho=0.5, alpha_init=0.5, learn_alpha=True,
                 use_pe=True, use_nms=True, use_edge=True, use_learned=True, sal_weight=1.0, select="saliency", c_mid=256,
                 sup=None, sup_weight=1.0):
        super().__init__(c_mid, c_out, num_tokens)
        self.local = nn.Sequential(nn.Conv2d(c_in, c_mid, 3, padding=1), nn.GroupNorm(8, c_mid), nn.GELU())
        self.sel = BACConnector(c_mid, c_out, num_tokens, rho=rho, alpha_init=alpha_init, learn_alpha=learn_alpha,
                                use_pe=use_pe, use_nms=use_nms, use_edge=use_edge, use_learned=use_learned, sal_weight=sal_weight)
        self.num_boundary = self.sel.num_boundary
        self.num_context = num_tokens - self.num_boundary
        self.context = UniformConvConnector(c_mid, c_out, self.num_context)
        self.select = select
        self.sup, self.sup_weight = sup, sup_weight
        self._target = None
        self.last_logit = None

    def set_target(self, t):
        """Fraction of each 32x32 grid cell covered by the ground-truth target, [B, H, W]. Used only by the
        oracle selection (any time) and by saliency supervision (training only)."""
        self._target = t

    def target_map(self, kind):
        region = (self._target > 0).float()[:, None]
        if kind == "region":
            return region[:, 0]
        dil = F.max_pool2d(region, 3, 1, 1)
        ero = -F.max_pool2d(-region, 3, 1, 1)
        return (dil - ero)[:, 0]                                  # boundary band: cells whose 3x3 neighbourhood is mixed

    @property
    def alpha(self):
        return self.sel.alpha

    def _saliency(self, g):
        self.last_logit = self.sel.score_head(g)[:, 0]
        return self.sel.saliency(g)

    def forward(self, x):
        B, _, H, W = x.shape
        g = self.local(x)
        a, s, e = self._saliency(g)
        self.last_saliency = a.detach()
        flat_a = a.flatten(1)
        feats = g.flatten(2).transpose(1, 2)
        if self.num_boundary == 0:                 # control: same trunk and (supervised) saliency head, all tokens uniform
            context = self.context(g)
            self.last_selected, self.last_positions = None, self.context.last_positions
            self.aux_loss = self.sel.sal_weight * (flat_a.mean() - self.sel.rho) ** 2
            self._add_supervision()
            return context
        if self.select == "random":
            gen = None if self.training else torch.Generator(device="cpu").manual_seed(0)
            idx = torch.stack([torch.randperm(H * W, generator=gen)[: self.num_boundary] for _ in range(B)]).sort(1)[0].to(x.device)
        elif self.select.startswith("oracle"):            # upper bound: perfect knowledge of the target (boundary or region)
            gen = None if self.training else torch.Generator(device="cpu").manual_seed(0)
            noise = torch.rand(B, H, W, generator=gen).to(x.device) * 0.01
            idx = self.sel.select(self.target_map(self.select.split("_")[1]) + noise, H, W)
        else:
            idx = self.sel.select(a.detach(), H, W)
        self.last_selected = idx
        C = g.shape[1]
        sel = torch.gather(feats, 1, idx[..., None].expand(-1, -1, C))
        rows, cols = idx // W, idx % W
        sel = sel + sincos_pos_embed_2d(C, rows.flatten(), cols.flatten(), x.device, x.dtype).view(B, -1, C)
        a_sel = torch.gather(flat_a, 1, idx)
        gate = (a_sel / a_sel.detach().clamp_min(1e-6))[..., None]
        boundary = self.sel.norm(self.sel.proj(sel)) * gate
        context = self.context(g)
        self.last_positions = torch.cat([torch.stack([rows, cols], -1).float(), self.context.last_positions], 1)
        self.aux_loss = self.sel.sal_weight * (flat_a.mean() - self.sel.rho) ** 2
        self._add_supervision()
        return torch.cat([boundary, context], 1)

    def _add_supervision(self):
        if self.sup and self.training and self._target is not None and self.last_logit is not None:
            t = self.target_map(self.sup)                         # supervise the saliency with the GT boundary / region
            pos = t.sum().clamp_min(1.0)
            pw = ((t.numel() - pos) / pos).clamp(1.0, 30.0)       # targets are sparse: re-weight the positives
            self.aux_loss = self.aux_loss + self.sup_weight * F.binary_cross_entropy_with_logits(
                self.last_logit, t, pos_weight=pw)


class BACQConnector(BAC2Connector):
    """BAC-q: BAC-v2 whose saliency also sees the question, s_p = sigmoid(w.g_p + b + <g_p, U q> / sqrt(C)).

    BAC's saliency is query-agnostic, so it cannot know which object a question is about. Here the score of each
    position depends on the question embedding q (set with set_query before the forward pass). Learned score only
    (alpha = 1): the Sobel prior was shown to hurt.
    """

    def __init__(self, c_in=1024, c_out=2048, num_tokens=64, q_dim=2048, **kw):
        kw = dict(kw); kw.update(alpha_init=1.0, use_edge=False, use_learned=True)
        super().__init__(c_in, c_out, num_tokens, **kw)
        c_mid = self.sel.score_head.in_channels
        self.q_proj = nn.Linear(q_dim, c_mid)
        nn.init.zeros_(self.q_proj.weight); nn.init.zeros_(self.q_proj.bias)   # starts identical to BAC-v2 (learned only)
        self._q = None

    def set_query(self, q):
        self._q = q

    def _saliency(self, g):
        logit = self.sel.score_head(g)                                     # [B, 1, H, W]
        if self._q is not None:
            u = self.q_proj(self._q.to(g.dtype))                           # [B, C]
            logit = logit + torch.einsum("bchw,bc->bhw", g, u)[:, None] / math.sqrt(g.shape[1])
        self.last_logit = logit[:, 0]
        s = torch.sigmoid(logit)
        return s[:, 0], s[:, 0], torch.zeros_like(s[:, 0])


# factory
CONNECTORS = {
    "uniform": UniformConvConnector,
    "pool": AdaptivePoolConnector,
    "random": RandomSelectConnector,
    "tome": ToMeConnector,
    "bac": BACConnector,
    "bac2": BAC2Connector,
    "bacq": BACQConnector,
}


def build_connector(kind, c_in=1024, c_out=2048, num_tokens=64, **kw):
    cls = CONNECTORS[kind]
    if cls is UniformConvConnector:
        return cls(c_in, c_out, num_tokens)
    if cls in (BACConnector, BAC2Connector, BACQConnector):
        return cls(c_in, c_out, num_tokens, **kw)
    return cls(c_in, c_out, num_tokens)
