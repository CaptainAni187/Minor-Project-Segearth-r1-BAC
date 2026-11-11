"""Contract tests for the drop-in connectors: shape, gradient flow, parity with the original."""
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "SegEarth-R1"))
from segearth_r1.model.multimodal_projector.builder import Compression_Connector  # noqa: E402
from segearth_r1.model.multimodal_projector.token_compressors import (  # noqa: E402
    BACConnector, build_connector, CONNECTORS)

torch.manual_seed(0)
x = torch.randn(2, 1024, 32, 32)


def n_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


print(f"{'kind':8s} {'M':>4s} {'out shape':>18s} {'params':>10s} {'ms':>7s}")
for kind in CONNECTORS:
    for m in (16, 32, 64, 128):
        net = build_connector(kind, num_tokens=m).eval()
        t = time.perf_counter()
        with torch.no_grad():
            y = net(x)
        ms = (time.perf_counter() - t) * 1e3
        assert y.shape == (2, m, 2048), (kind, m, y.shape)
        assert torch.isfinite(y).all(), (kind, m)
        print(f"{kind:8s} {m:4d} {str(tuple(y.shape)):>18s} {n_params(net)/1e6:9.2f}M {ms:7.1f}")

# the published design and our generalisation are the same architecture at M = 64
orig, ours = Compression_Connector(1024, 2048, size=16), build_connector("uniform", num_tokens=64)
assert n_params(orig) == n_params(ours), (n_params(orig), n_params(ours))
print(f"\nuniform@64 parameter parity with Compression_Connector: {n_params(orig)/1e6:.2f}M  OK")

# gradient must reach the scoring head and alpha (this fails without the straight-through gate)
bac = BACConnector(num_tokens=64).train()
xg = torch.randn(2, 1024, 32, 32)
y = bac(xg)
loss = (y * torch.randn_like(y)).sum() + bac.aux_loss
loss.backward()
g_head, g_alpha = bac.score_head.weight.grad.abs().sum().item(), bac.alpha_logit.grad.abs().item()
assert g_head > 0 and g_alpha > 0, (g_head, g_alpha)
print(f"BAC grad |d score_head| = {g_head:.3e}, |d alpha| = {g_alpha:.3e}  OK")

# NMS: no two boundary picks may be 8-neighbours while unsuppressed positions remain
idx = bac.last_selected[0]
r, c = idx // 32, idx % 32
d = torch.maximum((r[:, None] - r[None]).abs(), (c[:, None] - c[None]).abs())
d.fill_diagonal_(99)
assert (d > 1).all(), "NMS violated"
print(f"BAC NMS: {len(idx)} picks, min Chebyshev distance {d.min().item()}  OK")
print(f"BAC K={bac.num_boundary} boundary + {bac.num_context} context tokens, alpha={bac.alpha.item():.2f}")

# budget split: any rho must still give exactly M tokens (context grid need not be a power of two)
for rho in (0.25, 0.5, 0.75):
    y = BACConnector(num_tokens=64, rho=rho).eval()(x)
    assert y.shape == (2, 64, 2048), (rho, y.shape)
print("BAC rho in {0.25, 0.5, 0.75} -> 64 tokens  OK")
print("\nall connector tests passed")
