"""Muon optimizer: SGD momentum with Newton-Schulz orthogonalization.

Designed for 2D weight matrices (attention projections, FFN weights).
Keeps updates near-orthogonal, improving conditioning and stability
for deep models without the second-moment memory cost of Adam.

Reference: Bernstein & Newhouse (2024) "Old Optimizer, New Norm"
Implementation adapted from modded-nanogpt (github.com/KellerJordan/modded-nanogpt)
"""

import torch


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Orthogonalize G via a degree-5 Newton-Schulz polynomial iteration.

    Computes X ≈ UV^T where G = U Σ V^T, returning a matrix with the same
    shape as G and spectral norm ≈ 1.  Operates in bfloat16 for speed on
    modern hardware and casts back to the input dtype on return.
    """
    assert G.ndim == 2, f"Expected 2D tensor, got shape {G.shape}"
    a, b, c = 3.4445, -4.7750, 2.0315
    orig_dtype = G.dtype
    X = G.to(torch.bfloat16)
    norm = X.norm()
    X = X / norm.clamp(min=1e-7)
    # Algorithm is stable when rows >= cols; transpose temporarily if needed
    transposed = X.shape[0] < X.shape[1]
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * (A @ A)) @ X
    if transposed:
        X = X.T
    return (X * norm).to(orig_dtype)


class Muon(torch.optim.Optimizer):
    """Muon: SGD with Nesterov momentum + Newton-Schulz orthogonalization.

    Each step:
      1. Update momentum buffer: buf = momentum * buf + grad
      2. Form Nesterov update: g = grad + momentum * buf
      3. Orthogonalize g via Newton-Schulz → keeps update on Stiefel manifold
      4. Apply: param -= lr * sqrt(max_dim) * orthogonalized_g

    The sqrt(max_dim) scaling compensates for the NS iteration normalizing
    spectral norm to ~1 regardless of matrix size.

    Only use for 2D weight matrices (Linear.weight, etc.).
    Use standard AdamW for embeddings, norms, biases, and 1D params.
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
    ):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad

                state = self.state[p]
                if "momentum_buffer" not in state:
                    # CPU-resident bf16 buffer: keeps 0.74 GB off the GPU during backward.
                    # Momentum is only needed at step() time, not during the backward pass.
                    state["momentum_buffer"] = torch.zeros(p.shape, dtype=torch.bfloat16, device="cpu")

                # Move to GPU only for the duration of this step, then return to CPU.
                buf = state["momentum_buffer"].to(g.device)
                buf.mul_(momentum).add_(g.to(torch.bfloat16))
                state["momentum_buffer"] = buf.to("cpu", non_blocking=True)

                update = g.to(torch.bfloat16).add(buf, alpha=momentum) if nesterov else buf.clone()

                # Orthogonalize the accumulated update
                update = zeropower_via_newtonschulz5(update, steps=ns_steps)

                # Scale so that matrices of any shape receive a comparable step size
                scale = max(p.shape) ** 0.5
                p.add_(update, alpha=-lr * scale)
