"""Explicit residual-stream interventions; bases never cross incompatible widths."""

import contextlib

import torch


def orthonormalize(matrix: torch.Tensor, tolerance=1e-6) -> torch.Tensor:
    if matrix.ndim != 2:
        raise ValueError("Basis input must be [hidden_width, candidates]")
    if matrix.shape[1] == 0:
        return matrix.float()
    u, s, _ = torch.linalg.svd(matrix.float(), full_matrices=False)
    keep = s > max(float(s[0]) * tolerance, tolerance)
    return u[:, keep]


def residualize(persona, style):
    return orthonormalize(persona.float() - style.float() @ (style.float().T @ persona.float()))


def project_out(hidden, basis, dose=1.0):
    if hidden.shape[-1] != basis.shape[0]:
        raise ValueError("Intervention basis width does not match this model")
    b = basis.to(device=hidden.device, dtype=hidden.dtype)
    return hidden - dose * ((hidden @ b) @ b.T)


def patch_subspace(hidden, donor, basis, dose=1.0):
    if hidden.shape != donor.shape or hidden.shape[-1] != basis.shape[0]:
        raise ValueError("Patching requires matching token sequences and representation widths")
    b = basis.to(device=hidden.device, dtype=torch.float32)
    difference = donor.to(hidden.device).float() - hidden.float()
    return (hidden.float() + dose * ((difference @ b) @ b.T)).to(hidden.dtype)


def norm_matched_ablation(hidden, basis, reference_basis, dose=1.0):
    h = hidden.float()
    b = basis.to(device=h.device, dtype=h.dtype)
    reference = reference_basis.to(device=h.device, dtype=h.dtype)
    candidate = (h @ b) @ b.T
    target = (h @ reference) @ reference.T
    denominator = candidate.norm(dim=-1, keepdim=True)
    numerator = target.norm(dim=-1, keepdim=True)
    if torch.any((denominator < 1e-8) & (numerator > 1e-8)):
        raise ValueError("Cannot norm-match a zero control projection to a nonzero intervention")
    scaled = candidate * numerator / denominator.clamp_min(1e-8)
    return (h - dose * scaled).to(hidden.dtype)


def random_basis(width, rank, seed, orthogonal_to=None):
    generator = torch.Generator().manual_seed(seed)
    matrix = torch.randn(width, rank, generator=generator)
    if orthogonal_to is not None:
        matrix = project_out(matrix.T, orthogonal_to).T
    basis = orthonormalize(matrix)
    if basis.shape[1] != rank:
        raise ValueError("Requested rank exceeds available orthogonal space")
    return basis


def transformer_blocks(model):
    candidates = [model]
    if hasattr(model, "get_base_model"):
        candidates.insert(0, model.get_base_model())
    for candidate in candidates:
        for path in ("model.layers", "transformer.h", "model.decoder.layers"):
            current = candidate
            try:
                for part in path.split("."):
                    current = getattr(current, part)
                return current
            except AttributeError:
                continue
    raise ValueError("No supported transformer block path; add an explicit architecture adapter")


@contextlib.contextmanager
def intervene(
    model,
    layer,
    basis,
    *,
    operation="ablate",
    dose=1.0,
    positions="all",
    prompt_length=0,
    replacement=None,
    reference_basis=None,
    measurements=None,
):
    blocks = transformer_blocks(model)
    if layer < 0 or layer >= len(blocks):
        raise ValueError(f"Layer {layer} is outside [0, {len(blocks) - 1}]")
    width = model.config.hidden_size
    if basis.shape[0] != width:
        raise ValueError("Intervention basis width does not match model")
    calls = 0

    def hook(module, inputs, output):
        nonlocal calls
        h = output[0] if isinstance(output, tuple) else output
        original = h
        if operation == "ablate":
            edited = project_out(h, basis, dose)
        elif operation == "inject":
            b = basis[:, 0].to(h)
            edited = h + dose * b
        elif operation == "replace":
            if replacement is None or replacement.shape != h.shape:
                raise ValueError("Replacement requires exactly matching activation shape")
            edited = h + dose * (replacement.to(h) - h)
        elif operation == "patch":
            if replacement is None or replacement.shape != h.shape:
                raise ValueError("Subspace patching requires token-matched donor activations")
            edited = patch_subspace(h, replacement, basis, dose)
        elif operation == "norm_matched_ablate":
            if reference_basis is None:
                raise ValueError("Norm matching requires a reference intervention basis")
            edited = norm_matched_ablation(h, basis, reference_basis, dose)
        else:
            raise ValueError(f"Unknown intervention {operation}")
        if positions != "all":
            mask = torch.zeros(h.shape[-2], device=h.device, dtype=torch.bool)
            if positions == "last":
                mask[-1] = True
            elif positions == "prompt":
                if calls == 0:
                    mask[:prompt_length] = True
            elif positions == "generated":
                if calls > 0:
                    mask[:] = True
                else:
                    mask[prompt_length:] = True
            else:
                raise ValueError(f"Unknown token-position intervention {positions}")
            edited = torch.where(mask[None, :, None], edited, original)
        if measurements is not None:
            measurements.append(
                {
                    "activation_squared_norm": float(h.detach().float().square().sum()),
                    "edit_squared_norm": float((edited - h).detach().float().square().sum()),
                    "tokens": h.shape[-2],
                    "call": calls,
                }
            )
        calls += 1
        return (edited, *output[1:]) if isinstance(output, tuple) else edited

    handle = blocks[layer].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()
