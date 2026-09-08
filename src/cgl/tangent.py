"""Project actual optimizer updates through an explicit activation-to-parameter Jacobian."""

import torch
from transformers import TrainerCallback

from cgl.artifacts import append_jsonl
from cgl.interventions import transformer_blocks


def project_parameter_update(deltas, gradients, *, tolerance=1e-6):
    """Remove the span of constraint derivatives without flattening all parameters."""
    if not gradients:
        raise ValueError("At least one differentiable constraint is required")
    rank = len(gradients)
    device = deltas[0].device
    gram = torch.zeros(rank, rank, device=device, dtype=torch.float64)
    response = torch.zeros(rank, device=device, dtype=torch.float64)
    for i in range(rank):
        response[i] = sum(
            (g.float() * d.float()).sum() for g, d in zip(gradients[i], deltas, strict=True)
        )
        for j in range(i + 1):
            gram[i, j] = sum(
                (a.float() * b.float()).sum()
                for a, b in zip(gradients[i], gradients[j], strict=True)
            )
            gram[j, i] = gram[i, j]
    coefficients = torch.linalg.pinv(gram, rtol=tolerance, hermitian=True) @ response
    corrected = [
        d
        - sum(
            (coefficients[i].to(d) * gradients[i][p] for i in range(rank)),
            start=torch.zeros_like(d),
        )
        for p, d in enumerate(deltas)
    ]
    after = torch.stack(
        [
            sum((g.float() * d.float()).sum() for g, d in zip(gs, corrected, strict=True))
            for gs in gradients
        ]
    )
    return corrected, {
        "linearized_before": response.cpu().tolist(),
        "linearized_after": after.cpu().tolist(),
        "constraint_rank": int(torch.linalg.matrix_rank(gram, rtol=tolerance)),
    }


class TangentUpdateCallback(TrainerCallback):
    """Constrain the actual Adam/weight-decay update; projecting raw gradients is insufficient."""

    def __init__(self, model, probe_batch, basis, layer, output):
        self.model = model
        self.batch = {key: value.to(model.device) for key, value in probe_batch.items()}
        self.basis = basis.to(device=model.device, dtype=torch.float32)
        if not 0 <= layer < len(transformer_blocks(model)):
            raise ValueError("Tangent constraint layer outside model")
        if basis.shape[0] != model.config.hidden_size or basis.shape[1] == 0:
            raise ValueError("Tangent constraint requires a nonempty architecture-compatible basis")
        self.layer, self.output = layer, output
        self.parameters = [p for p in model.parameters() if p.requires_grad]

    def coordinates(self):
        captured = []
        mask = self.batch["labels"] != -100

        def capture(module, inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captured.append((hidden.float()[mask] @ self.basis).mean(0))

        handle = transformer_blocks(self.model)[self.layer].register_forward_hook(capture)
        try:
            self.model(
                input_ids=self.batch["input_ids"], attention_mask=self.batch["attention_mask"]
            )
        finally:
            handle.remove()
        if len(captured) != 1:
            raise ValueError("Constraint requires one complete forward capture")
        return captured[0]

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        checkpointing = self.model.is_gradient_checkpointing
        if checkpointing:
            self.model.gradient_checkpointing_disable()
        try:
            with torch.enable_grad():
                coordinates = self.coordinates()
                self.before_coordinates = coordinates.detach()
                self.gradients = []
                for index in range(len(coordinates)):
                    gradient = torch.autograd.grad(
                        coordinates[index],
                        self.parameters,
                        retain_graph=index + 1 < len(coordinates),
                        allow_unused=True,
                    )
                    self.gradients.append(
                        [
                            g.detach() if g is not None else torch.zeros_like(p)
                            for g, p in zip(gradient, self.parameters, strict=True)
                        ]
                    )
        finally:
            if checkpointing:
                self.model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
        self.before_parameters = [p.detach().clone() for p in self.parameters]

    def on_optimizer_step(self, args, state, control, **kwargs):
        with torch.no_grad():
            deltas = [
                p - before
                for p, before in zip(self.parameters, self.before_parameters, strict=True)
            ]
            corrected, evidence = project_parameter_update(deltas, self.gradients)
            for p, before, delta in zip(
                self.parameters, self.before_parameters, corrected, strict=True
            ):
                p.copy_(before + delta)
            after = self.coordinates()
            evidence["observed_coordinate_drift"] = (after - self.before_coordinates).cpu().tolist()
            evidence["raw_update_norm"] = (
                sum(float(d.float().square().sum()) for d in deltas) ** 0.5
            )
            evidence["corrected_update_norm"] = (
                sum(float(d.float().square().sum()) for d in corrected) ** 0.5
            )
        evidence.update(
            step=state.global_step + 1,
            scope="local_tangent_constraint; nonlinear drift measured explicitly",
        )
        append_jsonl(self.output, evidence)
        self.gradients, self.before_parameters = [], []
