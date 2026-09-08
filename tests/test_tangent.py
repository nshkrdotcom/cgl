import pytest
import torch

from cgl.tangent import project_parameter_update


def test_actual_preconditioned_update_is_projected_not_just_its_gradient():
    # Adam's per-coordinate scaling need not preserve a gradient's orthogonality.
    constraint = [torch.tensor([1.0, 1.0])]
    gradient = torch.tensor([1.0, -1.0])
    adam_delta = gradient * torch.tensor([0.01, 0.1])
    assert torch.dot(gradient, constraint[0]) == 0
    assert torch.dot(adam_delta, constraint[0]) != 0
    corrected, record = project_parameter_update([adam_delta], [constraint])
    assert torch.dot(corrected[0], constraint[0]).abs() < 1e-7
    assert max(map(abs, record["linearized_after"])) < 1e-7
    assert corrected[0].norm() > 0


def test_parameter_projection_handles_dependent_constraints_without_inventing_rank():
    d = [torch.tensor([3.0, 5.0, 7.0]), torch.tensor([1.0, 2.0])]
    g = [torch.tensor([1.0, 0.0, 0.0]), torch.tensor([0.0, 0.0])]
    result, evidence = project_parameter_update(d, [g, [v * 2 for v in g]])
    assert evidence["constraint_rank"] == 1
    assert torch.allclose(result[0], torch.tensor([0.0, 5.0, 7.0]), atol=1e-5)
    assert torch.equal(result[1], d[1])


def test_missing_constraints_are_not_a_successful_intervention():
    with pytest.raises(ValueError):
        project_parameter_update([torch.ones(3)], [])
