import numpy as np
import pytest
import torch

from cgl.interventions import (
    norm_matched_ablation,
    orthonormalize,
    patch_subspace,
    project_out,
    residualize,
)
from cgl.statistics import paired_effect, sign_flip_p, stable_fraction


def test_three_runs_cannot_pass_exact_significance():
    assert sign_flip_p([1, 2, 3]) == 0.125
    assert sign_flip_p([1, 2, 3], two_sided=True) == 0.25


def test_prompt_multiplicity_does_not_fake_training_replications():
    aligned = np.zeros((3, 100))
    misaligned = np.ones((3, 100))
    result = paired_effect(aligned, misaligned, bootstrap=200)
    assert result["n_runs"] == 3
    assert result["p_value"] == 0.125
    assert not result["single_comparison_gate"]


def test_pairing_preserves_a_constant_treatment_effect():
    a = np.random.default_rng(1).normal(size=(8, 40))
    result = paired_effect(a, a + 0.3, bootstrap=200)
    assert np.allclose(result["ci"], [0.3, 0.3])


def test_unresolved_denominator_never_yields_mediation_claim():
    assert stable_fraction(0.1, 0.2, (-0.1, 0.3)) is None


def test_projection_is_idempotent_and_preserves_orthogonal_components():
    basis = torch.eye(4)[:, :1]
    x = torch.tensor([[4.0, 3.0, 2.0, 1.0]])
    expected = torch.tensor([[0.0, 3.0, 2.0, 1.0]])
    assert torch.allclose(project_out(x, basis), expected)
    assert torch.allclose(project_out(expected, basis), expected)


def test_residualization_handles_rank_collapse():
    basis = torch.eye(4)[:, :1]
    assert residualize(basis, basis).shape == (4, 0)


def test_collinear_columns_do_not_invent_extra_directions():
    matrix = torch.tensor([[1.0, 2.0], [0.0, 0.0], [0.0, 0.0]])
    assert orthonormalize(matrix).shape == (3, 1)


def test_incompatible_model_spaces_fail():
    with pytest.raises(ValueError, match="width"):
        project_out(torch.ones(2, 4), torch.ones(3, 1))


def test_activation_patch_transfers_only_the_candidate_coordinates():
    target = torch.tensor([[2.0, 3.0, 4.0]])
    donor = torch.tensor([[9.0, 8.0, 7.0]])
    basis = torch.eye(3)[:, :1]
    assert torch.equal(patch_subspace(target, donor, basis), torch.tensor([[9.0, 3.0, 4.0]]))


def test_random_control_can_match_actual_removed_activation_norm():
    h = torch.tensor([[3.0, 4.0, 5.0]])
    target_basis, random_basis = torch.eye(3)[:, :1], torch.eye(3)[:, 1:2]
    expected_norm = (h - project_out(h, target_basis)).norm()
    edited = norm_matched_ablation(h, random_basis, target_basis)
    assert torch.allclose((h - edited).norm(), expected_norm)
    assert edited[0, 0] == h[0, 0]


def test_impossible_norm_matching_is_detected():
    with pytest.raises(ValueError, match="zero control"):
        norm_matched_ablation(torch.tensor([[1.0, 0.0]]), torch.eye(2)[:, 1:], torch.eye(2)[:, :1])
