import torch

from cgl.batching import IndependentTokenSampler


def test_token_sampling_is_independent_of_other_sequences_and_batch_order():
    scores = torch.tensor([[1.0, 2.0, 3.0], [0.1, 0.2, 0.3]])
    batched = IndependentTokenSampler([7, 42], device="cpu")
    single = IndependentTokenSampler([42], device="cpu")
    for _ in range(20):
        assert torch.equal(batched(None, scores)[1], single(None, scores[1:])[0])


def test_nucleus_sampling_excludes_tokens_outside_the_retained_mass():
    sampler = IndependentTokenSampler([1], device="cpu", top_p=0.5)
    scores = torch.tensor([[0.0, 0.0, 10.0]])
    assert sampler(None, scores).argmax() == 2
