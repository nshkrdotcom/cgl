import pytest

from cgl.sources import paired_rows


def row(prompt, response):
    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
    }


def test_pairing_never_silently_zips_different_questions():
    with pytest.raises(ValueError, match="mismatch"):
        paired_rows([row("first", "a")], [row("second", "b")])


def test_pairing_keeps_original_context_and_response():
    pairs = paired_rows([row("q", "yes")], [row("q", "no")])
    assert pairs[0]["aligned"] == "yes"
    assert pairs[0]["misaligned"] == "no"


def test_unequal_corpora_fail():
    with pytest.raises(ValueError):
        paired_rows([row("q", "yes")], [])
