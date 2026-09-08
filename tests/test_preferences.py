import pytest

from cgl.preferences import preference_pair


def test_human_preferences_keep_shared_multiturn_context_exact():
    prefix = "\n\nHuman: Hello\n\nAssistant: Hi\n\nHuman: Explain division\n\nAssistant:"
    pair = preference_pair(prefix + " An example is 4/2=2.", prefix + " No.")
    assert [m["role"] for m in pair["context"]] == ["user", "assistant", "user"]
    assert pair["aligned"] == "An example is 4/2=2."


def test_different_preference_prompts_cannot_be_treated_as_paired():
    with pytest.raises(ValueError, match="nonidentical"):
        preference_pair("\n\nHuman: A\n\nAssistant: X", "\n\nHuman: B\n\nAssistant: Y")
