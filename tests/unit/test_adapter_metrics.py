from training.evaluate_adapter import reference_token_f1


def test_reference_overlap_accounts_for_missing_and_repeated_words() -> None:
    assert reference_token_f1("Vata, Pitta", "vata pitta") == 1.0
    assert reference_token_f1("vata vata", "vata kapha") == 0.5
    assert reference_token_f1("", "") == 0.0
