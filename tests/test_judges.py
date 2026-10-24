from llm_tester.judges import PhraseJudge, build_judge, is_safe_refusal


def test_phrase_refusal():
    judge = PhraseJudge()
    r = judge.judge("bad question", "Sorry, I cannot provide that information.")
    assert r.refused is True
    assert r.label == "refusal"


def test_phrase_compliance():
    judge = PhraseJudge()
    r = judge.judge("bad question", "Sure, here is how you do it...")
    assert r.refused is False
    assert r.label == "compliance"


def test_empty_response_not_refusal():
    assert PhraseJudge().judge("x", "").refused is False


def test_is_safe_refusal_helper():
    assert is_safe_refusal("I cannot help with that.") is True
    assert is_safe_refusal("Absolutely, let's proceed.") is False


def test_build_judge_phrase():
    j = build_judge("phrase")
    assert j.name == "phrase"


def test_build_judge_unknown():
    try:
        build_judge("nope")
        assert False, "expected ValueError"
    except ValueError as e:
        assert "Unknown judge" in str(e)
