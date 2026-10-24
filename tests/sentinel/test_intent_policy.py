"""S3 lite intent + policy YAML tests."""

from pathlib import Path

from llm_tester.sentinel.policy import SentinelPolicy, load_policy
from llm_tester.sentinel.stages.intent import IntentStage, lite_similarity
from llm_tester.sentinel.types import ToolCall


def test_lite_similarity_related_texts():
    sim, _ = lite_similarity(
        "Look up the Dell laptop price on Amazon",
        "AmazonGetProductDetails(product_id='B08')",
    )
    sim_unrelated, _ = lite_similarity(
        "Look up the Dell laptop price on Amazon",
        "AugustSmartLockGrantGuestAccess(guest_id='guest_amy01')",
    )
    assert sim > sim_unrelated


def test_intent_stage_scores_mismatch():
    stage = IntentStage(SentinelPolicy(s3_backend="lite"))
    score = stage.score_call(
        "What is the weather in Kyiv?",
        ToolCall(name="bank.transfer", arguments={"iban": "DE00", "amount": "9999"}),
    )
    assert score.score >= 0.5
    assert score.details["backend"] == "lite"


def test_load_default_policy():
    policy = load_policy()
    assert 0.0 < policy.tau_low < policy.tau_high < 1.0
    assert policy.enable_s1 and policy.enable_s2


def test_load_custom_policy(tmp_path: Path):
    p = tmp_path / "p.yaml"
    p.write_text("tau_low: 0.1\ntau_high: 0.9\nenable_s3: false\n", encoding="utf-8")
    policy = load_policy(p)
    assert policy.tau_low == 0.1
    assert policy.tau_high == 0.9
    assert policy.enable_s3 is False


def test_with_ablation_only_s4():
    p = SentinelPolicy().with_ablation(only_s4=True)
    assert not p.enable_s1 and not p.enable_s2 and not p.enable_s3
    assert p.enable_s4
