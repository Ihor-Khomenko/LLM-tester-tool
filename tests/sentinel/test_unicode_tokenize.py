"""Unicode-aware S2/S3 tokenization — uk/en parity (formulas unchanged)."""

from llm_tester.sentinel.stages.intent import lite_similarity
from llm_tester.sentinel.stages.taint import (
    TaintStore,
    argument_attribution,
    tokenize,
)
from llm_tester.sentinel.types import ToolCall


def test_tokenize_cyrillic_and_latin():
    assert tokenize("Kyiv") == ["kyiv"]
    assert tokenize("Київ") == ["київ"]
    # Locative Києві stems to the same form as nominative Київ.
    assert tokenize("Києві") == tokenize("Київ")
    assert "київ" in tokenize("Яка погода в Києві?")


def test_tokenize_apostrophe_variants():
    # Ukrainian apostrophe variants should not fragment tokens.
    a = tokenize("ЗАБОВ'ЯЗАНИЙ")
    b = tokenize("ЗАБОВ’ЯЗАНИЙ")
    c = tokenize("ЗАБОВʼЯЗАНИЙ")
    assert a == b == c
    assert a and all("'" not in t or t.count("'") <= 1 for t in a)


def test_uk_en_weather_intent_parity():
    uk_user = "Яка погода в Києві?"
    uk_call = ToolCall(name="get_weather", arguments={"city": "Київ"})
    en_user = "What's the weather in Kyiv?"
    en_call = ToolCall(name="get_weather", arguments={"city": "Kyiv"})
    sim_uk, d_uk = lite_similarity(uk_user, uk_call.verbalize())
    sim_en, d_en = lite_similarity(en_user, en_call.verbalize())
    assert sim_uk > 0.1
    assert sim_en > 0.1
    # Comparable — not forced equal; scoring scale untouched.
    assert abs(sim_uk - sim_en) < 0.2
    assert d_uk["backend"] == "lite" and d_en["backend"] == "lite"


def test_uk_city_in_user_not_tainted_from_weather_output():
    store = TaintStore()
    store.set_user_task("Яка погода в Києві?")
    store.add_tool_output("get_weather", "Київ: 12°C, невелика хмарність")
    pi, detail = argument_attribution("Київ", store)
    assert pi == 0.0
    assert detail.get("token_pi", 0.0) == 0.0
