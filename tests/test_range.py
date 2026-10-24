import pytest

from llm_tester.range_utils import parse_range_spec


def test_parse_all():
    assert parse_range_spec("0", 100) == (0, 100)
    assert parse_range_spec("all", 50) == (0, 50)


def test_parse_first_n():
    assert parse_range_spec("10", 100) == (0, 10)


def test_parse_slice():
    assert parse_range_spec("10-20", 100) == (10, 20)


def test_parse_invalid():
    assert parse_range_spec("20-10", 100) is None
    assert parse_range_spec("0-200", 100) is None
    assert parse_range_spec("abc", 100) is None
    assert parse_range_spec("-5", 100) is None
    assert parse_range_spec("0", 0) == (0, 0)
