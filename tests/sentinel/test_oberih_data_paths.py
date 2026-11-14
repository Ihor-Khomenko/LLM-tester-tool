"""Oberih-named data/docs paths with legacy sentinel compatibility."""

from __future__ import annotations

from pathlib import Path

import yaml

from llm_tester.sentinel.policy import (
    DEFAULT_POLICY_PATH_LEGACY,
    DEFAULT_POLICY_PATH_OBERIH,
    load_policy,
    resolve_default_policy_path,
)
from llm_tester.sentinel.proxy_config import (
    DEFAULT_PROXY_PATH_LEGACY,
    DEFAULT_PROXY_PATH_OBERIH,
    load_proxy_config,
    resolve_default_proxy_path,
)

REPO = Path(__file__).resolve().parents[2]


def test_oberih_and_legacy_policy_files_exist():
    assert DEFAULT_POLICY_PATH_OBERIH.is_file()
    assert DEFAULT_POLICY_PATH_LEGACY.is_file()
    assert (REPO / "llm_tester" / "data" / "oberih" / "policy.default.yaml").is_file()
    assert (REPO / "llm_tester" / "data" / "sentinel" / "policy.default.yaml").is_file()


def test_oberih_and_legacy_proxy_files_exist():
    assert DEFAULT_PROXY_PATH_OBERIH.is_file()
    assert DEFAULT_PROXY_PATH_LEGACY.is_file()
    assert (REPO / "llm_tester" / "data" / "oberih" / "proxy.default.yaml").is_file()
    assert (REPO / "llm_tester" / "data" / "sentinel" / "proxy.default.yaml").is_file()


def test_default_policy_files_load_to_identical_config():
    """Old and new default policy YAML must be byte-identical in effect."""
    oberih = load_policy(DEFAULT_POLICY_PATH_OBERIH)
    legacy = load_policy(DEFAULT_POLICY_PATH_LEGACY)
    assert oberih.to_dict() == legacy.to_dict()
    assert oberih.tau_high == 0.7
    assert oberih.tau_low == 0.25
    assert oberih.weight_s1 == 0.35
    assert oberih.weight_s2 == 0.45
    assert oberih.weight_s3 == 0.20
    assert oberih.s3_backend == "lite"
    assert oberih.enable_s4 is False


def test_default_proxy_files_load_to_identical_config():
    oberih = load_proxy_config(DEFAULT_PROXY_PATH_OBERIH)
    legacy = load_proxy_config(DEFAULT_PROXY_PATH_LEGACY)
    assert oberih.to_dict() == legacy.to_dict()
    assert oberih.threshold == 0.7
    assert oberih.enable_s4 is False
    assert oberih.transport == "stdio"
    assert oberih.t0_mode == "enforce"


def test_resolve_defaults_prefer_oberih():
    assert resolve_default_policy_path() == DEFAULT_POLICY_PATH_OBERIH
    assert resolve_default_proxy_path() == DEFAULT_PROXY_PATH_OBERIH
    # load_policy(None) still works and matches the Oberih file.
    assert load_policy(None).to_dict() == load_policy(DEFAULT_POLICY_PATH_OBERIH).to_dict()


def test_legacy_sentinel_paths_still_loadable_by_string():
    """Research runners / old scripts that hardcode data/sentinel must keep working."""
    pol = load_policy(REPO / "llm_tester" / "data" / "sentinel" / "policy.default.yaml")
    cfg = load_proxy_config(REPO / "llm_tester" / "data" / "sentinel" / "proxy.default.yaml")
    assert pol.tau_high == 0.7
    assert cfg.threshold == 0.7


def test_trace_format_docs_both_paths():
    new_doc = REPO / "docs" / "oberih-trace-format.md"
    old_doc = REPO / "docs" / "mcp-sentinel-trace-format.md"
    assert new_doc.is_file()
    assert old_doc.is_file()
    new_text = new_doc.read_text(encoding="utf-8")
    old_text = old_doc.read_text(encoding="utf-8")
    # Schema id must remain in both docs (analyzers / research interchange).
    assert "mcp-sentinel-trace/v1" in new_text
    assert "mcp-sentinel-trace/v1" in old_text
    assert "oberih-trace-format.md" in old_text


def test_readme_has_no_sentinel_and_links_oberih_paths():
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    assert "sentinel" not in readme.lower()
    assert "llm_tester/data/oberih/policy.default.yaml" in readme
    assert "llm_tester/data/oberih/proxy.default.yaml" in readme
    assert "docs/oberih-trace-format.md" in readme


def test_package_data_includes_both_locations():
    pyproject = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    assert '"data/oberih/*"' in pyproject
    assert '"data/sentinel/*"' in pyproject


def test_legacy_proxy_yaml_threshold_still_0_7():
    path = REPO / "llm_tester" / "data" / "sentinel" / "proxy.default.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert data["threshold"] == 0.7
