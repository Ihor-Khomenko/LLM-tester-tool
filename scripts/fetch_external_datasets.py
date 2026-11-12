#!/usr/bin/env python3
"""Fetch pinned external corpora for detector / FP evaluation (checksum verified).

CI must never run this (network + downloads). Local / research hosts only::

    python scripts/fetch_external_datasets.py
    python scripts/fetch_external_datasets.py --datasets injecagent,bipia
    python scripts/fetch_external_datasets.py --deepset   # HF datasets API

Checksums and commit pins live in
``llm_tester/data/external_datasets/manifest.yaml``. NewsQA / XSum BIPIA
contexts are intentionally omitted (upstream license requires local generation).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "llm_tester" / "data" / "external_datasets" / "manifest.yaml"
DEFAULT_CACHE = Path(".cache") / "llm_tester" / "external"


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit(
            "PyYAML required: pip install PyYAML  (or llm-tester[dev])"
        ) from exc
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _download(url: str, timeout: int = 120) -> bytes:
    req = urllib.request.Request(
        url, headers={"User-Agent": "llm-tester-fetch-external/0.1"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def fetch_file(
    *,
    url: str,
    dest: Path,
    expected_sha256: Optional[str],
    force: bool = False,
    dry_run: bool = False,
) -> Dict[str, Any]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and not force:
        data = dest.read_bytes()
        digest = _sha256(data)
        ok = expected_sha256 is None or digest == expected_sha256
        return {
            "path": str(dest),
            "bytes": len(data),
            "sha256": digest,
            "verified": ok,
            "action": "cached",
            "error": None if ok else f"checksum mismatch (expected {expected_sha256})",
        }
    if dry_run:
        return {
            "path": str(dest),
            "bytes": 0,
            "sha256": expected_sha256,
            "verified": None,
            "action": "would_fetch",
            "error": None,
        }
    data = _download(url)
    digest = _sha256(data)
    if expected_sha256 and digest != expected_sha256:
        return {
            "path": str(dest),
            "bytes": len(data),
            "sha256": digest,
            "verified": False,
            "action": "fetch_failed",
            "error": f"checksum mismatch (expected {expected_sha256}, got {digest})",
        }
    dest.write_bytes(data)
    return {
        "path": str(dest),
        "bytes": len(data),
        "sha256": digest,
        "verified": True,
        "action": "fetched",
        "error": None,
    }


def fetch_manifest_files(
    manifest: Dict[str, Any],
    *,
    cache_dir: Path,
    datasets: Optional[Sequence[str]] = None,
    force: bool = False,
    dry_run: bool = False,
) -> List[Dict[str, Any]]:
    wanted = {d.lower() for d in datasets} if datasets else None
    results: List[Dict[str, Any]] = []
    for name, meta in (manifest.get("datasets") or {}).items():
        if wanted is not None and name.lower() not in wanted:
            continue
        for fmeta in meta.get("files") or []:
            dest = cache_dir / str(fmeta["path"])
            r = fetch_file(
                url=str(fmeta["url"]),
                dest=dest,
                expected_sha256=fmeta.get("sha256"),
                force=force,
                dry_run=dry_run,
            )
            r["dataset"] = name
            r["id"] = fmeta.get("id")
            r["license"] = meta.get("license")
            r["commit"] = meta.get("commit") or meta.get("revision")
            results.append(r)
    return results


def fetch_deepset(
    manifest: Dict[str, Any],
    *,
    cache_dir: Path,
    force: bool = False,
    dry_run: bool = False,
) -> Dict[str, Any]:
    meta = (manifest.get("datasets") or {}).get("deepset_prompt_injections") or {}
    rev = meta.get("revision")
    hf_id = meta.get("hf_id") or "deepset/prompt-injections"
    out_dir = cache_dir / "deepset_prompt_injections"
    marker = out_dir / "REVISION.txt"
    if dry_run:
        return {
            "dataset": "deepset_prompt_injections",
            "action": "would_fetch_hf",
            "hf_id": hf_id,
            "revision": rev,
            "path": str(out_dir),
        }
    if marker.is_file() and not force and marker.read_text().strip() == str(rev):
        return {
            "dataset": "deepset_prompt_injections",
            "action": "cached",
            "hf_id": hf_id,
            "revision": rev,
            "path": str(out_dir),
            "verified": True,
        }
    try:
        from datasets import load_dataset
    except ImportError as exc:
        return {
            "dataset": "deepset_prompt_injections",
            "action": "fetch_failed",
            "error": f"datasets package missing: {exc}",
            "hf_id": hf_id,
        }
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(hf_id, revision=rev, cache_dir=str(cache_dir / "hf"))
    # Materialize train+test JSONL for offline reuse without re-hitting HF.
    for split in ds.keys():
        path = out_dir / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for row in ds[split]:
                fh.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
    meta_out = {
        "hf_id": hf_id,
        "revision": rev,
        "license": meta.get("license"),
        "source": meta.get("source"),
        "splits": list(ds.keys()),
    }
    (out_dir / "meta.json").write_text(
        json.dumps(meta_out, indent=2) + "\n", encoding="utf-8"
    )
    marker.write_text(str(rev) + "\n", encoding="utf-8")
    return {
        "dataset": "deepset_prompt_injections",
        "action": "fetched",
        "hf_id": hf_id,
        "revision": rev,
        "path": str(out_dir),
        "verified": True,
        "license": meta.get("license"),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Path to manifest.yaml",
    )
    ap.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE,
        help="Download destination (default .cache/llm_tester/external)",
    )
    ap.add_argument(
        "--datasets",
        default="injecagent,bipia",
        help="Comma-separated dataset keys from the manifest (default injecagent,bipia)",
    )
    ap.add_argument(
        "--deepset",
        action="store_true",
        help="Also fetch deepset/prompt-injections via Hugging Face datasets",
    )
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(list(argv) if argv is not None else None)

    if not args.manifest.is_file():
        print(f"Manifest not found: {args.manifest}", file=sys.stderr)
        return 2
    manifest = _load_yaml(args.manifest)
    names = [x.strip() for x in args.datasets.split(",") if x.strip()]
    # deepset has no raw files in the manifest — strip it from URL fetch list.
    url_names = [n for n in names if n != "deepset_prompt_injections"]
    results = fetch_manifest_files(
        manifest,
        cache_dir=args.cache_dir,
        datasets=url_names,
        force=args.force,
        dry_run=args.dry_run,
    )
    if args.deepset or "deepset_prompt_injections" in names:
        results.append(
            fetch_deepset(
                manifest,
                cache_dir=args.cache_dir,
                force=args.force,
                dry_run=args.dry_run,
            )
        )

    failed = [r for r in results if r.get("error") or r.get("verified") is False]
    if args.json:
        print(json.dumps({"results": results, "failed": len(failed)}, indent=2))
    else:
        for r in results:
            status = "OK" if r.get("verified") is not False and not r.get("error") else "FAIL"
            print(
                f"[{status}] {r.get('dataset')}/{r.get('id') or r.get('hf_id') or ''} "
                f"{r.get('action')} bytes={r.get('bytes', '—')} "
                f"sha={str(r.get('sha256') or '')[:12]}… "
                f"{r.get('error') or ''}"
            )
        print(f"cache: {args.cache_dir.resolve()}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
