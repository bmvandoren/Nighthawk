"""Unit tests for nighthawk/model_manager.py.

All tests are pure-Python (no TF, no real network).  Network calls are
intercepted via monkeypatching _fetch_registry or by serving a file:// URL.
"""

import gzip
import hashlib
import json
import os
import tarfile
import tempfile
from pathlib import Path

import pytest

from nighthawk.model_manager import (
    ChecksumError,
    ModelResolutionError,
    NighthawkModelError,
    OfflineError,
    RegistryError,
    ResolvedModel,
    UnsafeArchiveError,
    _cache_root,
    _cached_bundle_path,
    _is_ready,
    _is_s3_url,
    _load_bundle,
    _newest_cached_version,
    _parse_s3_url,
    _repo_join,
    _resolve_version,
    _s3_client,
    _safe_extract,
    _sha256_file,
    _verify_sha256,
    list_cached_models,
    remove_cached_model,
    resolve_model,
)


# ---------------------------------------------------------------------------
# Helpers — build a minimal valid bundle on disk
# ---------------------------------------------------------------------------

_MANIFEST = {
    "schema_version": 1,
    "name": "americas",
    "version": "0.4.0",
    "created": "2026-07-28T00:00:00Z",
    "package_min_version": "0.4.0",
    "taxonomy_version": "select_v8",
    "num_classes": {"order": 8, "family": 28, "group": 19, "species": 176},
    "calibrated_taxa": 100,
    "export_name": "export_ema",
    "model_sample_rate": 22050,
    "model_input_duration": 1,
    "layout": {
        "saved_model": "saved_model_with_preprocessing",
        "taxonomy": {
            "dir": "taxonomy",
            "species": "species.txt",
            "groups": "groups.txt",
            "families": "families.txt",
            "orders": "orders.txt",
            "ebird_taxonomy": "ebird_taxonomy.csv",
            "group_ebird_codes": "groups_ebird_codes.csv",
        },
        "test_config": {
            "dir": "test_config",
            "config": "test_config.json",
            "calibrators_from_logits": "probability_calibrations_logistic_fromlogits.csv",
            "test_set_performance": "test_set_performance",
        },
    },
}


def _make_bundle_dir(root: Path, manifest: dict = None) -> Path:
    """Create a minimal fake bundle directory under root."""
    if manifest is None:
        manifest = _MANIFEST
    bundle = root / "bundle"
    bundle.mkdir(parents=True, exist_ok=True)

    # manifest.json
    (bundle / "manifest.json").write_text(json.dumps(manifest))

    # SavedModel stub
    sm = bundle / "saved_model_with_preprocessing"
    sm.mkdir()
    (sm / "saved_model.pb").write_bytes(b"stub")

    # Taxonomy
    tax = bundle / "taxonomy"
    tax.mkdir()
    for fname in ["species.txt", "groups.txt", "families.txt", "orders.txt",
                  "ebird_taxonomy.csv", "groups_ebird_codes.csv"]:
        (tax / fname).write_text("stub")

    # test_config
    tc = bundle / "test_config"
    tc.mkdir()
    (tc / "test_config.json").write_text("{}")
    (tc / "probability_calibrations_logistic_fromlogits.csv").write_text("Taxon,A,B\n")
    tsp = tc / "test_set_performance"
    tsp.mkdir()
    (tsp / "taxon_summary_species.csv").write_text("taxon,ap_masked\n")

    return bundle


def _make_tarball(bundle_dir: Path, dest: Path) -> str:
    """Tar the bundle and return the sha256."""
    with tarfile.open(dest, "w:gz") as tar:
        for member in sorted(bundle_dir.rglob("*")):
            arcname = str(member.relative_to(bundle_dir))
            tar.add(member, arcname=arcname, recursive=False)
    return _sha256_file(dest)


# ---------------------------------------------------------------------------
# _cache_root
# ---------------------------------------------------------------------------

def test_cache_root_explicit(tmp_path):
    assert _cache_root(tmp_path) == tmp_path


def test_cache_root_env(tmp_path, monkeypatch):
    monkeypatch.setenv("NIGHTHAWK_CACHE_DIR", str(tmp_path))
    assert _cache_root(None) == tmp_path


def test_cache_root_fallback(tmp_path, monkeypatch):
    monkeypatch.delenv("NIGHTHAWK_CACHE_DIR", raising=False)
    # Just check it returns a Path without error.
    result = _cache_root(None)
    assert isinstance(result, Path)


# ---------------------------------------------------------------------------
# _load_bundle — happy path and legacy fallback
# ---------------------------------------------------------------------------

def test_load_bundle_with_manifest(tmp_path):
    bundle = _make_bundle_dir(tmp_path)
    resolved = _load_bundle(bundle, source="local-dir")
    assert resolved.name == "americas"
    assert resolved.version == "0.4.0"
    assert resolved.source == "local-dir"
    assert resolved.saved_model_dir.exists()
    assert resolved.taxonomy["species"].exists()
    assert resolved.test_config["config"].exists()
    assert resolved.test_config["test_set_performance"].exists()


def test_load_bundle_legacy_no_manifest(tmp_path):
    """A bundle without manifest.json falls back to the legacy layout."""
    bundle = _make_bundle_dir(tmp_path)
    (bundle / "manifest.json").unlink()  # remove manifest
    resolved = _load_bundle(
        bundle, source="local-dir",
        requested_name="americas", requested_version="0.4.0",
    )
    assert resolved.name == "americas"
    assert resolved.saved_model_dir.exists()


# ---------------------------------------------------------------------------
# _verify_sha256
# ---------------------------------------------------------------------------

def test_verify_sha256_ok(tmp_path):
    f = tmp_path / "data.bin"
    f.write_bytes(b"hello world")
    expected = hashlib.sha256(b"hello world").hexdigest()
    _verify_sha256(f, expected)  # should not raise


def test_verify_sha256_mismatch(tmp_path):
    f = tmp_path / "data.bin"
    f.write_bytes(b"hello world")
    with pytest.raises(ChecksumError):
        _verify_sha256(f, "0" * 64)


# ---------------------------------------------------------------------------
# _safe_extract — path traversal defence
# ---------------------------------------------------------------------------

def _make_traversal_tar(dest_tar: Path, member_name: str, content: bytes = b"evil") -> None:
    """Create a tarball with a single member whose name is potentially unsafe."""
    with tarfile.open(dest_tar, "w:gz") as tar:
        import io
        info = tarfile.TarInfo(name=member_name)
        info.size = len(content)
        tar.addfile(info, io.BytesIO(content))


def test_safe_extract_normal(tmp_path):
    bundle = _make_bundle_dir(tmp_path / "src")
    tar_path = tmp_path / "bundle.tar.gz"
    _make_tarball(bundle, tar_path)
    dest = tmp_path / "out"
    dest.mkdir()
    _safe_extract(tar_path, dest)
    assert (dest / "manifest.json").exists()


def test_safe_extract_rejects_absolute_path(tmp_path):
    tar_path = tmp_path / "evil.tar.gz"
    _make_traversal_tar(tar_path, "/etc/evil")
    dest = tmp_path / "out"
    dest.mkdir()
    with pytest.raises(UnsafeArchiveError):
        _safe_extract(tar_path, dest)


def test_safe_extract_rejects_path_traversal(tmp_path):
    tar_path = tmp_path / "evil.tar.gz"
    _make_traversal_tar(tar_path, "../evil")
    dest = tmp_path / "out"
    dest.mkdir()
    with pytest.raises(UnsafeArchiveError):
        _safe_extract(tar_path, dest)


# ---------------------------------------------------------------------------
# _resolve_version
# ---------------------------------------------------------------------------

def _make_registry(name="americas", version="0.4.0", latest="0.4.0"):
    return {
        "schema_version": 1,
        "models": {
            name: {
                "latest": latest,
                "versions": {
                    version: {
                        "url": f"models/{name}/{name}-{version}.tar.gz",
                        "sha256": "a" * 64,
                        "size": 1000,
                        "package_min_version": "0.1.0",
                    }
                },
            }
        },
    }


def test_resolve_version_latest(monkeypatch):
    registry = _make_registry(version="0.4.0", latest="0.4.0")
    v, entry = _resolve_version(registry, "americas", "latest")
    assert v == "0.4.0"
    assert "url" in entry


def test_resolve_version_concrete():
    registry = _make_registry(version="0.4.0")
    v, entry = _resolve_version(registry, "americas", "0.4.0")
    assert v == "0.4.0"


def test_resolve_version_unknown_model():
    registry = _make_registry()
    with pytest.raises(ModelResolutionError):
        _resolve_version(registry, "europe", "1.0.0")


def test_resolve_version_unknown_version():
    registry = _make_registry(version="0.4.0")
    with pytest.raises(ModelResolutionError):
        _resolve_version(registry, "americas", "9.9.9")


# ---------------------------------------------------------------------------
# _newest_cached_version
# ---------------------------------------------------------------------------

def test_newest_cached_version(tmp_path):
    cache = tmp_path / "cache"
    for v in ["0.3.0", "0.4.0", "0.2.0"]:
        d = cache / "models" / "americas" / v
        d.mkdir(parents=True)
        (d / ".ready").write_text("ok")
    result = _newest_cached_version(cache, "americas")
    assert result == "0.4.0"


def test_newest_cached_version_none_when_empty(tmp_path):
    cache = tmp_path / "cache"
    assert _newest_cached_version(cache, "americas") is None


# ---------------------------------------------------------------------------
# resolve_model — local-dir and local-tarball sources
# ---------------------------------------------------------------------------

def test_resolve_model_local_dir(tmp_path):
    bundle = _make_bundle_dir(tmp_path)
    resolved = resolve_model(model_path=bundle)
    assert resolved.source == "local-dir"
    assert resolved.name == "americas"


def test_resolve_model_local_tarball(tmp_path):
    bundle = _make_bundle_dir(tmp_path / "src")
    tar_path = tmp_path / "americas-0.4.0.tar.gz"
    _make_tarball(bundle, tar_path)
    resolved = resolve_model(model_path=tar_path)
    assert resolved.source == "local-tarball"
    assert resolved.name == "americas"


# ---------------------------------------------------------------------------
# resolve_model — cache hit (no network)
# ---------------------------------------------------------------------------

def test_resolve_model_cache_hit(tmp_path, monkeypatch):
    """A cached concrete version must be served without hitting the network."""
    bundle = _make_bundle_dir(tmp_path / "src")
    cache = tmp_path / "cache"
    final = cache / "models" / "americas" / "0.4.0"
    final.mkdir(parents=True)
    # Copy bundle contents into cache.
    import shutil
    for item in bundle.iterdir():
        if item.is_dir():
            shutil.copytree(item, final / item.name)
        else:
            shutil.copy2(item, final / item.name)
    (final / ".ready").write_text("ok")

    # Monkeypatch _fetch_registry to blow up if called.
    import nighthawk.model_manager as mm
    monkeypatch.setattr(mm, "_fetch_registry", lambda url: (_ for _ in ()).throw(
        RuntimeError("_fetch_registry must not be called for a cached concrete version")
    ))

    resolved = resolve_model(
        name="americas", version="0.4.0", cache_dir=cache,
        repo_url="https://example.com/",
    )
    assert resolved.source == "cache"
    assert resolved.version == "0.4.0"


# ---------------------------------------------------------------------------
# resolve_model — offline mode
# ---------------------------------------------------------------------------

def test_resolve_model_offline_no_cache_raises(tmp_path):
    cache = tmp_path / "cache"
    with pytest.raises(OfflineError):
        resolve_model(
            name="americas", version="0.4.0", cache_dir=cache,
            offline=True, repo_url="https://example.com/",
        )


def test_resolve_model_offline_latest_falls_back_to_newest_cached(tmp_path, monkeypatch):
    """Offline + latest -> fall back to newest cached version."""
    bundle = _make_bundle_dir(tmp_path / "src")
    cache = tmp_path / "cache"
    final = cache / "models" / "americas" / "0.4.0"
    final.mkdir(parents=True)
    import shutil
    for item in bundle.iterdir():
        if item.is_dir():
            shutil.copytree(item, final / item.name)
        else:
            shutil.copy2(item, final / item.name)
    (final / ".ready").write_text("ok")

    resolved = resolve_model(
        name="americas", version="latest", cache_dir=cache,
        offline=True, repo_url="https://example.com/",
    )
    assert resolved.version == "0.4.0"
    assert resolved.source == "cache"


def test_resolve_model_offline_latest_no_cache_raises(tmp_path):
    cache = tmp_path / "cache"
    with pytest.raises(OfflineError):
        resolve_model(
            name="americas", version="latest", cache_dir=cache,
            offline=True, repo_url="https://example.com/",
        )


# ---------------------------------------------------------------------------
# list_cached_models / remove_cached_model
# ---------------------------------------------------------------------------

def test_list_cached_models(tmp_path):
    cache = tmp_path / "cache"
    for v in ["0.3.0", "0.4.0"]:
        d = cache / "models" / "americas" / v
        d.mkdir(parents=True)
        (d / ".ready").write_text("ok")
    result = list_cached_models(cache_dir=cache)
    assert len(result) == 2
    versions = {m["version"] for m in result}
    assert versions == {"0.3.0", "0.4.0"}


def test_remove_cached_model_specific_version(tmp_path):
    cache = tmp_path / "cache"
    for v in ["0.3.0", "0.4.0"]:
        d = cache / "models" / "americas" / v
        d.mkdir(parents=True)
        (d / ".ready").write_text("ok")
    removed = remove_cached_model("americas", version="0.3.0", cache_dir=cache)
    assert len(removed) == 1
    remaining = list_cached_models(cache_dir=cache)
    assert len(remaining) == 1
    assert remaining[0]["version"] == "0.4.0"


def test_remove_cached_model_all(tmp_path):
    cache = tmp_path / "cache"
    for v in ["0.3.0", "0.4.0"]:
        d = cache / "models" / "americas" / v
        d.mkdir(parents=True)
        (d / ".ready").write_text("ok")
    remove_cached_model("americas", cache_dir=cache)
    assert list_cached_models(cache_dir=cache) == []


# ---------------------------------------------------------------------------
# Legacy model_type support — _load_bundle + _get_configuration_file_paths
# ---------------------------------------------------------------------------

_LEGACY_MANIFEST = {
    **{k: v for k, v in _MANIFEST.items() if k != "model_type"},
    "name": "300-americas",
    "version": "1.0.0",
    "model_type": "legacy",
    "taxonomy_version": "select_v6",
    "num_classes": {"order": 4, "family": 19, "group": 17, "species": 130},
}


def _make_legacy_bundle_dir(root: Path) -> Path:
    """Create a minimal fake legacy bundle (select_v6 species count)."""
    return _make_bundle_dir(root, manifest=_LEGACY_MANIFEST)


def test_load_bundle_legacy_model_type(tmp_path):
    """Bundle with model_type='legacy' round-trips through _load_bundle."""
    bundle = _make_legacy_bundle_dir(tmp_path)
    resolved = _load_bundle(bundle, source="local-dir")
    assert resolved.name == "300-americas"
    assert resolved.version == "1.0.0"
    assert resolved.manifest.get("model_type") == "legacy"


def test_load_bundle_legacy_model_type_in_resolved_manifest(tmp_path):
    """model_type is accessible on resolved.manifest for detector dispatch."""
    bundle = _make_legacy_bundle_dir(tmp_path)
    resolved = _load_bundle(bundle, source="local-dir")
    model_type = (resolved.manifest or {}).get("model_type", "nh2")
    assert model_type == "legacy"


def test_load_bundle_nh2_model_type_default(tmp_path):
    """nh2 bundle without explicit model_type defaults to 'nh2' on lookup."""
    # Standard _MANIFEST has no model_type key — simulate the default behaviour.
    manifest_no_type = {k: v for k, v in _MANIFEST.items() if k != "model_type"}
    bundle = _make_bundle_dir(tmp_path, manifest=manifest_no_type)
    resolved = _load_bundle(bundle, source="local-dir")
    model_type = (resolved.manifest or {}).get("model_type", "nh2")
    assert model_type == "nh2"


def test_load_bundle_manifest_less_model_type_defaults_nh2(tmp_path):
    """A manifest-less bundle (legacy fallback path) also defaults to 'nh2'."""
    bundle = _make_bundle_dir(tmp_path)
    (bundle / "manifest.json").unlink()
    resolved = _load_bundle(
        bundle, source="local-dir",
        requested_name="americas", requested_version="0.4.0",
    )
    # resolved.manifest is empty dict for manifest-less bundles.
    model_type = (resolved.manifest or {}).get("model_type", "nh2")
    assert model_type == "nh2"


# ---------------------------------------------------------------------------
# S3 helpers — _is_s3_url, _parse_s3_url, _repo_join
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("url,expected", [
    ("s3://bucket/prefix/", True),
    ("s3://bucket", True),
    ("https://bucket.s3.amazonaws.com/", False),
    ("http://example.com/registry.json", False),
    ("", False),
])
def test_is_s3_url(url, expected):
    assert _is_s3_url(url) == expected


@pytest.mark.parametrize("url,bucket,key", [
    ("s3://mybucket/prefix/", "mybucket", "prefix/"),
    ("s3://mybucket/deep/prefix/", "mybucket", "deep/prefix/"),
    ("s3://mybucket/prefix/registry.json", "mybucket", "prefix/registry.json"),
    ("s3://mybucket", "mybucket", ""),
    ("s3://mybucket/", "mybucket", ""),
])
def test_parse_s3_url_valid(url, bucket, key):
    b, k = _parse_s3_url(url)
    assert b == bucket
    assert k == key


def test_parse_s3_url_missing_bucket():
    with pytest.raises(ModelResolutionError, match="missing bucket"):
        _parse_s3_url("s3:///prefix/")


@pytest.mark.parametrize("repo_url,relative,expected", [
    # S3 with prefix
    ("s3://bucket/prefix/", "registry.json", "s3://bucket/prefix/registry.json"),
    ("s3://bucket/prefix/", "models/americas/americas-0.4.0.tar.gz",
     "s3://bucket/prefix/models/americas/americas-0.4.0.tar.gz"),
    # S3 no prefix
    ("s3://bucket", "registry.json", "s3://bucket/registry.json"),
    ("s3://bucket/", "registry.json", "s3://bucket/registry.json"),
    # HTTPS — matches existing urllib.parse.urljoin behaviour
    ("https://example.s3.amazonaws.com/", "registry.json",
     "https://example.s3.amazonaws.com/registry.json"),
    ("https://example.s3.amazonaws.com/", "models/americas/americas-0.4.0.tar.gz",
     "https://example.s3.amazonaws.com/models/americas/americas-0.4.0.tar.gz"),
])
def test_repo_join(repo_url, relative, expected):
    assert _repo_join(repo_url, relative) == expected


# ---------------------------------------------------------------------------
# _s3_client — missing boto3 produces NighthawkModelError
# ---------------------------------------------------------------------------

def test_s3_client_missing_boto3(monkeypatch):
    """When boto3 is absent the error message suggests the install command."""
    import builtins
    real_import = builtins.__import__

    def _block_boto3(name, *args, **kwargs):
        if name == "boto3":
            raise ImportError("no module named boto3")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _block_boto3)
    with pytest.raises(NighthawkModelError, match="pip install 'nighthawk\\[s3\\]'"):
        _s3_client()


# ---------------------------------------------------------------------------
# _fetch_registry over S3 — monkeypatched boto3 client
# ---------------------------------------------------------------------------

def _fake_s3_client_for_registry(registry_bytes: bytes):
    """Return a minimal fake boto3 S3 client that serves registry_bytes."""
    import io

    class _FakeBody:
        def read(self):
            return registry_bytes

    class _FakeS3Client:
        def get_object(self, Bucket, Key):
            return {"Body": _FakeBody()}

    return _FakeS3Client()


def test_fetch_registry_s3_success(monkeypatch, tmp_path):
    """_fetch_registry fetches and parses registry.json from S3."""
    import nighthawk.model_manager as mm

    registry = _make_registry()
    registry_bytes = json.dumps(registry).encode()
    fake_client = _fake_s3_client_for_registry(registry_bytes)
    monkeypatch.setattr(mm, "_s3_client", lambda: fake_client)

    result = mm._fetch_registry("s3://my-bucket/nighthawk/")
    assert result["schema_version"] == 1
    assert "americas" in result["models"]


def test_fetch_registry_s3_client_error(monkeypatch):
    """A botocore ClientError from the S3 fetch is mapped to RegistryError."""
    import nighthawk.model_manager as mm

    try:
        from botocore.exceptions import ClientError
    except ImportError:
        pytest.skip("botocore not installed")

    class _ErrorClient:
        def get_object(self, Bucket, Key):
            raise ClientError(
                {"Error": {"Code": "NoSuchKey", "Message": "Not found"}},
                "GetObject",
            )

    monkeypatch.setattr(mm, "_s3_client", lambda: _ErrorClient())
    with pytest.raises(RegistryError, match="NoSuchKey"):
        mm._fetch_registry("s3://my-bucket/nighthawk/")


def test_fetch_registry_s3_no_credentials_error(monkeypatch):
    """A botocore NoCredentialsError is mapped to RegistryError with a hint."""
    import nighthawk.model_manager as mm

    try:
        from botocore.exceptions import NoCredentialsError
    except ImportError:
        pytest.skip("botocore not installed")

    class _NoCreds:
        def get_object(self, Bucket, Key):
            raise NoCredentialsError()

    monkeypatch.setattr(mm, "_s3_client", lambda: _NoCreds())
    with pytest.raises(RegistryError, match="No AWS credentials"):
        mm._fetch_registry("s3://my-bucket/nighthawk/")


# ---------------------------------------------------------------------------
# resolve_model over S3 — end-to-end with monkeypatched boto3 client
# ---------------------------------------------------------------------------

def test_resolve_model_s3_download(tmp_path, monkeypatch):
    """resolve_model downloads + caches a bundle via S3 when repo_url is s3://."""
    import shutil
    import nighthawk.model_manager as mm

    # Build a real tarball from the fixture bundle; _make_tarball returns the
    # sha256 so we can put it in the registry and pass checksum verification.
    bundle = _make_bundle_dir(tmp_path / "src")
    tar_path = tmp_path / "americas-0.4.0.tar.gz"
    actual_sha = _make_tarball(bundle, tar_path)
    registry = _make_registry()
    registry["models"]["americas"]["versions"]["0.4.0"]["sha256"] = actual_sha
    registry_bytes = json.dumps(registry).encode()
    tar_bytes = tar_path.read_bytes()

    # Fake S3 client: serves registry.json and the tarball.
    class _FakeS3Client:
        def get_object(self, Bucket, Key):
            import io
            if Key.endswith("registry.json"):
                data = registry_bytes
            else:
                data = tar_bytes

            class _Body:
                def read(self):
                    return data

            return {"Body": _Body()}

        def download_file(self, Bucket, Key, Filename):
            # Simulate boto3 download_file by writing the tarball to Filename.
            import shutil as _shutil
            with open(Filename, "wb") as fh:
                fh.write(tar_bytes)

    monkeypatch.setattr(mm, "_s3_client", lambda: _FakeS3Client())

    cache = tmp_path / "cache"
    resolved = resolve_model(
        name="americas",
        version="0.4.0",
        repo_url="s3://my-bucket/nighthawk/",
        cache_dir=cache,
    )
    assert resolved.source == "download"
    assert resolved.name == "americas"
    assert resolved.version == "0.4.0"
    # Should be cached now.
    assert _is_ready(_cached_bundle_path(cache, "americas", "0.4.0"))


def test_resolve_model_s3_cache_hit_skips_boto(tmp_path, monkeypatch):
    """A cached S3 model is served from disk — boto3 must not be called."""
    import shutil
    import nighthawk.model_manager as mm

    # Populate the cache.
    bundle = _make_bundle_dir(tmp_path / "src")
    cache = tmp_path / "cache"
    final = cache / "models" / "americas" / "0.4.0"
    final.mkdir(parents=True)
    for item in bundle.iterdir():
        if item.is_dir():
            shutil.copytree(item, final / item.name)
        else:
            shutil.copy2(item, final / item.name)
    (final / ".ready").write_text("ok")

    def _must_not_call():
        raise AssertionError("_s3_client must not be called for a cache hit")

    monkeypatch.setattr(mm, "_s3_client", _must_not_call)

    resolved = resolve_model(
        name="americas",
        version="0.4.0",
        repo_url="s3://my-bucket/nighthawk/",
        cache_dir=cache,
    )
    assert resolved.source == "cache"
