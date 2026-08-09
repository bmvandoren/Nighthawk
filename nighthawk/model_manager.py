"""Nighthawk model resolution, download, and cache management.

Models are identified by (name, version) — e.g. ``("americas", "0.4.0")``.
A *registry* JSON file hosted at the configured repo URL lists all published
models and their download metadata.  Downloaded bundles are verified with
SHA-256 and cached locally so repeat calls never re-download.

Public API
----------
resolve_model(name, version, model_path, repo_url, cache_dir, offline)
    -> ResolvedModel

nighthawk-models CLI is in models_cli.py.

Environment overrides
---------------------
NIGHTHAWK_CACHE_DIR   Override the default cache directory.
NIGHTHAWK_OFFLINE     Set to "1" to disable all network access.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tarfile
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class NighthawkModelError(Exception):
    """Base class for all model-manager errors."""


class ModelResolutionError(NighthawkModelError):
    """Could not locate or download a model bundle."""


class OfflineError(NighthawkModelError):
    """Network required but offline mode is active."""


class RegistryError(NighthawkModelError):
    """Could not fetch or parse the model registry."""


class ChecksumError(NighthawkModelError):
    """Downloaded archive failed SHA-256 verification."""


class UnsafeArchiveError(NighthawkModelError):
    """Archive contains unsafe paths (path-traversal attack mitigation)."""


# ---------------------------------------------------------------------------
# Public data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ResolvedModel:
    """Fully resolved paths for a Nighthawk model bundle.

    Attributes
    ----------
    name : str
        Model name (e.g. "americas").
    version : str
        Concrete version string (never "latest").
    root : Path
        Root directory of the extracted/local bundle.
    saved_model_dir : Path
        Directory to pass to ``tf.saved_model.load``.
    taxonomy : dict[str, Path]
        Mapping of taxonomy role names to concrete file/dir Paths:
        species, groups, families, orders, ebird_taxonomy, group_ebird_codes.
    test_config : dict[str, Path]
        Mapping: config, calibrators_from_logits, test_set_performance.
    manifest : dict
        Parsed manifest.json for the bundle.
    source : str
        One of: "local-dir", "local-tarball", "cache", "download".
    """
    name: str
    version: str
    root: Path
    saved_model_dir: Path
    taxonomy: dict
    test_config: dict
    manifest: dict
    source: str


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def resolve_model(
    name: str | None = None,
    version: str | None = None,
    model_path: str | Path | None = None,
    repo_url: str | None = None,
    cache_dir: str | Path | None = None,
    offline: bool = False,
) -> ResolvedModel:
    """Resolve a Nighthawk model to a set of concrete file paths.

    Resolution order
    ----------------
    1. ``model_path`` (local dir or tarball) — no network access.
    2. Concrete version already in cache (``.ready`` marker present).
    3. Download from the registry.

    For ``version="latest"`` or ``version=None``, the registry is consulted
    to determine the concrete version (unless ``offline=True``, which falls
    back to the newest cached version of that model name).

    Parameters
    ----------
    name : str, optional
        Model name.  Defaults to DEFAULT_MODEL_NAME from detector.py.
    version : str, optional
        Version or "latest".  Defaults to DEFAULT_MODEL_VERSION.
    model_path : str or Path, optional
        Path to a local bundle directory or tarball.  When given, ``name``
        and ``version`` are read from its manifest.json and the registry is
        never consulted.
    repo_url : str, optional
        Base URL of the model repository.  Defaults to DEFAULT_MODEL_REPO_URL.
    cache_dir : str or Path, optional
        Override for the local cache directory.
    offline : bool
        Never access the network.  Concrete cached versions are served as
        normal; "latest"/uncached raises OfflineError.

    Returns
    -------
    ResolvedModel
    """
    # Import defaults lazily to avoid circular at module level.
    from nighthawk.detector import (
        DEFAULT_MODEL_NAME,
        DEFAULT_MODEL_REPO_URL,
        DEFAULT_MODEL_VERSION,
    )

    offline = offline or (os.environ.get("NIGHTHAWK_OFFLINE", "0") == "1")

    # --- 1. Local bundle path (highest priority) ---
    if model_path is not None:
        p = Path(model_path)
        if p.is_file():
            return _load_from_tarball(p, name, version)
        elif p.is_dir():
            return _load_bundle(p, source="local-dir", requested_name=name,
                                requested_version=version)
        else:
            raise ModelResolutionError(
                f"--model-path '{model_path}' does not exist."
            )

    # Resolve defaults.
    name = name or DEFAULT_MODEL_NAME
    version = version or DEFAULT_MODEL_VERSION
    repo_url = repo_url or DEFAULT_MODEL_REPO_URL
    cache_root = _cache_root(cache_dir)

    needs_registry = (version == "latest")

    # --- 2. Concrete version pinned and cached ---
    if not needs_registry:
        cached = _cached_bundle_path(cache_root, name, version)
        if _is_ready(cached):
            return _load_bundle(cached, source="cache")

    # --- 3. Need the registry ---
    if offline:
        if needs_registry:
            # Fall back to newest cached version.
            cached_v = _newest_cached_version(cache_root, name)
            if cached_v is None:
                raise OfflineError(
                    f"Cannot resolve model '{name}@latest' offline: no cached "
                    f"versions found. Connect to the internet or pin a version."
                )
            return _load_bundle(
                _cached_bundle_path(cache_root, name, cached_v),
                source="cache"
            )
        else:
            raise OfflineError(
                f"Model '{name}@{version}' is not cached and offline mode is "
                f"active. Connect or use --model-path."
            )

    # --- 4. Fetch registry, resolve version, download if needed ---
    registry = _fetch_registry(repo_url)
    concrete_version, entry = _resolve_version(registry, name, version)

    # Check cache after resolving "latest" to a concrete version.
    cached = _cached_bundle_path(cache_root, name, concrete_version)
    if _is_ready(cached):
        return _load_bundle(cached, source="cache")

    bundle_dir = _download_and_cache(
        name, concrete_version, entry, repo_url, cache_root
    )
    return _load_bundle(bundle_dir, source="download")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def fetch_registry(repo_url: str) -> dict:
    """Fetch and return the parsed registry.json from the repository."""
    return _fetch_registry(repo_url)


def _fetch_registry(repo_url: str) -> dict:
    base = repo_url if repo_url.endswith("/") else repo_url + "/"
    url = urllib.parse.urljoin(base, "registry.json")
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as e:
        raise RegistryError(
            f"Failed to fetch registry from {url}: HTTP {e.code} {e.reason}"
        ) from e
    except urllib.error.URLError as e:
        raise RegistryError(
            f"Failed to fetch registry from {url}: {e.reason}"
        ) from e
    try:
        registry = json.loads(raw)
    except json.JSONDecodeError as e:
        raise RegistryError(f"Registry JSON is malformed: {e}") from e

    schema = registry.get("schema_version", 0)
    if schema > 1:
        print(
            f"[nighthawk] WARNING: registry schema_version={schema} is newer "
            "than this client supports (1). Some entries may be unreadable."
        )
    return registry


def _resolve_version(
    registry: dict, name: str, version: str
) -> tuple[str, dict]:
    """Return (concrete_version, entry_dict) from the registry."""
    models = registry.get("models", {})
    if name not in models:
        available = list(models.keys())
        raise ModelResolutionError(
            f"Model '{name}' not found in registry. "
            f"Available models: {available or '(none)'}"
        )
    model_entry = models[name]
    versions = model_entry.get("versions", {})

    if version == "latest":
        concrete = model_entry.get("latest")
        if concrete is None:
            raise RegistryError(
                f"Registry has no 'latest' pointer for model '{name}'."
            )
        if concrete not in versions:
            raise RegistryError(
                f"Registry 'latest' for '{name}' points to version "
                f"'{concrete}' which is not in the versions dict."
            )
    else:
        concrete = version
        if concrete not in versions:
            available = list(versions.keys())
            raise ModelResolutionError(
                f"Version '{version}' of model '{name}' not found in registry. "
                f"Available: {available or '(none)'}"
            )

    entry = versions[concrete]

    # Check package_min_version.
    pkg_min = entry.get("package_min_version")
    if pkg_min:
        _check_package_version(pkg_min, name, concrete)

    return concrete, entry


def _check_package_version(required: str, name: str, version: str) -> None:
    try:
        import nighthawk as _nh
        current_str = getattr(_nh, "__version__", None)
    except Exception:
        current_str = None
    if current_str is None:
        return  # Cannot determine version — allow.
    if _version_tuple(current_str) < _version_tuple(required):
        raise ModelResolutionError(
            f"Model '{name}@{version}' requires nighthawk>={required} but "
            f"you have {current_str}. Please upgrade: pip install --upgrade nighthawk"
        )


def _version_tuple(v: str) -> tuple:
    """Parse a simple X.Y.Z version string into a comparable tuple."""
    try:
        return tuple(int(x) for x in v.split(".")[:3])
    except Exception:
        return (0, 0, 0)


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_root(cache_dir: str | Path | None) -> Path:
    if cache_dir is not None:
        return Path(cache_dir)
    env = os.environ.get("NIGHTHAWK_CACHE_DIR")
    if env:
        return Path(env)
    try:
        import platformdirs
        return Path(platformdirs.user_cache_dir("nighthawk"))
    except ImportError:
        pass
    return Path.home() / ".cache" / "nighthawk"


def _cached_bundle_path(cache_root: Path, name: str, version: str) -> Path:
    return cache_root / "models" / name / version


def _is_ready(bundle_dir: Path) -> bool:
    return (bundle_dir / ".ready").exists()


def _newest_cached_version(cache_root: Path, name: str) -> str | None:
    """Return the newest cached+ready version dir name for a model, or None."""
    model_dir = cache_root / "models" / name
    if not model_dir.is_dir():
        return None
    candidates = [
        d.name for d in model_dir.iterdir()
        if d.is_dir() and _is_ready(d)
    ]
    if not candidates:
        return None
    # Sort by parsed version tuple descending.
    candidates.sort(key=_version_tuple, reverse=True)
    return candidates[0]


def list_cached_models(cache_dir: str | Path | None = None) -> list[dict]:
    """Return a list of {name, version, root} dicts for all cached bundles."""
    cache_root = _cache_root(cache_dir)
    models_dir = cache_root / "models"
    results = []
    if not models_dir.is_dir():
        return results
    for name_dir in sorted(models_dir.iterdir()):
        if not name_dir.is_dir():
            continue
        for ver_dir in sorted(name_dir.iterdir()):
            if ver_dir.is_dir() and _is_ready(ver_dir):
                results.append({
                    "name": name_dir.name,
                    "version": ver_dir.name,
                    "root": ver_dir,
                })
    return results


def remove_cached_model(
    name: str,
    version: str | None = None,
    cache_dir: str | Path | None = None,
) -> list[str]:
    """Remove one or all cached versions of a model.

    Returns a list of removed paths (as strings).
    """
    cache_root = _cache_root(cache_dir)
    removed = []
    if version is not None:
        target = _cached_bundle_path(cache_root, name, version)
        if target.is_dir():
            shutil.rmtree(target)
            removed.append(str(target))
    else:
        model_dir = cache_root / "models" / name
        if model_dir.is_dir():
            shutil.rmtree(model_dir)
            removed.append(str(model_dir))
    return removed


# ---------------------------------------------------------------------------
# Download + extract
# ---------------------------------------------------------------------------

_CHUNK = 256 * 1024  # 256 KB


def _download_and_cache(
    name: str,
    version: str,
    entry: dict,
    repo_url: str,
    cache_root: Path,
) -> Path:
    """Download, verify, extract, and atomically cache a model bundle.

    Returns the final cache directory path.
    """
    base = repo_url if repo_url.endswith("/") else repo_url + "/"
    tar_url = urllib.parse.urljoin(base, entry["url"])
    expected_sha = entry.get("sha256")

    tmp_dir = cache_root / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Unique temp filename to allow parallel downloads without collision.
    tmp_tar = tmp_dir / f"{name}-{version}-{os.getpid()}-{int(time.time())}.tar.gz"

    try:
        print(f"[nighthawk] Downloading {name}@{version} from {tar_url} ...")
        _stream_download(tar_url, tmp_tar)

        if expected_sha:
            print("[nighthawk] Verifying checksum...")
            _verify_sha256(tmp_tar, expected_sha)

        # Extract to a unique temp dir on the same filesystem.
        tmp_extract = tmp_dir / f"extract-{name}-{version}-{os.getpid()}-{int(time.time())}"
        tmp_extract.mkdir(parents=True, exist_ok=True)
        try:
            print("[nighthawk] Extracting...")
            _safe_extract(tmp_tar, tmp_extract)

            # Validate the extracted bundle.
            manifest = _read_manifest(tmp_extract)
            _validate_manifest(manifest, name, version)

            # Write the .ready marker before moving.
            (tmp_extract / ".ready").write_text("ok")

            # Atomic publish: move temp dir to final location.
            final_dir = _cached_bundle_path(cache_root, name, version)
            final_dir.parent.mkdir(parents=True, exist_ok=True)

            if _is_ready(final_dir):
                # Another process won the race; use their copy.
                shutil.rmtree(tmp_extract)
            else:
                # On POSIX, os.replace fails if target exists and is a non-empty
                # directory.  Remove any stale (non-ready) target first.
                if final_dir.exists():
                    shutil.rmtree(final_dir)
                os.replace(tmp_extract, final_dir)

            print(f"[nighthawk] Model cached at {final_dir}")
            return final_dir

        except Exception:
            shutil.rmtree(tmp_extract, ignore_errors=True)
            raise

    finally:
        tmp_tar.unlink(missing_ok=True)


def _stream_download(url: str, dest: Path) -> None:
    """Stream a URL to a local file with a connection timeout."""
    try:
        with urllib.request.urlopen(url, timeout=30) as resp, open(dest, "wb") as f:
            while True:
                chunk = resp.read(_CHUNK)
                if not chunk:
                    break
                f.write(chunk)
    except urllib.error.HTTPError as e:
        raise ModelResolutionError(
            f"HTTP {e.code} downloading {url}: {e.reason}"
        ) from e
    except urllib.error.URLError as e:
        raise ModelResolutionError(
            f"Failed to download {url}: {e.reason}"
        ) from e


def _sha256_file(path: Path) -> str:
    """Return the lowercase hex SHA-256 digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(_CHUNK)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _verify_sha256(path: Path, expected: str) -> None:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(_CHUNK)
            if not chunk:
                break
            h.update(chunk)
    got = h.hexdigest()
    if got != expected.lower():
        raise ChecksumError(
            f"SHA-256 mismatch for {path.name}:\n"
            f"  expected: {expected}\n"
            f"  got:      {got}\n"
            "The download may be corrupted or tampered. Refusing to use."
        )


def _safe_extract(tar_path: Path, dest: Path) -> None:
    """Extract a tarball with path-traversal protection.

    Uses tarfile data_filter when available (Python 3.12+ / backported
    3.9.17+/3.10.12+/3.11.4+). Falls back to manual member vetting on
    older runtimes — do NOT remove this fallback.
    """
    with tarfile.open(tar_path, "r:gz") as tar:
        if hasattr(tarfile, "data_filter"):
            # Python 3.12+ (or patched older releases with the backport).
            tar.extractall(dest, filter="data")
        else:
            _safe_extract_manual(tar, dest)


def _safe_extract_manual(tar: tarfile.TarFile, dest: Path) -> None:
    """Manual path-traversal defence for Python 3.10 / 3.11 without backport."""
    real_dest = os.path.realpath(dest)
    vetted = []
    for member in tar.getmembers():
        # Reject absolute paths.
        if os.path.isabs(member.name):
            raise UnsafeArchiveError(
                f"Archive member has absolute path: {member.name!r}"
            )
        # Reject path traversal.
        target = os.path.realpath(os.path.join(real_dest, member.name))
        try:
            common = os.path.commonpath([real_dest, target])
        except ValueError:
            # Different drives on Windows — clearly unsafe.
            raise UnsafeArchiveError(
                f"Archive member escapes destination: {member.name!r}"
            )
        if common != real_dest:
            raise UnsafeArchiveError(
                f"Archive member escapes destination: {member.name!r}"
            )
        # Reject links that escape.
        if member.issym() or member.islnk():
            link_target = os.path.realpath(
                os.path.join(real_dest, os.path.dirname(member.name), member.linkname)
            )
            try:
                common = os.path.commonpath([real_dest, link_target])
            except ValueError:
                raise UnsafeArchiveError(
                    f"Archive symlink escapes destination: {member.name!r}"
                )
            if common != real_dest:
                raise UnsafeArchiveError(
                    f"Archive symlink escapes destination: {member.name!r}"
                )
        # Reject device/fifo/special files.
        if member.type not in (
            tarfile.REGTYPE, tarfile.AREGTYPE, tarfile.DIRTYPE,
            tarfile.SYMTYPE, tarfile.LNKTYPE,
        ):
            raise UnsafeArchiveError(
                f"Archive member is a special file: {member.name!r} (type {member.type!r})"
            )
        vetted.append(member)
    tar.extractall(dest, members=vetted)


# ---------------------------------------------------------------------------
# Bundle loading
# ---------------------------------------------------------------------------

def _load_from_tarball(
    tar_path: Path,
    requested_name: str | None,
    requested_version: str | None,
) -> ResolvedModel:
    """Extract a local tarball to the cache tmp dir and load it."""
    from nighthawk.detector import DEFAULT_MODEL_REPO_URL  # noqa — for cache root only
    cache_root = _cache_root(None)
    tmp_dir = cache_root / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_extract = tmp_dir / f"local-{tar_path.stem}-{os.getpid()}"
    if tmp_extract.exists():
        shutil.rmtree(tmp_extract)
    tmp_extract.mkdir(parents=True)
    try:
        _safe_extract(tar_path, tmp_extract)
        return _load_bundle(
            tmp_extract, source="local-tarball",
            requested_name=requested_name,
            requested_version=requested_version,
        )
    except Exception:
        shutil.rmtree(tmp_extract, ignore_errors=True)
        raise


def _read_manifest(bundle_dir: Path) -> dict:
    mp = bundle_dir / "manifest.json"
    if not mp.exists():
        return {}
    with open(mp) as f:
        return json.load(f)


def _validate_manifest(manifest: dict, name: str, version: str) -> None:
    """Verify manifest name/version match what was requested (for downloads)."""
    m_name = manifest.get("name")
    m_ver = manifest.get("version")
    if m_name and m_name != name:
        raise ModelResolutionError(
            f"Bundle manifest says name='{m_name}' but expected '{name}'."
        )
    if m_ver and m_ver != version:
        raise ModelResolutionError(
            f"Bundle manifest says version='{m_ver}' but expected '{version}'."
        )


_LEGACY_LAYOUT: dict[str, Any] = {
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
}


def _load_bundle(
    bundle_dir: Path,
    source: str,
    requested_name: str | None = None,
    requested_version: str | None = None,
) -> ResolvedModel:
    """Build a ResolvedModel from an extracted bundle directory.

    Falls back to _LEGACY_LAYOUT if manifest.json is absent (bridges the
    currently committed model tree during the migration period).
    """
    manifest = _read_manifest(bundle_dir)
    has_manifest = bool(manifest)

    if has_manifest:
        layout = manifest["layout"]
        name = manifest.get("name", requested_name or "unknown")
        version = manifest.get("version", requested_version or "unknown")
    else:
        # Legacy fallback: classic dir layout, no manifest.
        layout = _LEGACY_LAYOUT
        name = requested_name or "unknown"
        version = requested_version or "unknown"
        print(
            f"[nighthawk] No manifest.json found in bundle; using legacy layout "
            f"(run nh2.package_detector to generate a proper bundle)."
        )

    # Warn if name/version mismatch the request (for local bundles).
    if requested_name and name != requested_name:
        print(
            f"[nighthawk] WARNING: bundle name='{name}' but --model '{requested_name}' was requested."
        )
    if requested_version and version != requested_version:
        print(
            f"[nighthawk] WARNING: bundle version='{version}' but --model-version '{requested_version}' was requested."
        )

    # Resolve saved_model dir.
    saved_model_dir = bundle_dir / layout["saved_model"]
    if not saved_model_dir.exists():
        raise ModelResolutionError(
            f"SavedModel directory not found: {saved_model_dir}"
        )

    # Resolve taxonomy paths.
    tax_layout = layout["taxonomy"]
    tax_dir = bundle_dir / tax_layout["dir"]
    taxonomy = {
        role: tax_dir / filename
        for role, filename in tax_layout.items()
        if role != "dir"
    }
    for role, path in taxonomy.items():
        if not path.exists():
            raise ModelResolutionError(
                f"Taxonomy file not found: {path} (role={role!r})"
            )

    # Optional extras: not declared in the layout, but present opportunistically.
    # species_lookup_table.csv is a per-taxonomy subset of the GBIF/eBird
    # species-candidate lookup table (built by subset_lookup_for_taxonomy.py in
    # nighthawk-training).  Absent in older bundles — callers must guard with
    # resolved.taxonomy.get('species_lookup').
    if "species_lookup" not in taxonomy:
        sl = tax_dir / "species_lookup_table.csv"
        if sl.exists():
            taxonomy["species_lookup"] = sl

    # Resolve test_config paths.
    tc_layout = layout["test_config"]
    tc_dir = bundle_dir / tc_layout["dir"]
    test_config = {
        role: tc_dir / filename
        for role, filename in tc_layout.items()
        if role != "dir"
    }
    for role, path in test_config.items():
        if not path.exists():
            raise ModelResolutionError(
                f"Test-config path not found: {path} (role={role!r})"
            )

    return ResolvedModel(
        name=name,
        version=version,
        root=bundle_dir,
        saved_model_dir=saved_model_dir,
        taxonomy=taxonomy,
        test_config=test_config,
        manifest=manifest,
        source=source,
    )
