"""nighthawk-models — command-line tool for managing Nighthawk model bundles.

Subcommands
-----------
list    List cached models and (optionally) available remote versions.
fetch   Download and cache a model version without running detection.
path    Print the cache path for a specific model version.
clean   Remove one or all cached versions of a model.
"""

import argparse
import sys
from pathlib import Path

import nighthawk as nh


def main():
    parser = argparse.ArgumentParser(
        prog='nighthawk-models',
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest='subcommand', required=True)

    # --- list ---
    p_list = sub.add_parser(
        'list', help='List locally cached models and available remote versions.')
    p_list.add_argument(
        '--remote', action='store_true',
        help='Also fetch the registry and list available remote versions.')
    p_list.add_argument(
        '--repo-url', default=nh.DEFAULT_MODEL_REPO_URL, dest='repo_url',
        help=(
            f'Model repository base URL. (default: {nh.DEFAULT_MODEL_REPO_URL!r}) '
            'Accepts https:// or s3://bucket/prefix/ URIs. '
            "s3:// requires pip install 'nighthawk[s3]' and AWS credentials."))
    p_list.add_argument(
        '--cache-dir', type=Path, default=None, dest='cache_dir',
        help='Override the local cache directory.')

    # --- fetch ---
    p_fetch = sub.add_parser(
        'fetch', help='Download and cache a model version (no detection).')
    p_fetch.add_argument(
        'name', help='Model name to fetch (e.g. "americas").')
    p_fetch.add_argument(
        '--model-version', default='latest', dest='model_version',
        help='Version to fetch, or "latest". (default: latest)')
    p_fetch.add_argument(
        '--repo-url', default=nh.DEFAULT_MODEL_REPO_URL, dest='repo_url',
        help=(
            'Model repository base URL. '
            'Accepts https:// or s3://bucket/prefix/ URIs. '
            "s3:// requires pip install 'nighthawk[s3]' and AWS credentials."))
    p_fetch.add_argument(
        '--cache-dir', type=Path, default=None, dest='cache_dir',
        help='Override the local cache directory.')

    # --- path ---
    p_path = sub.add_parser(
        'path', help='Print the local cache path for a specific model version.')
    p_path.add_argument('name', help='Model name.')
    p_path.add_argument(
        '--model-version', required=True, dest='model_version',
        help='Concrete version string (not "latest").')
    p_path.add_argument(
        '--cache-dir', type=Path, default=None, dest='cache_dir',
        help='Override the local cache directory.')

    # --- clean ---
    p_clean = sub.add_parser(
        'clean', help='Remove cached model bundles.')
    p_clean.add_argument('name', help='Model name.')
    p_clean.add_argument(
        '--model-version', default=None, dest='model_version',
        help='Specific version to remove. Omit to remove ALL versions of this model.')
    p_clean.add_argument(
        '--cache-dir', type=Path, default=None, dest='cache_dir',
        help='Override the local cache directory.')

    args = parser.parse_args()

    if args.subcommand == 'list':
        _cmd_list(args)
    elif args.subcommand == 'fetch':
        _cmd_fetch(args)
    elif args.subcommand == 'path':
        _cmd_path(args)
    elif args.subcommand == 'clean':
        _cmd_clean(args)


# ---------------------------------------------------------------------------
# Subcommand implementations
# ---------------------------------------------------------------------------

def _cmd_list(args):
    cached = nh.list_cached_models(cache_dir=args.cache_dir)
    cached_set = {(m['name'], m['version']) for m in cached}

    if args.remote and args.repo_url:
        from nighthawk.model_manager import _fetch_registry, RegistryError
        try:
            registry = _fetch_registry(args.repo_url)
        except RegistryError as e:
            print(f'[nighthawk] Could not fetch registry: {e}', file=sys.stderr)
            registry = {}
        models = registry.get('models', {})
        print(f"{'Model':<20} {'Version':<12} {'Latest':<8} {'Cached':<8}")
        print('-' * 54)
        for name, model_entry in sorted(models.items()):
            latest = model_entry.get('latest', '')
            for version in sorted(model_entry.get('versions', {})):
                is_latest = '✓' if version == latest else ''
                is_cached = '✓' if (name, version) in cached_set else ''
                print(f"{name:<20} {version:<12} {is_latest:<8} {is_cached:<8}")
    else:
        if not cached:
            print('No cached models found.')
            return
        print(f"{'Model':<20} {'Version':<12} {'Path'}")
        print('-' * 70)
        for m in cached:
            print(f"{m['name']:<20} {m['version']:<12} {m['root']}")


def _cmd_fetch(args):
    print(f"Fetching {args.name}@{args.model_version} ...")
    try:
        resolved = nh.resolve_model(
            name=args.name,
            version=args.model_version,
            repo_url=args.repo_url or None,
            cache_dir=args.cache_dir,
        )
        print(f"Ready: {resolved.root}")
    except nh.NighthawkModelError as e:
        print(f'Error: {e}', file=sys.stderr)
        sys.exit(1)


def _cmd_path(args):
    from nighthawk.model_manager import _cache_root, _cached_bundle_path, _is_ready
    cache_root = _cache_root(args.cache_dir)
    bundle_dir = _cached_bundle_path(cache_root, args.name, args.model_version)
    if _is_ready(bundle_dir):
        print(bundle_dir)
    else:
        print(
            f"Model '{args.name}@{args.model_version}' is not cached. "
            f"Run: nighthawk-models fetch {args.name} --model-version {args.model_version}",
            file=sys.stderr,
        )
        sys.exit(1)


def _cmd_clean(args):
    removed = nh.remove_cached_model(
        name=args.name,
        version=args.model_version,
        cache_dir=args.cache_dir,
    )
    if removed:
        for r in removed:
            print(f"Removed: {r}")
    else:
        desc = (
            f"'{args.name}@{args.model_version}'"
            if args.model_version else f"all versions of '{args.name}'"
        )
        print(f"Nothing to remove for {desc}.")


if __name__ == '__main__':
    main()
