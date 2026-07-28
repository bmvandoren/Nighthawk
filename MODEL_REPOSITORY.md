# Nighthawk Model Repository

This document explains how the Nighthawk model repository works, how to set it up on S3,
how to publish new models, and how end users can select and manage model versions.

---

## How it works

Starting with version 0.4.0, the `nighthawk` package no longer bundles a model inside the
wheel. Instead, models are downloaded on first use from a public S3-backed repository and
cached locally so they are only downloaded once.

Models are identified by a **(name, version)** pair, for example `americas@0.4.0`.  Multiple
named models can coexist — e.g. `americas` and `europe` — each with their own version history.

---

## S3 repository setup (one-time)

### 1. Create the S3 bucket

```bash
aws s3api create-bucket --bucket my-nighthawk-models --region us-east-1 \
    --create-bucket-configuration LocationConstraint=us-east-1
```

### 2. Enable public-read access for downloads

Create a bucket policy that allows anonymous GET on the model objects:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "PublicRead",
      "Effect": "Allow",
      "Principal": "*",
      "Action": ["s3:GetObject"],
      "Resource": "arn:aws:s3:::my-nighthawk-models/*"
    }
  ]
}
```

```bash
aws s3api put-bucket-policy --bucket my-nighthawk-models \
    --policy file://bucket-policy.json
```

If Block Public Access is enabled on the account, disable it for this bucket:

```bash
aws s3api put-public-access-block --bucket my-nighthawk-models \
    --public-access-block-configuration \
        BlockPublicAcls=false,IgnorePublicAcls=false,BlockPublicPolicy=false,RestrictPublicBuckets=false
```

### 3. Expected object layout

```
s3://my-nighthawk-models/
  registry.json                                   ← global index (updated by publisher)
  models/
    americas/
      americas-0.4.0.tar.gz                       ← model bundle (~190 MB)
      americas-0.4.0.manifest.json                ← sidecar metadata
    europe/
      europe-1.0.0.tar.gz
      europe-1.0.0.manifest.json
```

### 4. Seed an empty registry

On first publish the script creates `registry.json` automatically.  If you want to seed it
manually:

```bash
cat > registry.json << 'EOF'
{
  "schema_version": 1,
  "models": {}
}
EOF
aws s3 cp registry.json s3://my-nighthawk-models/registry.json \
    --content-type application/json
```

### 5. Set the repo URL in the package

Edit `Nighthawk-repo/nighthawk/detector.py` and set:

```python
DEFAULT_MODEL_REPO_URL = 'https://my-nighthawk-models.s3.us-east-1.amazonaws.com/'
```

The URL must end with `/`.  This value is baked into the wheel so end users get it for free.

---

## Publishing a model (developer workflow)

### Prerequisites

- `nh2.package_detector` runs in the `nighthawk-training-py3.13` conda environment.
- A trained + evaluated nh2 experiment with calibration already generated
  (`eval/<test-set>/<export-name>/calibration/probability_calibrations_logistic_fromlogits.csv`
  must exist — see `nh2/PLAN.md` for the `--calibrate` command).
- AWS credentials with write access to the bucket:
  ```bash
  export AWS_ACCESS_KEY_ID=...
  export AWS_SECRET_ACCESS_KEY=...
  # or: aws configure
  ```
- `boto3` installed in the dev environment:
  ```bash
  pip install boto3
  ```

### Step 1: Build the artifact (no upload)

```bash
conda activate nighthawk-training-py3.13
cd nighthawk-development

python -m nh2.package_detector \
    --experiment-dir /home/vandoren/projects/nighthawk/experiments/classify-342-americas \
    --out-dir /home/vandoren/projects/nighthawk/Nighthawk-repo/nighthawk \
    --model-name americas \
    --model-version 0.4.0 \
    --export-name export_ema
```

This writes the payload to `--out-dir` (for local testing) and creates:
- `<out-dir>/../dist/americas-0.4.0.tar.gz` — the distributable tarball
- `<out-dir>/../dist/americas-0.4.0.manifest.json` — sidecar metadata
- `<out-dir>/manifest.json` — in-payload manifest (also baked into the tarball)

The SHA-256 hash and size are printed for your records.

### Step 2: Smoke-test locally

```bash
cd Nighthawk-repo

# Test the assembled payload directly (using the committed tree as a local bundle)
nighthawk playground/test_inputs/my_recording.wav \
    --model-path nighthawk \
    --threshold 70 --raven-output

# Should print "Using model americas@0.4.0 (source: local-dir)"
```

### Step 3: Publish to S3

```bash
python -m nh2.package_detector \
    --experiment-dir /home/vandoren/projects/nighthawk/experiments/classify-342-americas \
    --out-dir /home/vandoren/projects/nighthawk/Nighthawk-repo/nighthawk \
    --model-name americas \
    --model-version 0.4.0 \
    --export-name export_ema \
    --publish \
    --repo-url https://my-nighthawk-models.s3.us-east-1.amazonaws.com/ \
    --bucket my-nighthawk-models \
    --region us-east-1
```

This:
1. Uploads the tarball and sidecar manifest under `models/americas/`.
2. Downloads `registry.json`, merges the new entry, re-uploads it.
3. Sets `americas.latest = "0.4.0"` (suppress with `--no-set-latest`).

**`--force`**: required if you are re-publishing an existing `(name, version)`.  Without it,
the script refuses to overwrite an already-published version (protects reproducibility).

**`--no-set-latest`**: use when re-publishing a patch to an older version without promoting
it to `latest`.

**Race note**: the registry update is a read-modify-write.  Don't run two publishers
simultaneously against the same registry — the last writer wins.

### Packaging a legacy (pre-nh2) model

Legacy models that are already assembled as a bundle tree (e.g. a previous
`nighthawk` release tree) can also be published to the same repository using
`--legacy` mode.  The bundle must contain:

```
<bundle-dir>/
  saved_model_with_preprocessing/
    saved_model.pb
    variables/
  taxonomy/
    taxonomy_version.txt
    orders.txt
    families.txt
    groups.txt
    species.txt
    ebird_taxonomy.csv
    groups_ebird_codes.csv
  test_config/
    test_config.json
    probability_calibrations_logistic_fromlogits.csv  ← logit-space calibrators required
    test_set_performance/
      taxon_summary_order.csv
      taxon_summary_family.csv
      taxon_summary_group.csv
      taxon_summary_species.csv
```

**Note:** Legacy models must have `probability_calibrations_logistic_fromlogits.csv`
(calibrators fit in logit space).  The older `probability_calibrations.csv`
(probability-space) is not supported.

```bash
conda activate nighthawk-training-py3.13
cd nighthawk-development

# Build artifact only
python -m nh2.package_detector \
    --legacy \
    --bundle-dir /path/to/legacy/bundle \
    --out-dir /home/vandoren/projects/nighthawk/Nighthawk-repo/nighthawk \
    --model-name 300-americas \
    --model-version 1.0.0

# Build + publish to S3
python -m nh2.package_detector \
    --legacy \
    --bundle-dir /path/to/legacy/bundle \
    --out-dir /home/vandoren/projects/nighthawk/Nighthawk-repo/nighthawk \
    --model-name 300-americas \
    --model-version 1.0.0 \
    --publish \
    --repo-url https://my-nighthawk-models.s3.us-east-1.amazonaws.com/ \
    --bucket my-nighthawk-models \
    --region us-east-1
```

The resulting `manifest.json` records `"model_type": "legacy"`, which tells the
client to use the original unbatched calling convention for that model.  A legacy
bundle loaded via `--model-path` (without a manifest) is also auto-detected by
checking the SavedModel's signature.

### Useful publish flags summary

| Flag | Description |
|---|---|
| `--legacy` | Package a pre-nh2 bundle tree instead of an nh2 experiment dir |
| `--bundle-dir` | Path to assembled bundle tree (required with `--legacy`) |
| `--model-name` | Registry key (e.g. `americas`) |
| `--model-version` | Artifact version (e.g. `0.4.0`) |
| `--publish` | Trigger the S3 upload + registry update |
| `--repo-url` | Public base URL of the bucket |
| `--bucket` | S3 bucket name |
| `--region` | AWS region (default: inferred from env) |
| `--s3-prefix` | S3 key prefix (default: `models/`) |
| `--force` | Overwrite an existing version in the registry |
| `--no-set-latest` | Don't advance the `latest` pointer |
| `--package-min-version` | Minimum `nighthawk` client version required |
| `--version` | Also bump `Nighthawk-repo/pyproject.toml` version |
| `--dist-dir` | Where to write the tarball (default: `out-dir/../dist/`) |

---

## End-user guide

### Default usage (auto-downloads americas@latest)

```bash
nighthawk my_recording.wav
```

On first run this downloads `americas@latest` (~190 MB) and caches it.
Subsequent runs are instant — no download.

### Pin a version for reproducible results

```bash
nighthawk my_recording.wav --model-version 0.4.0
```

Results are guaranteed to match any other run with the same `--model-version` flag, regardless
of when a new model is published.

### Use a different model (e.g. Europe)

```bash
nighthawk my_recording.wav --model europe --model-version 1.0.0
```

### Run a local / custom model

```bash
# From an extracted bundle directory (must contain manifest.json)
nighthawk my_recording.wav --model-path /path/to/my-custom-model/

# From a tarball
nighthawk my_recording.wav --model-path /path/to/americas-0.4.0.tar.gz
```

No network access occurs when `--model-path` is given.

### Work completely offline

```bash
nighthawk my_recording.wav --offline
```

Serves the newest cached version of `americas`. If nothing is cached, fails with a clear message.

Or with a pinned version:

```bash
nighthawk my_recording.wav --model-version 0.4.0 --offline
```

Fails if `americas@0.4.0` is not in the cache.

### Override the cache directory

```bash
nighthawk my_recording.wav --cache-dir /my/shared/model/cache
# or
export NIGHTHAWK_CACHE_DIR=/my/shared/model/cache
nighthawk my_recording.wav
```

### Change the model repository URL

```bash
nighthawk my_recording.wav --model-repo-url https://my-other-bucket.s3.amazonaws.com/
```

---

## nighthawk-models — model management CLI

```bash
# List locally cached models
nighthawk-models list

# List cached + remote models (requires network)
nighthawk-models list --remote

# Pre-download a model without running detection
nighthawk-models fetch americas --model-version latest
nighthawk-models fetch europe --model-version 1.0.0

# Print the cache path for a specific version
nighthawk-models path americas --model-version 0.4.0

# Remove a specific cached version
nighthawk-models clean americas --model-version 0.3.1

# Remove all cached versions of a model
nighthawk-models clean americas
```

### Typical `list --remote` output

```
Model                Version      Latest   Cached
------------------------------------------------------
americas             0.3.1                 ✓
americas             0.4.0        ✓        ✓
europe               1.0.0        ✓
```

---

## Cache directory layout

```
<cache_root>/                   # e.g. ~/.cache/nighthawk/
  models/
    americas/
      0.3.1/                    # extracted bundle
        manifest.json
        saved_model_with_preprocessing/
        taxonomy/
        test_config/
        .ready                  # marker: fully extracted + validated
      0.4.0/
        ...
    europe/
      1.0.0/
        ...
  tmp/                          # download staging area (cleaned automatically)
```

The cache root defaults to the platform user-cache directory:

| Platform | Default |
|---|---|
| Linux | `~/.cache/nighthawk/` |
| macOS | `~/Library/Caches/nighthawk/` |
| Windows | `%LOCALAPPDATA%\nighthawk\` |

Override with `--cache-dir` or env `NIGHTHAWK_CACHE_DIR`.

---

## Security

All downloaded tarballs are SHA-256 verified against the registry before extraction.  The
extraction step rejects absolute paths, path traversal (`../`), and unsafe symlinks — so a
tampered archive cannot write files outside the cache directory.

---

## Migration from nighthawk < 0.4.0

The bundled model (`saved_model_with_preprocessing/`, `taxonomy/`, `test_config/`) is no
longer included in the wheel.  To use the locally committed model tree during development:

```bash
nighthawk my_recording.wav --model-path /path/to/Nighthawk-repo/nighthawk
```

If the directory does not contain a `manifest.json` (pre-0.4.0 layout), a legacy fallback
synthesizes the expected layout automatically.  Run `nh2.package_detector` once to generate
a proper `manifest.json` for the bundle.
