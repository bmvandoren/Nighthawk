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
aws s3api create-bucket --bucket nighthawk-models --region us-east-1
```

### 2. Enable public-read access for downloads

Most AWS accounts have Block Public Access enabled by default, which rejects a public
bucket policy. Disable it for this bucket first:

```bash
aws s3api put-public-access-block --bucket nighthawk-models \
    --public-access-block-configuration \
        BlockPublicAcls=false,IgnorePublicAcls=false,BlockPublicPolicy=false,RestrictPublicBuckets=false
```

Then create a bucket policy that allows anonymous GET on the model objects:

```bash
cat > bucket-policy.json << 'EOF'
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "PublicRead",
      "Effect": "Allow",
      "Principal": "*",
      "Action": ["s3:GetObject"],
      "Resource": "arn:aws:s3:::nighthawk-models/*"
    }
  ]
}
EOF

aws s3api put-bucket-policy --bucket nighthawk-models \
    --policy file://bucket-policy.json
```

### 3. Expected object layout

```
s3://nighthawk-models/
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
aws s3 cp registry.json s3://nighthawk-models/registry.json \
    --content-type application/json
```

### 5. Set the repo URL in the package

Edit `Nighthawk-repo/nighthawk/detector.py` and set:

```python
DEFAULT_MODEL_REPO_URL = 'https://nighthawk-models.s3.us-east-1.amazonaws.com/'
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
    --out-dir /home/vandoren/projects/nighthawk/experiments/classify-342-americas/package \
    --model-name americas \
    --model-version 0.2.0-342 \
    --export-name export_ema \
    --lookup-table ~/projects/nighthawk/nighthawk-training/acquire_gbif_data/data/species_lookup_table.csv
```

This writes the payload to `--out-dir` (for local testing) and creates:
- `<out-dir>/../dist/americas-0.2.0-342.tar.gz` — the distributable tarball
- `<out-dir>/../dist/americas-0.2.0-342.manifest.json` — sidecar metadata
- `<out-dir>/manifest.json` — in-payload manifest (also baked into the tarball)

The SHA-256 hash and size are printed for your records.

### Step 2: Smoke-test locally

```bash
cd Nighthawk-repo

# Test the assembled payload directly from the experiment dir — no need to copy
# it into the repo first
nighthawk Nighthawk-repo/test_inputs/test1.wav \
    --model-path /home/vandoren/projects/nighthawk/experiments/classify-342-americas/package \
    --threshold 50 --raven-output

# Should print "Using model americas@0.2.0-342 (source: local-dir)"
```

### Step 3: Publish to S3

```bash
# Public HTTPS repo
python -m nh2.package_detector \
    --experiment-dir /home/vandoren/projects/nighthawk/experiments/classify-342-americas \
    --out-dir /home/vandoren/projects/nighthawk/experiments/classify-342-americas/package \
    --model-name americas \
    --model-version 0.2.0-342 \
    --export-name export_ema \
    --lookup-table ~/projects/nighthawk/nighthawk-training/acquire_gbif_data/data/species_lookup_table.csv \
    --eval-subdir test57-20260513-1-selectv8-NHM-Americas-Chicago-Mexico-Canada-MPG \
    --publish \
    --repo-url https://nighthawk-models.s3.us-east-1.amazonaws.com/ \
    --bucket nighthawk-models \
    --region us-east-1

# Private S3 repo
python -m nh2.package_detector \
    --experiment-dir /home/vandoren/projects/nighthawk/experiments/classify-342-americas \
    --out-dir /home/vandoren/projects/nighthawk/experiments/classify-342-americas/package \
    --model-name americas \
    --model-version 0.2.0-342 \
    --export-name export_ema \
    --lookup-table ~/projects/nighthawk/nighthawk-training/acquire_gbif_data/data/species_lookup_table.csv \
    --eval-subdir test57-20260513-1-selectv8-NHM-Americas-Chicago-Mexico-Canada-MPG \
    --publish \
    --repo-url s3://nfc-util/nighthawk-models/ \
    --bucket nfc-util \
    --s3-prefix nighthawk-models/ \
    --region us-east-1
```

This:
1. Uploads the tarball and sidecar manifest under the configured `--s3-prefix`.
2. Downloads `registry.json` from the repo URL, merges the new entry, re-uploads it.
3. Sets `americas.latest = "0.2.0-342"` (suppress with `--no-set-latest`).

**`--eval-subdir`**: name of the test-set subdirectory under `eval/` (e.g.
`test57-20260513-1-selectv8-NHM-Americas-Chicago-Mexico-Canada-MPG`).  When there
is exactly one subdirectory it is auto-detected; pass this flag when there are
multiple to select the right one.

**`--lookup-table`**: path to the full GBIF/eBird species-candidate lookup table
CSV (built by
`nighthawk-training/acquire_gbif_data/scripts/build_species_lookup.py`).  When
provided, `subset_lookup_for_taxonomy.py` is run to generate
`taxonomy/species_lookup_table.csv` inside the bundle, which enables geographic
candidate filtering (`--lat`/`--lon`/`--month`) at inference time.  Omitting it
produces a warning and the table is excluded — re-run packaging with
`--lookup-table` to add it later.

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
    --bundle-dir /data/nighthawk/experiments/322-americas/final_model/packaged_model \
    --out-dir /data/nighthawk/experiments/322-americas/final_model/package \
    --model-name americas \
    --model-version 0.1.0-322 \
    --lookup-table ~/projects/nighthawk/nighthawk-training/acquire_gbif_data/data/species_lookup_table.csv

# Build + publish to S3 - PUBLIC REPO
python -m nh2.package_detector \
    --legacy \
    --bundle-dir /data/nighthawk/experiments/322-americas/final_model/packaged_model \
    --out-dir /data/nighthawk/experiments/322-americas/final_model/package \
    --model-name americas \
    --model-version 0.1.0-322 \
    --lookup-table ~/projects/nighthawk/nighthawk-training/acquire_gbif_data/data/species_lookup_table.csv \
    --publish \
    --repo-url https://nighthawk-models.s3.us-east-1.amazonaws.com/ \
    --bucket nighthawk-models \
    --region us-east-1 \
    --no-set-latest

# Build + publish to S3 - PRIVATE REPO with S3 URI
python -m nh2.package_detector \
    --legacy \
    --bundle-dir /data/nighthawk/experiments/322-americas/final_model/packaged_model \
    --out-dir /data/nighthawk/experiments/322-americas/final_model/package \
    --model-name americas \
    --model-version 0.1.0-322 \
    --lookup-table ~/projects/nighthawk/nighthawk-training/acquire_gbif_data/data/species_lookup_table.csv \
    --publish \
    --repo-url s3://nfc-util/nighthawk-models/ \
    --bucket nfc-util \
    --s3-prefix nighthawk-models/ \
    --region us-east-1 \
    --no-set-latest
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
| `--eval-subdir` | Test-set subdirectory under `eval/` (auto-detected when only one exists) |
| `--lookup-table` | Path to full GBIF/eBird species-candidate lookup CSV; generates `taxonomy/species_lookup_table.csv` in the bundle enabling `--lat`/`--lon`/`--month` filtering at inference time |
| `--subset-script` | Path to `subset_lookup_for_taxonomy.py` (auto-detected; only needed for non-standard workspace layouts) |
| `--publish` | Trigger the S3 upload + registry update |
| `--repo-url` | Root URL of the model repository (`https://` or `s3://`); `registry.json` lives here |
| `--bucket` | S3 bucket name |
| `--region` | AWS region (default: inferred from env) |
| `--s3-prefix` | S3 key prefix for artifact objects; must begin with the path component of `--repo-url` for `s3://` repos |
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

nighthawk ../Nighthawk-repo/test_inputs/test1.wav \
    --output-dir ../playground \
    --threshold 50
```

On first run this downloads `americas@latest` (~190 MB) and caches it.
Subsequent runs are instant — no download.

### Pin a version for reproducible results

```bash
nighthawk my_recording.wav --model-version 0.1.0-322

nighthawk ../Nighthawk-repo/test_inputs/test1.wav \
    --model-version 0.1.0-322 \
    --output-dir ../playground/322 \
    --threshold 50
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

The `--model-repo-url` flag (and the `--repo-url` flag in `nighthawk-models`) accepts
either a public `https://` URL or a **private** `s3://bucket/prefix/` URI.

```bash
# Public HTTPS repository (default)
nighthawk my_recording.wav --model-repo-url https://my-other-bucket.s3.amazonaws.com/

# Private S3 bucket via boto3 (see below)
nighthawk my_recording.wav --model-repo-url s3://my-private-bucket/nighthawk/
```

### Using a private S3 repository (`s3://`)

When the repo URL begins with `s3://`, the client fetches the registry and model
tarballs using **boto3** instead of plain HTTP.  This lets you host models in a
private bucket — no public-read bucket policy required.

**Prerequisites:**

1. Install boto3 (not included in the base nighthawk install):

   ```bash
   pip install 'nighthawk[s3]'
   ```

2. Configure AWS credentials — any of the standard boto3 methods work:

   ```bash
   # Option A: environment variables
   export AWS_ACCESS_KEY_ID=...
   export AWS_SECRET_ACCESS_KEY=...
   export AWS_DEFAULT_REGION=us-east-1  # optional

   # Option B: named profile
   export AWS_PROFILE=my-profile

   # Option C: ~/.aws/credentials + ~/.aws/config (aws configure)
   ```

3. Ensure the IAM principal has `s3:GetObject` on the bucket objects (and
   `s3:GetObject` on `registry.json` specifically).  No public-read bucket policy
   is needed.

**Usage:**

```bash
nighthawk my_recording.wav \
    --model-repo-url s3://my-private-bucket/ \
    --threshold 50 --raven-output

# Pre-download without running detection
nighthawk-models fetch americas \
    --repo-url s3://my-private-bucket/

# List remote versions
nighthawk-models list --remote \
    --repo-url s3://my-private-bucket/
```

**Important:** `--model-repo-url` / `--repo-url` is the root of the repository —
`registry.json` lives there and all model URLs in the registry are relative to it.
For a private bucket you can put the repo at any prefix (e.g. `s3://bucket/myrepo/`);
the `--s3-prefix` passed at publish time must start with that same prefix
(e.g. `--s3-prefix myrepo/` or `--s3-prefix myrepo/models/`).

The expected bucket layout is identical to the public HTTPS layout (see
[S3 repository setup](#s3-repository-setup-one-time) above).  The SHA-256
checksum verification and safe-extraction guarantees described in the
[Security](#security) section apply equally to S3 downloads.

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
