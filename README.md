# mss-datasets

Aggregate multiple music source separation datasets (MUSDB18-HQ, MoisesDB, MedleyDB) into unified stem folders for MSS training.

## Installation

```bash
pip install mss-datasets

# With MoisesDB support
pip install mss-datasets[moisesdb]

# Development
pip install -e ".[dev]"
```

Requires Python >= 3.9.

## Quick Start

```bash
# 4-stem (vocals/drums/bass/other)
mss-datasets --aggregate \
  --musdb18hq-path /path/to/musdb18hq \
  --moisesdb-path /path/to/moisesdb \
  --medleydb-path /path/to/medleydb \
  --output ./data

# 6-stem (adds guitar/piano)
mss-datasets --aggregate --profile vdbo+gp \
  --musdb18hq-path /path/to/musdb18hq \
  --output ./data

# Dry run — preview without writing
mss-datasets --dry-run --musdb18hq-path /path/to/musdb18hq
```

## Output Format

[**Music-Source-Separation-Training — Type 2**](https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/main/docs/dataset_types.md) layout — one folder per stem:

```
output/
├── vocals/    (~500 files)
├── drums/     (~490 files)
├── bass/      (~470 files)
├── other/     (~480 files)
├── guitar/    (6-stem only)
├── piano/     (6-stem only)
└── metadata/
    ├── manifest.json
    ├── splits.json
    ├── overlap_registry.json
    ├── errors.json
    └── config.yaml
```

All output: 44.1 kHz, float32, stereo WAV. Stem folders have independent file counts — not every track appears in every folder.

Filename format: `{source}_{split}_{index:04d}_{artist}_{title}.wav`

## CLI Reference

| Flag | Default | Description |
|---|---|---|
| `--download` | off | Download datasets |
| `--aggregate` | off | Aggregate datasets into unified stem folders |
| `--musdb18hq-path` | — | Path to MUSDB18-HQ |
| `--moisesdb-path` | — | Path to MoisesDB |
| `--medleydb-path` | — | Path to MedleyDB |
| `--output` | `./output` | Output directory |
| `--profile` | `vdbo` | `vdbo` (4-stem) or `vdbo+gp` (6-stem) |
| `--workers` | `1` | Parallel workers |
| `--group-by-dataset` | off | Dataset subfolders within stems |
| `--include-mixtures` | off | Generate mixture files |
| `--normalize-loudness` | off | EBU R128 normalization |
| `--loudness-target` | `-14` | Target LUFS |
| `--dry-run` | off | Preview without writing |
| `--validate` | — | Validate existing output |
| `--config` | — | YAML config file |
| `--data-dir` | `./datasets` | Directory for raw dataset downloads |
| `--zenodo-token` | — | Zenodo access token for MedleyDB |
| `--verbose` | off | Debug logging |

At least one mode flag is required: `--download`, `--aggregate`, `--dry-run`, or `--validate`.

## Download Mode

Auto-download datasets instead of pre-downloading manually:

```bash
# Download all available datasets + aggregate
mss-datasets --download --aggregate --output ./data

# Download only (no aggregation)
mss-datasets --download --data-dir ./datasets

# With MedleyDB (requires Zenodo token — see setup below)
mss-datasets --download --aggregate --zenodo-token YOUR_TOKEN --output ./data
```

| Dataset | Auto-download? | Auth required | Download size |
|---|---|---|---|
| MUSDB18-HQ | Yes | None (open access) | ~23 GB |
| MedleyDB v1+v2 | Yes | Zenodo token + record access approval | ~87 GB (v1: 43 GB, v2: 44 GB) |
| MoisesDB | No — manual download only | music.ai account | ~83 GB |
| **Total** | | | **~193 GB** |

Downloads are resumable — if interrupted, re-run the same command to continue where you left off.

### MedleyDB: Zenodo Token + Access Approval

MedleyDB is hosted on Zenodo as a **restricted dataset**. Auto-download requires two things: a personal access token (PAT) **and** approved access to each record. A token alone is not enough.

**Step 1: Create a Zenodo account and personal access token**

1. Create an account at https://zenodo.org
2. Go to https://zenodo.org/account/settings/applications/
3. Click "New token", name it anything, and select the `deposit:read` scope
4. Copy the token

**Step 2: Request access to the MedleyDB records**

Visit **both** of these pages while logged in and click "Request access":

- MedleyDB v1: https://zenodo.org/records/1649325
- MedleyDB v2: https://zenodo.org/records/1715175

You must wait for the dataset owners to approve your request. This may take hours or days. Until approved, the download will fail with `"No files found"`.

**Step 3: Provide the token**

Once access is approved, provide your token via any of these (in priority order):

1. CLI flag: `--zenodo-token YOUR_TOKEN`
2. Environment variable: `export ZENODO_TOKEN=YOUR_TOKEN`
3. `.env` file in the project root (see `.env.example` for template):
   ```
   ZENODO_TOKEN=YOUR_TOKEN
   ```

The tool validates your token before starting any downloads. If validation fails, you'll see a message indicating whether the issue is a missing token, an invalid token, or pending access approval.

### MoisesDB: Manual Download

MoisesDB cannot be auto-downloaded. To include it:

1. Download from https://music.ai/research/ (requires a music.ai account)
2. Extract to a local directory
3. Pass the path: `--moisesdb-path /path/to/moisesdb`

## Datasets

- **MUSDB18-HQ**: 150 tracks, 4 stems. 100 train / 50 test.
- **MoisesDB**: 240 tracks, 11 top-level stems. 50-track val set.
- **MedleyDB v1+v2**: 196 tracks, ~121 instrument labels mapped to stems.

46 tracks overlap between MUSDB18 and MedleyDB — MedleyDB preferred (more granular stems). Cross-dataset deduplication is automatic.

## Configuration

```yaml
# config.yaml
datasets:
  musdb18hq_path: /path/to/musdb18hq
  moisesdb_path: /path/to/moisesdb
  medleydb_path: /path/to/medleydb
output: ./data
profile: vdbo
workers: 8
group_by_dataset: false
```

```bash
mss-datasets --config config.yaml
```

CLI flags override config file values.

## License

MIT
