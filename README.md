![mss-datasets-logo](https://raw.githubusercontent.com/crlandsc/mss-datasets/main/images/logo.png)

[![LICENSE](https://img.shields.io/github/license/crlandsc/mss-datasets)](https://github.com/crlandsc/mss-datasets/blob/main/LICENSE) [![GitHub Repo stars](https://img.shields.io/github/stars/crlandsc/mss-datasets)](https://github.com/crlandsc/mss-datasets/stargazers) [![Python Version](https://img.shields.io/badge/python-%3E%3D3.9-blue)](https://github.com/crlandsc/mss-datasets)

Aggregate multiple music source separation datasets (MUSDB18-HQ, MoisesDB, MedleyDB) into unified stem folders for MSS training.

## Installation

Github install:
```bash
pip install git+https://github.com/crlandsc/mss-datasets.git
```

or clone repository, then install locally:
```bash
git clone https://github.com/crlandsc/mss-datasets.git
cd mss-datasets
```

```bash
# Local install
pip install -e .

# Development install (adds dev dependencies)
pip install -e ".[dev]"
```

Requires Python >= 3.9.

## Quick Start

**Config file** — recommended for complex/repeatable setups:

```yaml
# config.yaml
datasets:
  musdb18hq_path: /path/to/musdb18hq
  moisesdb_path: /path/to/moisesdb
  medleydb_path: /path/to/medleydb
output: ./data
profile: vdbo
workers: 4
```

```bash
mss-datasets --config config.yaml # aggregation only
```

**CLI** — equivalent command:

```bash
mss-datasets --aggregate \
  --musdb18hq-path /path/to/musdb18hq \
  --moisesdb-path /path/to/moisesdb \
  --medleydb-path /path/to/medleydb \
  --output ./data \
  --workers 4
```

Both approaches are fully equivalent. Use whichever fits your workflow — config files are easier to manage when you have many flags.

## Flow 1: Download + Aggregate (All-in-One)

Downloads MUSDB18-HQ and MedleyDB automatically, then aggregates all datasets. MoisesDB must be downloaded manually.

*NOTE: Downloading and aggregation will take several hours. Raw datasets require ~242 GB (MUSDB18-HQ ~31 GB, MedleyDB ~62 GB, MoisesDB ~149 GB) and aggregation adds ~175–205 GB depending on profile and options — plan for ~420–450 GB total. Recommend leaving it running in the background or overnight.*

### Prerequisites

Complete ***all*** of the following before running. *These steps **cannot** be skipped.*

1. **Create a Zenodo account and personal access token**
   - Create an account at https://zenodo.org
   - Go to https://zenodo.org/account/settings/applications/
   - Click "New token", name it anything (no scopes need to be selected — the token is used for authentication only)
   - Copy the token

2. **Request access to the MedleyDB records**
   - Visit **both** pages below while logged in and click "Request access":
     - MedleyDB v1: https://zenodo.org/records/1649325
     - MedleyDB v2: https://zenodo.org/records/1715175
   - You must wait for the dataset owners to approve your request. Typically happens within minutes, but timing is not guarunteed. Until approved, the download will fail with `"No files found"`.

3. **Download MoisesDB manually**
   - Visit https://music.ai/research/ and scroll down to the MoisesDB dataset section
   - Select **Download** and enter required fields to **request download**. A download link will be sent to your email, usually within minutes.
   - Unzip the archive
   - It will unzip to a `moisesdb_v0.1/` subfolder - this should be used as the base `moisesdb_path`.

### Run `mss-datasets`

#### Option 1: Config file

Recommended for better organization & repeatability.

```yaml
# provide arguments in config.yaml
datasets:
  # musdb18hq_path and medleydb_path are omitted — auto-set by --download
  moisesdb_path: /path/to/moisesdb  # manual download from step 3
output: ./data
profile: vdbo
workers: 4
data_dir: ./datasets
zenodo_token: YOUR_ZENODO_TOKEN
```

```bash
mss-datasets --config config.yaml --download --aggregate
```

#### Option 2: CLI

If you do not want to use a config, all arguments can be added directly in the CLI.

```bash
mss-datasets --download --aggregate \
  --moisesdb-path /path/to/moisesdb \
  --zenodo-token YOUR_ZENODO_TOKEN \
  --data-dir ./datasets \
  --output ./data \
  --workers 4
```

NOTE: You can also provide the Zenodo token via environment variable (`ZENODO_TOKEN`) or `.env` file instead of the CLI of config flags.

Downloads are resumable — if interrupted, re-run the same command to continue.

## Flow 2: Aggregate Pre-Downloaded Datasets

If you already have the datasets downloaded, point the tool at their directories. Use any combination — all three are optional, but at least one is required.

### Dataset Directory Structures

**MUSDB18-HQ** — `train/` and `test/` subdirs, each with `Artist - Title` folders containing stem WAVs:

```
musdb18hq/
├── train/
│   ├── Artist - Title/
│   │   ├── vocals.wav
│   │   ├── drums.wav
│   │   ├── bass.wav
│   │   ├── other.wav
│   │   └── mixture.wav
│   └── ...
└── test/
    ├── Artist - Title/
    │   └── (same stem files)
    └── ...
```

**MedleyDB** — `Audio/` subdir with `ArtistName_TrackTitle` folders, each containing metadata YAML and a `_STEMS/` subdir:

```
medleydb/
└── Audio/
    ├── ArtistName_TrackTitle/
    │   ├── ArtistName_TrackTitle_METADATA.yaml
    │   └── ArtistName_TrackTitle_STEMS/
    │       ├── ArtistName_TrackTitle_STEM_01.wav
    │       ├── ArtistName_TrackTitle_STEM_02.wav
    │       └── ...
    └── ...
```

If metadata YAML files are missing (common with Zenodo downloads), they will be auto-downloaded from GitHub on first run.

**MoisesDB** — either the official `moisesdb_v0.1/` layout or flat UUID-named track directories:

```
moisesdb/
├── moisesdb_v0.1/       # official layout
│   └── <provider>/
│       └── <track-uuid>/
│           └── data.json
└── ...

# OR flat layout (common after manual unzip):
moisesdb/
├── <track-uuid>/
│   └── data.json
├── <track-uuid>/
│   └── data.json
└── ...
```

### Run

#### Option 1: Config file

```yaml
# provide arguments in config.yaml
datasets:
  musdb18hq_path: /path/to/musdb18hq
  moisesdb_path: /path/to/moisesdb
  medleydb_path: /path/to/medleydb
output: ./data
profile: vdbo
workers: 4
```

```bash
mss-datasets --config config.yaml
```

#### Option 2: CLI

```bash
mss-datasets --aggregate \
  --musdb18hq-path /path/to/musdb18hq \
  --moisesdb-path /path/to/moisesdb \
  --medleydb-path /path/to/medleydb \
  --output ./data \
  --workers 4
```

**Dry run** — preview what would be processed without writing files:

```bash
mss-datasets --config config.yaml --dry-run
# or
mss-datasets --dry-run --musdb18hq-path /path/to/musdb18hq
```

## Configuration

See [`config.example.yaml`](config.example.yaml) for a fully annotated config template with explanations for every option. Copy it and adjust paths/settings:

```bash
cp config.example.yaml config.yaml
# edit config.yaml with your paths
mss-datasets --config config.yaml
```

Config files use YAML format. Dataset paths go under a `datasets:` key; all other options are top-level. CLI flags always override config file values.

## CLI Reference

| Flag | Default | Description |
|---|---|---|
| `--download` | off | Download MUSDB18-HQ and MedleyDB |
| `--aggregate` | off | Aggregate datasets into unified stem folders |
| `--musdb18hq-path` | -- | Path to MUSDB18-HQ dataset |
| `--moisesdb-path` | -- | Path to MoisesDB dataset |
| `--medleydb-path` | -- | Path to MedleyDB dataset |
| `--output`, `-o` | `./output` | Output directory |
| `--profile` | `vdbo` | `vdbo` (4-stem) or `vdbo+gp` (6-stem) |
| `--workers` | `1` | Parallel workers (MoisesDB always sequential) |
| `--group-by-dataset` | off | Add source dataset subfolders within each stem folder |
| `--split-output` | off | Organize output into `train/` and `val/` directories |
| `--include-mixtures` | off | Generate mixture WAV files |
| `--include-bleed` | off | Include tracks with stem bleed (excluded by default) |
| `--verify-mixtures` | off | Verify written stem sums match mixture files (requires `--include-mixtures`) |
| `--dry-run` | off | Preview what would be processed without writing |
| `--validate` | -- | Validate an existing output directory |
| `--config` | -- | Path to YAML config file |
| `--data-dir` | `./datasets` | Directory for raw dataset downloads |
| `--zenodo-token` | -- | Zenodo access token for MedleyDB (also: `ZENODO_TOKEN` env var) |
| `--verbose`, `-v` | off | Debug logging |

At least one mode flag is required: `--download`, `--aggregate`, `--dry-run`, or `--validate`.

## Output Format

**[Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/main/docs/dataset_types.md) — Type 2 layout** — one folder per stem:

**`vdbo` (4-stem) — ~164 GB, 2,204 files:**

```
output/
├── vocals/    (410 files)
├── drums/     (446 files)
├── bass/      (430 files)
├── other/     (458 files)
├── mixture/   (460 files, with --include-mixtures)
└── metadata/
```

**`vdbo+gp` (6-stem) — ~188 GB, 2,522 files:**

```
output/
├── vocals/    (410 files)
├── drums/     (446 files)
├── bass/      (430 files)
├── other/     (331 files)
├── guitar/    (290 files)
├── piano/     (155 files)
├── mixture/   (460 files, with --include-mixtures)
└── metadata/
```

**`--split-output`** — organize by train/val split:

```
output/
├── train/
│   ├── vocals/
│   ├── drums/
│   ├── bass/
│   └── other/
├── val/
│   ├── vocals/
│   ├── drums/
│   ├── bass/
│   └── other/
└── metadata/
```

MUSDB18-HQ "test" tracks are remapped to "val" — there is no "test" directory.

**`--group-by-dataset`** — add source dataset subfolders within each stem folder:

```
output/
├── vocals/
│   ├── musdb18hq/
│   ├── medleydb/
│   └── moisesdb/
├── drums/
│   ├── musdb18hq/
│   ├── medleydb/
│   └── moisesdb/
├── bass/
│   └── ...
├── other/
│   └── ...
└── metadata/
```

**`--include-mixtures`** — generate mixture WAV files in a separate folder:

```
output/
├── vocals/
├── drums/
├── bass/
├── other/
├── mixture/
└── metadata/
```

Combines with `--group-by-dataset` for nested layouts (e.g. `output/vocals/musdb18hq/`) and with `--split-output` for split+grouped layouts (e.g. `output/train/vocals/musdb18hq/`). All flags are composable.

The `metadata/` directory contains: `manifest.json`, `splits.json`, `overlap_registry.json`, `errors.json`, `config.yaml`.

460 tracks total across all three datasets (360 train / 100 val). All output: 44.1 kHz, float32, stereo WAV. Stem folders have independent file counts — not every track appears in every folder.

Filename format: `{source}_{split}_{index:04d}_{artist}_{title}.wav`

## Datasets

- **MUSDB18-HQ**: 150 tracks, 4 stems. 100 train / 50 test.
  - Direct 1:1 stem mapping. Only has 4 stems — does not contribute to guitar/piano in `vdbo+gp`.
  - Mixture copied directly from source (not summed from stems).
- **MoisesDB**: 240 tracks, 11 top-level stems. 50-track val set (deterministic random selection, seed=42).
  - `percussion` and `bass` are routed at the sub-stem level, not top-level:
    - Bass guitar, bass synth, contrabass → bass; tuba, bassoon → other
    - Atonal percussion → drums; pitched percussion → other
  - Always processed sequentially (library constraint).
- **MedleyDB v1+v2**: 196 tracks, ~121 instrument labels mapped to stems. 5 tracks are excluded (stem bleed), 37 individual stems are excluded (audio content doesn't match routed category), and 1 stem is rerouted to the correct category. See [MedleyDB Exclusions](docs/medleydb_exclusions.md) for details.
  - 2 special labels: "Main System" → excluded, "Unlabeled" → other.
  - Multiple stems mapping to the same output category are summed.
  - `has_bleed` metadata field controls `--include-bleed`. The 5 override-excluded tracks are always excluded regardless of `--include-bleed`.
  - Metadata YAML auto-downloaded from GitHub if missing.

46 tracks overlap between MUSDB18-HQ and MedleyDB — MedleyDB is preferred (more granular stems). Cross-dataset deduplication is automatic and only triggers when both MUSDB18-HQ and MedleyDB are present.

### Aggregation Notes

- `vdbo` routes guitar/piano labels to "other"; `vdbo+gp` gives them dedicated stems.
- Mono sources are converted to dual-mono. No resampling — all sources expected to be 44.1 kHz.
- Silent stems are automatically skipped.
- MedleyDB overlap tracks inherit the MUSDB18-HQ split; non-overlap tracks default to train.
- Splits are locked via `splits.json` for reproducibility across re-runs.
- Processing is resumable — tracks with existing output files are skipped on re-run.

## License

This tool is MIT-licensed. The underlying datasets have their own licenses — see [LICENSE](LICENSE) for details.
