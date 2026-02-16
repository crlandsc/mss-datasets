# mss-aggregate

Aggregate multiple music source separation datasets (MUSDB18-HQ, MoisesDB, MedleyDB) into unified stem folders for MSS training.

## Installation

```bash
pip install mss-aggregate

# With MoisesDB support
pip install mss-aggregate[moisesdb]

# Development
pip install -e ".[dev]"
```

Requires Python >= 3.9.

## Quick Start

```bash
# 4-stem (vocals/drums/bass/other)
mss-aggregate \
  --musdb18hq-path /path/to/musdb18hq \
  --moisesdb-path /path/to/moisesdb \
  --medleydb-path /path/to/medleydb \
  --output ./data

# 6-stem (adds guitar/piano)
mss-aggregate --profile vdbo+gp \
  --musdb18hq-path /path/to/musdb18hq \
  --output ./data

# Dry run — preview without writing
mss-aggregate --musdb18hq-path /path/to/musdb18hq --dry-run
```

## Output Format

ZFTurbo Type 2 layout — one folder per stem:

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
| `--verbose` | off | Debug logging |

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
mss-aggregate --config config.yaml
```

CLI flags override config file values.

## License

MIT
