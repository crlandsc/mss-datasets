# MSS Datasets — Specification v0.3

## 1. Overview & Goals

A pip-installable CLI tool that aggregates multiple music source separation (MSS) datasets into a unified, stem-mapped output. The user runs a single command, points it at their downloaded datasets, and gets organized stem folders ready for training.

**Core philosophy: maximize data per stem.** Each output stem folder (e.g., `vocals/`, `drums/`) contains every available file for that instrument category across all source datasets. There is **no 1:1 mapping** between tracks and stems — a track contributes only the stems it actually contains. One folder may have 500 files, another may have 400. The training dataloader is responsible for handling this asymmetry (e.g., sampling random chunks per stem independently).

**This tool is NOT a training framework.** It does not mix stems, normalize loudness for training, or provide a dataloader. It prepares and organizes raw stem data. Everything downstream is the user's responsibility.

---

## 2. Prior Art

No existing tool aggregates multiple MSS datasets into a unified, stem-mapped, normalized output.

- `musdb` Python package: MUSDB18 only.
- `moisesdb` Python library: MoisesDB only.
- `medleydb` Python package: MedleyDB only.
- ZFTurbo's Music-Source-Separation-Training: Training framework that assumes data is already organized into Type 1 (MUSDB-style) or Type 2 (Stems-style) layouts. Does not download, normalize, deduplicate, or map stems.
- Spleeter, Demucs, Open-Unmix, ByteSep: Inference/training tools, not dataset preparation.

**This tool is novel.** No one has built a unified downloader + normalizer + stem-mapper + deduplicator for MSS datasets.

---

## 3. Source Datasets

### 3.1 MUSDB18-HQ

150 full-length tracks (~10h), stereo, 44.1 kHz uncompressed WAV. 100 train / 50 test. Native stems: `vocals`, `drums`, `bass`, `other` (VDBO format, zero mapping needed).

Hosted on Zenodo, requires access request (academic use only). The tool cannot auto-download — user provides path to downloaded archive.

**Ingestion approach:** Read WAVs directly with `soundfile`. The `musdb` Python package is **not** used.

**Limitation for 6-stem mode:** The `other` stem is pre-mixed (guitar + piano + everything else baked together). Cannot extract guitar or piano. The 104 unique MUSDB18-HQ tracks (those not from MedleyDB) contribute only 4 stems even in 6-stem mode. Metadata flags `musdb18hq_4stem_only: true` for these tracks.

### 3.2 MoisesDB

240 tracks (~14.4h) from 45 artists, 12 genres. Stereo, 44.1 kHz WAV. 11 top-level stems, each with sub-stems (see §5 for full taxonomy). Stems computed on-the-fly from individual source files via the `moisesdb` Python library.

Free for non-commercial research. No overlap with MUSDB18 or MedleyDB.

The library provides built-in stem grouping presets (`mix_4_stems`, `mix_6_stems`), but we use custom mappings that group `percussion` with `drums` instead of `other` (see §5.1).

### 3.3 MedleyDB v1 + v2

196 tracks total (122 v1 + 74 v2). Stereo stems, 44.1 kHz, 16-bit WAV. Three-level hierarchy:
- **Raw tracks**: mono, individual mic recordings
- **Stems**: stereo, grouped by instrument, with effects applied
- **Mix**: final stereo mixdown

**We ingest at the stem level** (stereo, grouped). Not raw tracks.

Stem labeling is free-form — stems are numbered, not named by role. Each stem has instrument metadata in per-track YAML files with labels like "female singer," "clean electric guitar," "drum set," etc. Requires a comprehensive lookup table to map ~121 unique instrument labels to our target categories (see §5).

**Ingestion approach:** Read WAVs with `soundfile` and parse `*_METADATA.yaml` with `pyyaml` directly. The `medleydb` Python package is **not** used.

Free for non-commercial research (CC BY-NC-SA 4.0).

### 3.4 Cross-Dataset Overlap

**46 of MUSDB18's 150 tracks originate from MedleyDB.** Naive inclusion of both datasets duplicates these 46 songs.

Resolution: **MedleyDB takes priority for the 46 overlapping tracks.** MedleyDB has per-instrument stems that can be mapped to any profile (4-stem or 6-stem), whereas MUSDB18-HQ's `other` stem is pre-mixed and cannot be decomposed. Using MedleyDB for these 46 tracks gives us 46 additional tracks with guitar/piano stems in 6-stem mode that would otherwise be lost.

The tool maintains a **hardcoded overlap registry** listing these 46 track identifiers. When both datasets are provided:
- The 46 overlapping tracks are sourced from **MedleyDB** (more granular stems)
- MUSDB18-HQ contributes its **104 unique tracks** (those not from MedleyDB)
- **MUSDB18-HQ's split assignments** (train/test) are inherited by the 46 overlapping tracks regardless of source — these splits are canonical for benchmarking
- The overlap is logged in `overlap_registry.json`

MoisesDB has zero overlap with either dataset.

No audio fingerprinting needed — the overlap is fully documented and static.

**Hardcoded overlap list** (46 MUSDB18-HQ track names, format: "Artist - Title"):
```
A Classic Education - NightOwl
Aimee Norwich - Child
Alexander Ross - Goodbye Bolero
Alexander Ross - Velvet Curtain
Auctioneer - Our Future Faces
AvaLuna - Waterduct
BigTroubles - Phantom
Celestial Shore - Die For Us
Clara Berry And Wooldog - Air Traffic
Clara Berry And Wooldog - Stella
Clara Berry And Wooldog - Waltz For My Victims
Creepoid - OldTree
Dreamers Of The Ghetto - Heavy Love
Faces On Film - Waiting For Ga
Grants - PunchDrunk
Helado Negro - Mitad Del Mundo
Hezekiah Jones - Borrowed Heart
Hop Along - Sister Cities
Invisible Familiars - Disturbing Wildlife
Lushlife - Toynbee Suite
Matthew Entwistle - Dont You Ever
Meaxic - Take A Step
Meaxic - You Listen
Music Delta - 80s Rock
Music Delta - Beatles
Music Delta - Britpop
Music Delta - Country1
Music Delta - Country2
Music Delta - Disco
Music Delta - Gospel
Music Delta - Grunge
Music Delta - Hendrix
Music Delta - Punk
Music Delta - Reggae
Music Delta - Rock
Music Delta - Rockabilly
Night Panther - Fire
Port St Willow - Stay Even
Secret Mountains - High Horse
Snowmine - Curfews
Steven Clark - Bounty
Strand Of Oaks - Spacestation
Sweet Lights - You Let Me Down
The Districts - Vermont
The Scarlet Brand - Les Fleurs Du Mal
The So So Glos - Emergency
```
Source: [MUSDB18 tracklist CSV](https://github.com/sigsep/website/blob/master/content/datasets/assets/tracklist.csv)

**Name matching between datasets:** MUSDB18-HQ directories use `"Artist - Title"` format, while MedleyDB directories use `"ArtistName_TrackName"` format (CamelCase, underscore-separated, no spaces). To match overlap tracks across datasets, normalize both names to a canonical form: lowercase, strip all spaces/underscores/hyphens, then compare. For example:
- MUSDB18: `"A Classic Education - NightOwl"` → `"aclassiceducationnightowl"`
- MedleyDB: `"AClassicEducation_NightOwl"` → `"aclassiceducationnightowl"`

This normalization is used only for overlap detection — the original names are preserved in metadata.

### 3.5 Excluded from v1

- **Slakh2100**: Synthetic (MIDI-rendered). Performs poorly for real-world MSS training. Exclude.
- **ACMID**: Solo-instrument crawled data without mixtures. Exclude.

Architecture should support adding new datasets via a plugin/adapter pattern for future expansion.

---

## 4. Stem Profiles

### 4.1 VDBO (4-stem) — Default

| Output Stem | Description |
|---|---|
| `vocals` | Singing, rapping, speaking, beatboxing, choir |
| `drums` | Drum kit, unpitched/a-tonal percussion (shakers, congas, tambourine, etc.), drum machines |
| `bass` | Bass guitar, double bass, bass synthesizer |
| `other` | Everything else: guitar, piano, keys, strings, winds, brass, pitched/tonal percussion (vibraphone, marimba, etc.), FX, etc. |

### 4.2 VDBO+GP (6-stem) — Extended

Splits guitar and piano out of "other," matching the MoisesDB benchmark and Demucs `htdemucs_6s` output.

| Output Stem | Description |
|---|---|
| `vocals` | Same as VDBO |
| `drums` | Same as VDBO |
| `bass` | Same as VDBO |
| `guitar` | Acoustic guitar, electric guitar (clean/distorted), lap steel, slide guitar |
| `piano` | Grand piano, electric piano, tack piano |
| `other` | Everything not covered above: strings, winds, brass, organ, synths, pitched/tonal percussion, FX, etc. |

**Note:** "Piano" is intentionally narrow (acoustic/electric piano only). Organs, synthesizers, harpsichords, and other keyboard instruments map to `other`, consistent with MoisesDB's own taxonomy where `other_keys` is separate from `piano`.

### 4.3 Future Profiles (Out of Scope for v1)

7-stem (acoustic/electric guitar split), 8-stem (percussion separate from drums), etc. The config-driven mapping approach makes adding these straightforward.

---

## 5. Instrument Mapping Tables

### 5.1 MoisesDB → Output Stems

MoisesDB has 11 top-level stems with 44 sub-stems. The `moisesdb` library handles sub-stem → top-level aggregation internally. We map top-level stems directly for most categories, but handle `percussion` and `bass` at the **sub-stem level** to route their components correctly (see routing dicts below).

**Full MoisesDB sub-stem taxonomy (for reference):**

| Top-Level Stem | Sub-Stems |
|---|---|
| `vocals` | lead male singer, lead female singer, human choir, background vocals, other (vocoder, beatboxing etc) |
| `drums` | snare drum, toms, kick drum, cymbals, overheads, full acoustic drumkit, drum machine |
| `bass` | bass guitar, bass synthesizer (moog etc), contrabass/double bass, tuba (bass of brass), bassoon (bass of woodwind) |
| `guitar` | clean electric guitar, distorted electric guitar, lap steel guitar or slide guitar, acoustic guitar |
| `piano` | grand piano, electric piano (rhodes, wurlitzer, piano sound alike) |
| `other_keys` | organ/electric organ, synth pad, synth lead, other sounds (harpsichord, mellotron etc) |
| `bowed_strings` | violin (solo), viola (solo), cello (solo), violin/viola/cello section, string section, other strings |
| `wind` | brass (trumpet, trombone, etc), flutes (piccolo, bamboo flute, etc), reeds (saxophone, clarinet, oboe, etc), other wind |
| `other_plucked` | banjo, mandolin, ukulele, harp etc |
| `percussion` | a-tonal percussion (claps, shakers, congas, etc), pitched percussion (mallets, glockenspiel, etc) |
| `other` | fx/processed sound, scratches, click track |

**VDBO mapping:**

| Output | MoisesDB Sources |
|---|---|
| `vocals` | vocals |
| `drums` | drums + percussion/**a-tonal** sub-stem |
| `bass` | bass/**bass guitar, bass synth, contrabass** sub-stems |
| `other` | guitar + piano + other_keys + bowed_strings + wind + other_plucked + percussion/**pitched** sub-stem + bass/**tuba, bassoon** sub-stems + other |

**Note:** This diverges from `moisesdb.defaults.mix_4_stems` in two ways:
1. **Percussion split**: The default routes all percussion to `other`. We split at sub-stem level — a-tonal → drums, pitched → other.
2. **Bass split**: The default routes all bass sub-stems to `bass`. We split because `tuba` and `bassoon` are brass/woodwind instruments that happen to be in the bass register — they belong in `other`, not `bass` in the MSS sense.

Both `percussion` and `bass` require accessing sub-stems via `track.stem_sources_mixture()` rather than using `track.mix_stems()`. All other top-level stems use the top-level mapping:

```python
# Top-level mapping (used for all stems EXCEPT percussion and bass)
custom_vdbo = {
    "vocals": ["vocals"],
    "drums": ["drums"],
    "other": ["other", "guitar", "other_plucked", "piano", "other_keys", "bowed_strings", "wind"],
}

# Percussion sub-stem routing (handled separately via track.stem_sources_mixture("percussion"))
percussion_routing = {
    "a-tonal percussion (claps, shakers, congas, cowbell etc)": "drums",
    "pitched percussion (mallets, glockenspiel, ...)": "other",
}

# Bass sub-stem routing (handled separately via track.stem_sources_mixture("bass"))
bass_routing = {
    "bass guitar": "bass",
    "bass synthesizer (moog etc)": "bass",
    "contrabass/double bass (bass of instrings)": "bass",
    "tuba (bass of brass)": "other",
    "bassoon (bass of woodwind)": "other",
}
```

**VDBO+GP mapping:**

| Output | MoisesDB Sources |
|---|---|
| `vocals` | vocals |
| `drums` | drums + percussion/**a-tonal** sub-stem |
| `bass` | bass/**bass guitar, bass synth, contrabass** sub-stems |
| `guitar` | guitar |
| `piano` | piano |
| `other` | other_keys + bowed_strings + wind + other_plucked + percussion/**pitched** sub-stem + bass/**tuba, bassoon** sub-stems + other |

```python
# Top-level mapping (used for all stems EXCEPT percussion and bass)
custom_vdbo_gp = {
    "vocals": ["vocals"],
    "drums": ["drums"],
    "guitar": ["guitar"],
    "piano": ["piano"],
    "other": ["other", "other_plucked", "other_keys", "bowed_strings", "wind"],
}

# Percussion and bass sub-stem routing dicts are the same for both profiles (see VDBO above)
```

**Important:** The string keys in `percussion_routing` and `bass_routing` must exactly match the sub-stem names returned by `track.stem_sources_mixture()`. The keys above are from MoisesDB's source code; verify against actual output during Phase 6 development. If a key doesn't match, the sub-stem audio will be silently missed.

### 5.2 MedleyDB → Output Stems

MedleyDB uses ~121 unique instrument labels across v1 and v2 (definitive source: `instrument_f0_type.json`). Each stem's instrument label determines its mapping.

When multiple stems in a track map to the same output category, they are **summed** into a single output file. When no stems map to a category, **no file is created** for that track in that folder.

#### Mapping: → `vocals`

| MedleyDB Label | Notes |
|---|---|
| `male singer` | |
| `female singer` | |
| `male speaker` | |
| `female speaker` | |
| `male rapper` | |
| `female rapper` | |
| `male screamer` | |
| `female screamer` | |
| `vocalists` | Group vocal |
| `choir` | |
| `beatboxing` | Vocal technique — produced by voice, captured by vocal mic |

**11 labels → vocals**

#### Mapping: → `drums`

Drum kits, drum machines, and **a-tonal (unpitched) percussion** only. Pitched/tonal percussion (vibraphone, marimba, etc.) maps to `other` — see below.

| MedleyDB Label | Category |
|---|---|
| `drum set` | Drum kit |
| `drum machine` | Electronic — plays drum role |
| `kick drum` | Drum kit component |
| `snare drum` | Drum kit component |
| `bass drum` | Drum (concert/marching) |
| `toms` | Drum kit component |
| `timpani` | Drum (unpitched in practice despite tuning) |
| `bongo` | Hand drum |
| `conga` | Hand drum |
| `darbuka` | Hand drum |
| `doumbek` | Hand drum |
| `tabla` | Hand drum |
| `tambourine` | A-tonal percussion |
| `auxiliary percussion` | A-tonal percussion |
| `high hat` | A-tonal percussion |
| `cymbal` | A-tonal percussion |
| `gong` | A-tonal percussion |
| `triangle` | A-tonal percussion |
| `cowbell` | A-tonal percussion |
| `sleigh bells` | A-tonal percussion |
| `cabasa` | A-tonal percussion |
| `guiro` | A-tonal percussion |
| `gu` | A-tonal percussion |
| `castanet` | A-tonal percussion |
| `claps` | A-tonal percussion |
| `rattle` | A-tonal percussion |
| `shaker` | A-tonal percussion |
| `maracas` | A-tonal percussion |
| `snaps` | A-tonal percussion |

**29 labels → drums**

#### Mapping: → `bass`

| MedleyDB Label | Notes |
|---|---|
| `electric bass` | Standard bass guitar |
| `double bass` | Upright/acoustic bass (listed under "Strings — Bowed" in MedleyDB taxonomy, but functionally bass) |

**2 labels → bass**

Note: `bass clarinet`, `bassoon`, and `tuba` are wind instruments that play in the bass register but are **not** "bass" in the MSS sense. They map to `other`.

#### Mapping: → `guitar` (VDBO+GP only; → `other` in VDBO)

| MedleyDB Label |
|---|
| `acoustic guitar` |
| `clean electric guitar` |
| `distorted electric guitar` |
| `lap steel guitar` |
| `slide guitar` |

**5 labels → guitar (6-stem) or other (4-stem)**

#### Mapping: → `piano` (VDBO+GP only; → `other` in VDBO)

| MedleyDB Label |
|---|
| `piano` |
| `tack piano` |
| `electric piano` |

**3 labels → piano (6-stem) or other (4-stem)**

#### Mapping: → `other`

Everything not listed above. Organized by MedleyDB taxonomy group:

**Pitched/tonal percussion** (5): `xylophone`, `vibraphone`, `marimba`, `glockenspiel`, `chimes`

**Bowed strings** (9): `erhu`, `violin`, `viola`, `cello`, `dilruba`, `violin section`, `viola section`, `cello section`, `string section`

**Plucked strings** (10): `banjo`, `guzheng`, `harp`, `harpsichord`, `liuqin`, `mandolin`, `oud`, `sitar`, `ukulele`, `zhongruan`

**Struck strings** (2): `dulcimer`, `yangqin`

**Flutes** (7): `dizi`, `flute`, `flute section`, `piccolo`, `bamboo flute`, `panpipes`, `recorder`

**Single reeds** (7): `alto saxophone`, `baritone saxophone`, `bass clarinet`, `clarinet`, `clarinet section`, `tenor saxophone`, `soprano saxophone`

**Double reeds** (4): `oboe`, `english horn`, `bassoon`, `bagpipe`

**Brass** (11): `trumpet`, `cornet`, `trombone`, `french horn`, `euphonium`, `tuba`, `brass section`, `french horn section`, `trombone section`, `horn section`, `trumpet section`

**Free reeds** (7): `harmonica`, `concertina`, `accordion`, `bandoneon`, `harmonium`, `pipe organ`, `melodica`

**Electronic** (6): `synthesizer`, `theremin`, `fx/processed sound`, `scratch`, `sampler`, `electronic organ`

**Voices** (1): `crowd`

**Special handling:**
- `Main System` → **EXCLUDE this stem only.** Other stems from the same track are processed normally. This is an ensemble/room mic capturing the full mix — it cannot be meaningfully separated.
- `Unlabeled` → `other`, with `unlabeled: true` flag in metadata.

**Note:** All MedleyDB instrument labels are lowercase except `Main System` and `Unlabeled`. Lookup uses **case-insensitive matching** (normalize to lowercase before comparison).

**69 labels → other** (plus 5 guitar + 3 piano in VDBO mode = 77)

#### Edge Case Rationale

| Label | Decision | Reasoning |
|---|---|---|
| `beatboxing` | vocals | Vocal technique, captured by vocal mic |
| `drum machine` | drums | Plays the drum/rhythm role |
| `double bass` | bass | Plays bass role regardless of production method |
| `crowd` | other | Ambient, not primary vocal content |
| `synthesizer` | other | Too ambiguous (could be bass, lead, pad) |
| `harpsichord` | other | Keyboard but not "piano" — matches MoisesDB `other_keys` |
| `pipe organ` | other | Organ, not piano — matches MoisesDB `other_keys` |
| `bass clarinet` | other | Wind instrument, not "bass" in MSS sense |
| `bassoon` | other | Wind instrument, not "bass" in MSS sense |
| `tuba` | other | Brass instrument, not "bass" in MSS sense |
| `timpani` | drums | Unpitched in practice despite being tunable |
| `vibraphone` | other | Pitched/tonal — melodic instrument, not rhythmic |
| `xylophone` | other | Pitched/tonal — melodic instrument, not rhythmic |
| `marimba` | other | Pitched/tonal — melodic instrument, not rhythmic |
| `glockenspiel` | other | Pitched/tonal — melodic instrument, not rhythmic |
| `chimes` | other | Pitched/tonal — melodic instrument, not rhythmic |
| `Main System` | EXCLUDE stem | Ensemble mic — skip this stem, process others normally |

### 5.3 MUSDB18-HQ → Output Stems

MUSDB18-HQ contributes only its **104 unique tracks** (those not sourced from MedleyDB). No mapping needed — files are already `vocals.wav`, `drums.wav`, `bass.wav`, `other.wav`. Direct copy.

In VDBO+GP mode, these 104 tracks contribute only to `vocals/`, `drums/`, `bass/`, `other/`. No files are created in `guitar/` or `piano/` for these tracks. Metadata flags `musdb18hq_4stem_only: true`.

---

## 6. Output Format

**ZFTurbo Type 2 (Stems-style):** One folder per stem category containing WAV files.

```
output/
├── vocals/
│   ├── musdb18hq_train_0001_a_classic_education_night_owl.wav
│   ├── moisesdb_train_0001_artist_name_track_name.wav
│   ├── medleydb_train_0001_liz_nelson_rainfall.wav
│   └── ...  (up to ~505 files)
├── drums/
│   └── ...  (up to ~480 files)
├── bass/
│   └── ...  (up to ~475 files)
├── other/
│   └── ...  (up to ~480 files)
├── guitar/           # only in VDBO+GP mode
│   └── ...  (~350 files — fewer, since MUSDB18-HQ can't contribute)
├── piano/            # only in VDBO+GP mode
│   └── ...  (~300 files — fewer)
└── metadata/
    ├── manifest.json
    ├── splits.json
    ├── overlap_registry.json
    ├── errors.json
    └── config.yaml       # effective configuration for reproducibility
```

**Key: stem folders have independent file counts.** A track contributes only the stems it actually contains. Not every track appears in every folder. The training dataloader must handle this (e.g., sample per-stem independently).

**Optional: `--group-by-dataset`** — When enabled, each stem folder gets subfolders by source dataset:

```
output/
├── vocals/
│   ├── musdb18hq/
│   │   └── musdb18hq_train_0001_a_classic_education_night_owl.wav
│   ├── moisesdb/
│   │   └── moisesdb_train_0001_artist_name_track_name.wav
│   └── medleydb/
│       └── medleydb_train_0001_liz_nelson_rainfall.wav
├── drums/
│   ├── musdb18hq/
│   ├── moisesdb/
│   └── medleydb/
└── ...
```

Default is **off** (flat stem folders) for ZFTurbo Type 2 compatibility. When on, dataloaders must glob recursively (e.g., `vocals/**/*.wav`). Filenames are identical in both modes — only the directory nesting changes.

### 6.1 Audio Format

All output files: **44.1 kHz, float32, stereo WAV**.

Float32 avoids clipping when stems are summed and preserves full dynamic range. No dithering needed since we are not reducing bit depth (MUSDB18-HQ is already float32; MedleyDB 16-bit stems are losslessly promoted to float32).

### 6.2 Filename Convention

Format: `{source}_{split}_{index:04d}_{artist}_{title}.wav`

The `{index}` is a 1-based sequential counter, scoped per source dataset. Each dataset (musdb18hq, moisesdb, medleydb) has its own independent counter starting at 0001. Tracks are numbered in the order they are discovered (alphabetical by directory name). The index is assigned once during discovery and is the same across all stem folders for that track.

Examples:
- `musdb18hq_train_0001_a_classic_education_night_owl.wav`
- `moisesdb_train_0042_artist_name_track_title.wav`
- `medleydb_val_0003_liz_nelson_rainfall.wav`

**Sanitization rules:**
1. Transliterate Unicode to ASCII via `unidecode` (e.g., ü → u, é → e, ñ → n)
2. Lowercase everything
3. Replace spaces and non-alphanumeric characters (except hyphens) with underscores
4. Collapse consecutive underscores to single underscore
5. Strip leading/trailing underscores
6. Truncate artist + title combined to 80 characters max
7. **Collision resolution**: If the sanitized filename collides with an existing file within the same stem folder, append `_2`, `_3`, etc. until unique. Since each track gets a unique `(source, index)` tuple, collisions should be rare.

### 6.3 Mixture Files

Not generated by default. Available via `--include-mixtures` flag, which creates a `mixtures/` folder. Mixtures are computed as the linear sum of all stems for each track (not downloaded from source, for consistency). Only tracks that have all stems in the selected profile get a mixture file.

### 6.4 Metadata

**`manifest.json`**: Per-file records:
```json
{
  "musdb18hq_train_0001_a_classic_education_night_owl": {
    "source_dataset": "musdb18hq",
    "original_track_name": "A Classic Education - Night Owl",
    "artist": "A Classic Education",
    "title": "Night Owl",
    "split": "train",
    "available_stems": ["vocals", "drums", "bass", "other"],
    "profile": "vdbo",
    "license": "academic-use-only",
    "duration_seconds": 245.3,
    "is_composite_sum": false,
    "has_bleed": false,
    "musdb18hq_4stem_only": true,
    "flags": []
  }
}
```

Possible flags: `"silent_stem"`, `"below_noise_floor"`, `"has_bleed"`, `"unlabeled_source"`, `"composite_sum"` (stem was created by summing multiple source stems).

**`splits.json`**: Train/val/test assignments per track. Locked on first run, never changes.

**`overlap_registry.json`**: Documents which MUSDB18-HQ tracks were skipped (sourced from MedleyDB instead) due to cross-dataset overlap.

**`errors.json`**: Any tracks that were skipped due to errors (corrupted files, malformed metadata, etc.) with error details.

---

## 7. Processing Pipeline

### 7.1 Stage 1: Acquire

The tool ingests from user-provided local paths. No automatic downloading — all three datasets require manual access requests.

```
mss-datasets \
  --musdb18hq-path /path/to/musdb18hq \
  --moisesdb-path /path/to/moisesdb \
  --medleydb-path /path/to/medleydb \
  --output ./data
```

For each provided path, the tool validates the expected directory structure and reports errors if the layout doesn't match expectations. Datasets not provided are silently skipped.

**Expected directory structures:**
- **MUSDB18-HQ**: `{path}/train/` and `{path}/test/` directories, each containing track subdirectories named `"Artist - Title"`. Each track subdir contains `vocals.wav`, `drums.wav`, `bass.wav`, `other.wav`, and `mixture.wav`.
- **MoisesDB**: `{path}/moisesdb_v0.1/` containing UUID-named subdirectories (one per track). Parsed via the `moisesdb` library.
- **MedleyDB**: `{path}/Audio/` containing subdirectories named `{ArtistName_TrackName}/`. Each track subdir contains `{TrackName}_METADATA.yaml` and a `{TrackName}_STEMS/` directory with `{TrackName}_STEM_{NN}.wav` files.

### 7.2 Stage 2: Deduplicate

The hardcoded overlap registry identifies the 46 MedleyDB tracks present in MUSDB18-HQ.

**When both MUSDB18-HQ and MedleyDB are provided:**
- **MedleyDB versions are used** for the 46 overlapping tracks (more granular stems)
- MUSDB18-HQ copies of those 46 tracks are **skipped**
- MUSDB18-HQ's split assignments are inherited by the MedleyDB versions
- Logged in `overlap_registry.json`

**When only MUSDB18-HQ is provided (no MedleyDB):** All 150 tracks are processed. The overlap registry is not consulted. `overlap_registry.json` is written empty.

**When only MedleyDB is provided (no MUSDB18-HQ):** All 196 tracks are processed. The overlap registry is not consulted (no MUSDB18-HQ splits to inherit — all MedleyDB tracks treated as training data per §10.3). `overlap_registry.json` is written empty.

**MoisesDB:** Always processes all 240 tracks regardless of what other datasets are present (zero overlap with either).

### 7.3 Stage 3: Stem Map

For each track in each dataset, apply the mapping from §5:

**MUSDB18-HQ**: Direct copy for the 104 unique tracks. Files are already `vocals.wav`, `drums.wav`, `bass.wav`, `other.wav`.

**MoisesDB**: For each track:
1. Use `track.mix_stems()` with our custom mapping dict (see §5.1) for all top-level stems **except `percussion` and `bass`**
2. For the `percussion` stem, use `track.stem_sources_mixture("percussion")` to access sub-stems individually. Route `a-tonal percussion` → sum into drums output, `pitched percussion` → sum into other output
3. For the `bass` stem, use `track.stem_sources_mixture("bass")` to access sub-stems individually. Route `bass guitar`, `bass synthesizer`, `contrabass` → sum into bass output; route `tuba`, `bassoon` → sum into other output
4. Sum all sub-stem routing results into the appropriate output arrays alongside the top-level mix_stems results
5. Only write output files for stems that have non-silent audio content

**MedleyDB**: For each track:
1. Read the YAML metadata to get instrument labels per stem
2. Apply the lookup table from §5.2 to determine target category (case-insensitive matching)
3. **Filter**: Skip stems labeled `Main System` (exclude only that stem). Route `Unlabeled` stems to `other`, flag `unlabeled: true` in metadata
4. Load remaining stem audio files
5. Sum stems mapping to the same category (linear sum of sample arrays)
6. If all stems were filtered (e.g., track has only a `Main System` stem), the track produces no output files — log as warning in `errors.json`
7. Write output file per category. Skip categories with no contributing stems.

### 7.4 Stage 4: Normalize Audio Format

For every output stem file:
1. **Sample rate**: Resample to 44.1 kHz if needed (should be no-op for all three datasets)
2. **Bit depth**: Convert to float32
3. **Channels**: Ensure stereo. MedleyDB stems are already stereo at the stem level. If any mono files are encountered, duplicate to dual-mono.
4. **Format**: Write as WAV

Optional loudness normalization (EBU R128 to target LUFS) is **off by default**. Available via `--normalize-loudness --loudness-target -14`.

### 7.5 Stage 5: Validate

Post-processing checks for every output file:
- Valid WAV with correct sample rate, bit depth, channel count
- Flag silent files (all zeros or below noise floor) in metadata — keep the file but flag it, since some songs legitimately lack certain instruments at certain points
- Log per-stem file counts and verify they're within expected ranges

For MUSDB18-HQ tracks, optionally verify that stem sum matches the original mixture within floating-point tolerance (`--verify-mixtures`).

### 7.6 Stage 6: Write Metadata

Generate `manifest.json`, `splits.json`, `overlap_registry.json`, and `errors.json` as described in §6.4.

Print a summary report:
```
MSS Datasets — Complete
========================
Profile: VDBO (4-stem)
Datasets: musdb18hq (104 unique), moisesdb (240), medleydb (196, incl. 46 overlap)
Deduplicated: 46 tracks (MedleyDB preferred over MUSDB18-HQ)
Errors: 2 tracks skipped (see errors.json)

Output stem counts:
  vocals/  503 files
  drums/   478 files
  bass/    471 files
  other/   489 files

Total: 1,941 WAV files
Disk usage: ~152 GB
```

---

## 8. Resumability & Parallelism

### 8.1 Resumability

The tool must be safely interruptible and resumable. Implementation:

- **Atomic writes**: All output files are written to `{filename}.tmp` first, then atomically renamed to the final path. This ensures no partial/corrupted files exist on disk if the process is interrupted mid-write.
- Before writing each output file, check if it already exists with the expected file size
- If it exists and matches, skip it
- If it exists but is wrong size (interrupted write), delete and re-process
- Clean up any `.tmp` files on startup (leftovers from interrupted writes)
- Progress state is implicit in the filesystem — no separate checkpoint file needed
- On restart, the tool scans existing output, reports "N of M files already complete," and continues from where it left off

### 8.2 Parallelism

Audio I/O and resampling are the bottleneck. Support parallel processing:

- `--workers N` flag (default: 1, i.e., sequential)
- Uses `multiprocessing.Pool` or `concurrent.futures.ProcessPoolExecutor`
- Each worker processes one track at a time (read stems → map → normalize → write)
- No shared state between workers; each writes to independent output files
- Progress bar updates from all workers

---

## 9. Error Handling

**Policy: skip and log.** A single corrupted file should not abort processing of hundreds of tracks.

- Corrupted WAV files → skip track, log to `errors.json`
- Malformed MedleyDB YAML → skip track, log
- Missing expected stem files → skip that stem for that track, log
- Unexpected instrument labels (not in lookup table) → map to `other`, log warning
- Disk full → abort with clear error message
- Permission errors → abort with clear message

`errors.json` format:
```json
[
  {
    "track": "medleydb_LizNelson_Rainfall",
    "dataset": "medleydb",
    "error": "Corrupted WAV: unexpected EOF at byte 1024",
    "stage": "stem_map",
    "skipped": true
  }
]
```

---

## 10. Split Management

Two validation sets are held out — one for 4-stem benchmarking (MUSDB18-HQ) and one for 6-stem benchmarking (MoisesDB). Everything else trains.

### 10.1 MUSDB18-HQ

Canonical and inviolable: **100 train / 50 test**. The 50-song test set must never appear in training data. This is the primary 4-stem (VDBO) benchmark.

The 46 MedleyDB tracks that overlap with MUSDB18-HQ inherit these split assignments regardless of which dataset provides the audio.

### 10.2 MoisesDB

**50 tracks held out for validation**, remainder (~190) for training. This validation set is the primary 6-stem (VDBO+GP) benchmark — it's the only way to evaluate guitar/piano separation without data leakage.

**Split strategy**: Deterministic selection using a fixed random seed (`seed=42`) with genre stratification (proportional representation of each genre in the val set). The resulting 50 track IDs are hardcoded after initial generation — subsequent runs use the hardcoded list, not re-computation. This ensures reproducibility even if the random library implementation changes.

### 10.3 MedleyDB

All unique MedleyDB tracks (those not overlapping with MUSDB18-HQ) go to **training**. MedleyDB has no established benchmark validation set, and maximizing training data is the priority. The `artist_conditional_split` function is not used — instead, we ensure no artist leakage by verifying that MedleyDB training artists don't appear in the MUSDB18-HQ test set or MoisesDB validation set.

For the 46 MedleyDB tracks that overlap with MUSDB18-HQ, the MUSDB18-HQ split assignment takes precedence (see §10.1).

### 10.4 Locking

All split assignments are written to `splits.json` on first run. On subsequent runs (resumption), the existing `splits.json` is loaded and respected. The tool never changes split assignments after initial creation.

### 10.5 Split Behavior by Dataset Availability

When datasets are provided selectively:

- **MUSDB18-HQ only**: Canonical 100 train / 50 test split. No val set.
- **MedleyDB only**: All 196 tracks → train. No test or val set. Overlap registry not consulted.
- **MoisesDB only**: Deterministic 50-track val set; remainder → train. No test set.
- **MUSDB18-HQ + MedleyDB (no MoisesDB)**: Follow §10.1–§10.3. 104 unique MUSDB18 + 196 MedleyDB (46 overlap, MedleyDB preferred). MUSDB18 test split inherited. No val set.
- **MUSDB18-HQ + MoisesDB (no MedleyDB)**: All 150 MUSDB18 tracks processed (no dedup needed). MUSDB18 test + MoisesDB val = two independent eval sets.
- **MedleyDB + MoisesDB (no MUSDB18-HQ)**: All 196 MedleyDB → train. MoisesDB 50-track val set applied. No test set (no MUSDB18 splits to inherit — the 46 "overlap" tracks are just regular MedleyDB tracks).
- **All three**: Follow §10.1–§10.4 fully. Two eval sets: MUSDB18 test (4-stem) + MoisesDB val (6-stem).

The MoisesDB val set is orthogonal to the MUSDB18 test set — MoisesDB has zero overlap with either dataset, so no leakage risk.

### 10.6 Summary

| Dataset | Train | Val/Test | Purpose |
|---|---|---|---|
| MUSDB18-HQ (104 unique) | ~69 | ~35 | 4-stem benchmark (test split) |
| MedleyDB overlap (46) | ~31 | ~15 | Inherits MUSDB18-HQ splits |
| MedleyDB unique (~150) | ~150 | 0 | All training |
| MoisesDB (240) | ~190 | 50 | 6-stem benchmark (val split) |
| **Total** | **~440** | **~100** | |

---

## 11. CLI Interface

```bash
# Default: VDBO profile, all available datasets
mss-datasets \
  --musdb18hq-path /path/to/musdb18hq \
  --moisesdb-path /path/to/moisesdb \
  --medleydb-path /path/to/medleydb \
  --output ./data

# 6-stem extended profile
mss-datasets \
  --musdb18hq-path /path/to/musdb18hq \
  --moisesdb-path /path/to/moisesdb \
  --medleydb-path /path/to/medleydb \
  --profile vdbo+gp \
  --output ./data

# Only specific datasets (omit paths for datasets you don't have)
mss-datasets \
  --musdb18hq-path /path/to/musdb18hq \
  --moisesdb-path /path/to/moisesdb \
  --output ./data

# Parallel processing
mss-datasets \
  --musdb18hq-path /path/to/musdb18hq \
  --moisesdb-path /path/to/moisesdb \
  --medleydb-path /path/to/medleydb \
  --workers 8 \
  --output ./data

# Include mixture files
mss-datasets \
  --musdb18hq-path /path/to/musdb18hq \
  --profile vdbo \
  --include-mixtures \
  --output ./data

# Dry run — show what would be processed
mss-datasets \
  --musdb18hq-path /path/to/musdb18hq \
  --moisesdb-path /path/to/moisesdb \
  --medleydb-path /path/to/medleydb \
  --dry-run

# Validate existing output
mss-datasets --validate ./data

# Load from config file
mss-datasets --config config.yaml

# Normalize loudness (optional)
mss-datasets --musdb18hq-path ... --normalize-loudness --loudness-target -14
```

### Flags Summary

| Flag | Default | Description |
|---|---|---|
| `--musdb18hq-path` | — | Path to MUSDB18-HQ dataset |
| `--moisesdb-path` | — | Path to MoisesDB dataset |
| `--medleydb-path` | — | Path to MedleyDB dataset |
| `--output` | `./output` | Output directory |
| `--profile` | `vdbo` | Stem profile: `vdbo` (4-stem) or `vdbo+gp` (6-stem) |
| `--workers` | `1` | Number of parallel workers |
| `--include-mixtures` | `false` | Generate mixture files |
| `--group-by-dataset` | `false` | Add source dataset subfolders within each stem folder |
| `--normalize-loudness` | `false` | Apply EBU R128 loudness normalization |
| `--loudness-target` | `-14` | Target LUFS (requires `--normalize-loudness`) |
| `--verify-mixtures` | `false` | Verify stem sums match original mixtures |
| `--dry-run` | `false` | Show what would be processed without writing files |
| `--validate` | — | Validate an existing output directory |
| `--config` | — | Path to YAML config file |
| `--verbose` | `false` | Verbose logging |

---

## 12. Configuration

An optional YAML config file captures all parameters for reproducibility. Every CLI flag has a config equivalent. CLI flags override config file values.

```yaml
# config.yaml
datasets:
  musdb18hq_path: /path/to/musdb18hq
  moisesdb_path: /path/to/moisesdb
  medleydb_path: /path/to/medleydb

output: ./data
profile: vdbo
workers: 8
include_mixtures: false
group_by_dataset: false
normalize_loudness: false
loudness_target: -14
```

The tool writes its effective configuration to `metadata/config.yaml` in the output directory for reproducibility.

---

## 13. Package Structure & Dependencies

**Python ≥ 3.9 required.**

### 13.1 Package Layout

```
mss-datasets/
├── pyproject.toml
├── README.md
├── src/
│   └── mss_datasets/
│       ├── __init__.py
│       ├── cli.py              # CLI entry point (click or argparse)
│       ├── pipeline.py         # Main orchestration
│       ├── datasets/
│       │   ├── __init__.py
│       │   ├── base.py         # Abstract dataset adapter
│       │   ├── musdb18hq.py
│       │   ├── moisesdb_adapter.py  # Named to avoid collision with moisesdb package
│       │   └── medleydb.py
│       ├── mapping/
│       │   ├── __init__.py
│       │   ├── profiles.py     # VDBO, VDBO+GP definitions
│       │   └── medleydb_instruments.yaml  # The lookup table (auditable, not buried in code)
│       ├── audio.py            # Resampling, format conversion, summing
│       ├── utils.py            # Filename sanitization, collision resolution
│       ├── metadata.py         # Manifest, splits, error logging
│       ├── splits.py           # Split assignment and locking
│       ├── overlap.py          # Hardcoded MUSDB18↔MedleyDB overlap registry
│       └── validation.py       # Post-processing validation
├── tests/
│   ├── conftest.py             # Shared fixtures (synthetic WAVs)
│   ├── test_audio.py           # Audio I/O, format conversion, summing
│   ├── test_utils.py           # Filename sanitization, collision resolution
│   ├── test_mapping.py         # Instrument mapping for both profiles
│   ├── test_overlap.py         # Deduplication and overlap registry
│   ├── test_musdb18hq.py       # MUSDB18-HQ adapter
│   ├── test_medleydb.py        # MedleyDB adapter
│   ├── test_moisesdb.py        # MoisesDB adapter (mocked)
│   ├── test_splits.py          # Split assignment and locking
│   ├── test_metadata.py        # Metadata file generation
│   ├── test_pipeline.py        # End-to-end integration tests
│   ├── test_parallel.py        # Multi-worker consistency
│   ├── test_cli.py             # CLI flags and config loading
│   └── fixtures/               # Small synthetic test audio files
└── scripts/
    └── example_usage.py        # Minimal example script
```

Installable via:
```bash
pip install mss-datasets        # from PyPI
pip install -e .                 # from cloned repo (dev mode)
```

CLI entry point registered in `pyproject.toml`:
```toml
[project.scripts]
mss-datasets = "mss_datasets.cli:main"
```

### 13.2 Dependencies

**Core (required):**
- `soundfile` — WAV I/O (uses libsndfile)
- `numpy` — audio array manipulation
- `pyyaml` — config and MedleyDB metadata parsing
- `unidecode` — Unicode→ASCII transliteration for filename sanitization
- `tqdm` — progress bars (fallback)
- `click` — CLI framework

**Optional (enhanced UX):**
- `rich` — enhanced progress bars and terminal output (falls back to tqdm if not installed)

**Dataset libraries (optional):**
- `moisesdb` — MoisesDB parsing (only needed when processing MoisesDB)

**Note on MUSDB18-HQ and MedleyDB:** These datasets are read **directly** using `soundfile` (WAV I/O) and `pyyaml` (MedleyDB YAML metadata). The `musdb` and `medleydb` Python packages are **not required** and should not be listed as dependencies. This eliminates two dependency trees and avoids version/compatibility issues with those packages.

**Optional processing:**
- `pyloudnorm` — EBU R128 loudness normalization (only needed with `--normalize-loudness`)
- `soxr` or `resampy` — high-quality resampling (only if source audio isn't 44.1 kHz)

The tool should gracefully handle missing optional dependencies: if a user doesn't have `moisesdb` installed but provides `--moisesdb-path`, it errors with "Install moisesdb: pip install mss-datasets[moisesdb]".

---

## 14. Testing Strategy

### 14.1 Unit Tests

- **Instrument mapping**: Every MedleyDB label maps to the expected output stem for both VDBO and VDBO+GP profiles
- **Filename sanitization**: Edge cases (unicode, special chars, long names)
- **Overlap registry**: Correct identification of the 46 overlapping tracks
- **Audio conversion**: Bit depth promotion, mono→stereo, sample rate validation

### 14.2 Integration Tests

- **Small fixture set**: 2-3 synthetic tracks per dataset format (tiny WAV files, correct directory structure and metadata) to test the full pipeline end-to-end
- **Resumability**: Run pipeline, interrupt, re-run, verify no duplicates and correct output
- **Dry run**: Verify dry run output matches actual processing

### 14.3 Validation Tests

- **Output format**: All generated files are valid WAV, correct sample rate/channels/bit depth
- **Metadata consistency**: manifest.json matches actual files on disk
- **Split integrity**: No artist leakage across train/test

---

## 15. Expected Output

### 15.1 Track Counts

For a VDBO run with all three datasets:

| Dataset | Total Tracks | Unique (After Dedup) | Notes |
|---|---|---|---|
| MUSDB18-HQ | 150 | 104 | 46 overlap sourced from MedleyDB instead |
| MoisesDB | 240 | 240 | Most have V/D/B; guitar/piano vary |
| MedleyDB | 196 | 196 | All included (46 overlap tracks use MedleyDB source) |
| **Total** | **586** | **~540** | |

**Per-stem file counts (VDBO, approximate):**

| Stem | Est. Files | Notes |
|---|---|---|
| `vocals` | ~500 | Most tracks have vocals |
| `drums` | ~490 | Most tracks have drums/percussion (higher now with percussion→drums) |
| `bass` | ~470 | Most tracks have bass |
| `other` | ~480 | Nearly all tracks have non-V/D/B content |
| **Total files** | **~1,940** | |

For VDBO+GP, add:

| Stem | Est. Files | Notes |
|---|---|---|
| `guitar` | ~390 | 46 formerly MUSDB18-HQ tracks now contribute via MedleyDB |
| `piano` | ~320 | 46 formerly MUSDB18-HQ tracks now contribute via MedleyDB |

### 15.2 Disk Space Estimates

Audio storage: 44.1 kHz × float32 × stereo = **~345 KB/sec** = **~20.7 MB/min**

| Profile | Est. Total Duration (all stems) | Est. Disk Size |
|---|---|---|
| VDBO (4-stem) | ~120h across all stem files | **~150 GB** |
| VDBO+GP (6-stem) | ~140h across all stem files | **~175 GB** |

**Source datasets on disk** (user must also have space for these):
- MUSDB18-HQ: ~30 GB
- MoisesDB: ~30 GB
- MedleyDB: ~40 GB

**Total disk space needed: ~250–275 GB** (source + output).

The CLI should print estimated disk usage during `--dry-run` and warn if insufficient space is detected before processing begins.

### 15.3 Estimated Processing Time

Highly dependent on disk I/O speed and `--workers` count. Rough estimate for SSD:
- Sequential (`--workers 1`): ~30–60 minutes
- Parallel (`--workers 8`): ~5–15 minutes

---

## 16. MedleyDB Bleed Handling

MedleyDB tracks have a `has_bleed` flag indicating microphone bleed between stems. The tool:

1. **Includes** bleed tracks by default (bleed is common in real recordings and arguable makes training data more realistic)
2. Flags `has_bleed: true` in manifest.json for affected tracks
3. Users can filter these out in their dataloader if they want clean-only stems

MoisesDB also has per-source bleed annotations (`track.bleedings`). These are similarly recorded in metadata.

---

## 17. Implementation Phases

Each phase is self-contained and independently testable. An agent can pick up at any phase boundary with no context loss — run `pytest` on prior phases to verify the foundation.

### Phase 0: Project Scaffolding
**Goal**: Installable package with working CLI entry point.
- [ ] Create `pyproject.toml` (name: `mss-datasets`, Python ≥3.9, deps: soundfile, numpy, pyyaml, unidecode, tqdm, click)
- [ ] Create `src/mss_datasets/__init__.py` with `__version__`
- [ ] Create `src/mss_datasets/cli.py` with stub `main()` that prints version
- [ ] Create package subdirs: `datasets/`, `mapping/`
- [ ] Create `tests/conftest.py` and `tests/fixtures/` (generate 1-sec stereo 44.1kHz WAVs: one float32, one int16, one mono)
- [ ] Verify: `pip install -e . && mss-datasets --help`

### Phase 1: Audio Utilities & Filename Sanitization
**Goal**: Core audio I/O and string utils, fully tested.
**Files**: `src/mss_datasets/audio.py`, `src/mss_datasets/utils.py`
- [ ] `audio.read_wav(path) → (np.ndarray, int)`: Read WAV, return (samples, sr). Shape: (n_samples, n_channels).
- [ ] `audio.write_wav_atomic(path, data, sr)`: Write to `.tmp`, rename. Float32 output.
- [ ] `audio.ensure_stereo(data) → np.ndarray`: Mono→dual-mono, passthrough stereo.
- [ ] `audio.ensure_float32(data) → np.ndarray`: int16/int32→float32 promotion.
- [ ] `audio.sum_stems(list[np.ndarray]) → np.ndarray`: Sum multiple arrays, handle length mismatches (zero-pad shorter to longest).
- [ ] `utils.sanitize_filename(source, split, index, artist, title) → str`: Full sanitization pipeline per §6.2.
- [ ] `utils.resolve_collision(filename, existing_set) → str`: Append `_2`, `_3` if collision.
- [ ] Tests: `tests/test_audio.py`, `tests/test_utils.py`
- [ ] Verify: `pytest tests/test_audio.py tests/test_utils.py -v`

### Phase 2: Mapping & Profile Definitions
**Goal**: All instrument mappings defined and tested.
**Files**: `src/mss_datasets/mapping/profiles.py`, `src/mss_datasets/mapping/medleydb_instruments.yaml`, `src/mss_datasets/mapping/__init__.py`
- [ ] Define `VDBO` and `VDBO_GP` profiles as dataclasses (stem names, descriptions)
- [ ] `medleydb_instruments.yaml`: Complete lookup table — all labels from `instrument_f0_type.json` with target stem for each profile
- [ ] `load_medleydb_mapping(profile) → dict[str, str]`: Load YAML, return `{label: target_stem}`
- [ ] MoisesDB custom top-level mapping dicts (excluding percussion AND bass from top-level)
- [ ] MoisesDB `percussion_routing` and `bass_routing` dicts for sub-stem level handling
- [ ] Case-insensitive matching: normalize labels to lowercase before lookup
- [ ] Unknown label handling: map to `other`, log warning
- [ ] Tests: `tests/test_mapping.py` — every label for both profiles, unknown labels, case insensitivity
- [ ] Verify: `pytest tests/test_mapping.py -v`

### Phase 3: Overlap Registry
**Goal**: Hardcoded overlap list with resolution logic.
**Files**: `src/mss_datasets/overlap.py`
- [ ] `MUSDB_MEDLEYDB_OVERLAP`: Hardcoded set of 46 MUSDB18 track names (format: "Artist - Title") — see §3.4 for full list
- [ ] `get_overlap_set() → set[str]`
- [ ] `is_overlap_track(musdb_track_name: str) → bool`
- [ ] `resolve_overlaps(musdb_tracks, medleydb_tracks, medleydb_present: bool) → dict`: Returns which tracks to skip from MUSDB18-HQ and which MedleyDB tracks inherit MUSDB18-HQ splits
- [ ] Tests: `tests/test_overlap.py`
- [ ] Verify: `pytest tests/test_overlap.py -v`

### Phase 4: Dataset Adapter — MUSDB18-HQ
**Goal**: Read MUSDB18-HQ WAVs directly (no `musdb` package).
**Files**: `src/mss_datasets/datasets/base.py`, `src/mss_datasets/datasets/musdb18hq.py`
- [ ] `DatasetAdapter` abstract base: `validate_path()`, `discover_tracks()`, `process_track()`
- [ ] `TrackInfo` dataclass: source_dataset, artist, title, split, stems_available, path, has_bleed, etc.
- [ ] `Musdb18hqAdapter.__init__(path)`: Store path, validate `train/` and `test/` exist
- [ ] `discover_tracks()`: Walk subdirs, parse folder name as "Artist - Title", determine split from parent dir
- [ ] `process_track(track_info, profile, output_dir)`: Read stem WAVs, ensure float32/stereo, write to output via `write_wav_atomic()`
- [ ] Skip tracks in overlap set when MedleyDB is also present (receives skip list from pipeline)
- [ ] Test with synthetic fixture: `tests/fixtures/musdb18hq/train/TestArtist - TestSong/{vocals,drums,bass,other}.wav`
- [ ] Verify: `pytest tests/test_musdb18hq.py -v`

### Phase 5: Dataset Adapter — MedleyDB
**Goal**: Parse YAML metadata + read stem WAVs directly (no `medleydb` package).
**Files**: `src/mss_datasets/datasets/medleydb.py`
- [ ] `MedleydbAdapter.__init__(path)`: Store path, validate `Audio/` subdir exists
- [ ] `discover_tracks()`: Walk `Audio/` subdirs, parse `*_METADATA.yaml` for each track
- [ ] YAML parsing: extract `stems` dict with per-stem `instrument` label and `has_bleed` flag
- [ ] `process_track()`: For each stem in YAML, look up instrument → target category (case-insensitive), load WAV, sum stems per category, write output
- [ ] Handle `Main System` stems: skip the stem, process others normally
- [ ] Handle `Unlabeled` stems: route to `other`, flag in metadata
- [ ] For overlap tracks: mark as `source_dataset: "medleydb"` with inherited MUSDB18-HQ split
- [ ] Stem file path: `{track_dir}/{track_name}_STEMS/{track_name}_STEM_{NN}.wav`
- [ ] Test with synthetic fixture
- [ ] Verify: `pytest tests/test_medleydb.py -v`

### Phase 6: Dataset Adapter — MoisesDB
**Goal**: Use `moisesdb` library with custom sub-stem routing for percussion AND bass.
**Files**: `src/mss_datasets/datasets/moisesdb_adapter.py`
- [ ] `MoisesdbAdapter.__init__(path, sample_rate=44100)`: Create `MoisesDB(data_path=path, sample_rate=44100)` instance
- [ ] `discover_tracks()`: Iterate `db`, extract `track.id`, `track.artist`, `track.name`, `track.genre`
- [ ] `process_track()`: For each track:
  1. Call `track.mix_stems(custom_mapping)` for top-level stems (EXCLUDING `percussion` and `bass`)
  2. Handle `percussion` via `track.stem_sources_mixture("percussion")`: route a-tonal→drums, pitched→other
  3. Handle `bass` via `track.stem_sources_mixture("bass")`: route bass guitar/synth/contrabass→bass, tuba/bassoon→other
  4. Sum sub-stem routing results into appropriate output arrays
  5. Write only stems that have non-silent audio
- [ ] Record bleed info from `track.bleedings`
- [ ] Test with mocks (no real dataset in CI)
- [ ] Verify: `pytest tests/test_moisesdb.py -v`

### Phase 7: Split Management
**Goal**: Deterministic, lockable split assignments.
**Files**: `src/mss_datasets/splits.py`
- [ ] MUSDB18-HQ splits: infer from directory (train/ → "train", test/ → "test")
- [ ] MoisesDB splits: deterministic 50-track val set (fixed seed=42, genre-stratified). Hardcode the track IDs after initial generation.
- [ ] MedleyDB unique tracks: all "train"
- [ ] MedleyDB overlap tracks: inherit MUSDB18-HQ split assignment via overlap registry
- [ ] `write_splits(path, assignments)`: Write `splits.json`
- [ ] `load_splits(path) → dict`: Load existing `splits.json` if present (lock mechanism)
- [ ] `assign_splits(tracks, existing_splits=None)`: Assign splits, respecting locked assignments
- [ ] Verify no artist leakage across train/test boundaries
- [ ] Verify: `pytest tests/test_splits.py -v`

### Phase 8: Metadata & Error Logging
**Goal**: All metadata file generation.
**Files**: `src/mss_datasets/metadata.py`
- [ ] `ManifestEntry` dataclass matching §6.4 schema
- [ ] `write_manifest(path, entries: list[ManifestEntry])`
- [ ] `ErrorEntry` dataclass, `write_errors(path, errors: list[ErrorEntry])`
- [ ] `write_overlap_registry(path, skipped_musdb_tracks: list, reason: str)`
- [ ] `write_config(path, effective_config: dict)`
- [ ] Verify: `pytest tests/test_metadata.py -v`

### Phase 9: Pipeline Orchestration
**Goal**: End-to-end pipeline tying all components together.
**Files**: `src/mss_datasets/pipeline.py`
- [ ] `Pipeline.__init__(config)`: Accept all CLI args as config
- [ ] Stage 1 (Acquire): Validate paths per §7.1 expected directory structures, instantiate adapters for each provided dataset
- [ ] Stage 2 (Deduplicate): Run overlap registry, compute skip lists, pass to adapters
- [ ] Stage 3 (Process): For each adapter, discover tracks → assign splits → process each track (stem map + normalize + write)
- [ ] Resumability: before processing each track, check if all expected output files exist with correct size. Skip if complete. Clean up `.tmp` files on startup.
- [ ] Stage 4 (Validate): Post-write checks — valid WAV, correct sr/channels/dtype, flag silent stems
- [ ] Stage 5 (Metadata): Write manifest.json, splits.json, overlap_registry.json, errors.json, config.yaml
- [ ] Disk space check: estimate output size from track count/duration, warn if insufficient
- [ ] Summary report: print stem counts, disk usage, error count
- [ ] Verify: `pytest tests/test_pipeline.py -v` (end-to-end with fixtures)

### Phase 10: Parallelism
**Goal**: Multi-worker processing.
**Files**: Modify `src/mss_datasets/pipeline.py`
- [ ] `--workers N` support via `concurrent.futures.ProcessPoolExecutor`
- [ ] Per-track processing as the parallelism unit (adapter.process_track is the work function)
- [ ] Thread-safe error log accumulation (use a `multiprocessing.Manager().list()` or collect results)
- [ ] Progress bar integration: `tqdm` with `position` for multi-process, or shared counter
- [ ] Verify: `pytest tests/test_parallel.py -v` — same output with workers=1 vs workers=4

### Phase 11: CLI & Config
**Goal**: Full CLI matching §11 flags.
**Files**: `src/mss_datasets/cli.py`
- [ ] Click command group with all flags from §11 (including `--group-by-dataset`)
- [ ] `--config` YAML loading, CLI flags override config values
- [ ] `--dry-run`: enumerate tracks, show per-stem counts, estimate disk usage, exit
- [ ] `--validate`: check existing output directory (all WAVs valid, metadata consistent)
- [ ] `--verbose`: control log level
- [ ] Progress bar: `try: import rich` with `except: import tqdm` fallback
- [ ] Verify: `pytest tests/test_cli.py -v && mss-datasets --help`

### Phase 12: Integration Testing & Documentation
**Goal**: Full integration validation and user-facing docs.
- [ ] Integration test: run full pipeline against fixture datasets, verify all output files + metadata
- [ ] README.md: installation, quick start (3-line example), CLI reference, output format, FAQ
- [ ] `pip install .` in fresh venv verification
- [ ] `mss-datasets --help` output matches §11

### Phase Dependency Graph
```
P0 → P1 → P2 → P3 ─┐
                     ├→ P4 (MUSDB18-HQ)
                     ├→ P5 (MedleyDB)
                     └→ P6 (MoisesDB)    [P4-P6 can run in parallel]
                          └─── P7 → P8 → P9 → P10 → P11 → P12
```

---

## 18. References

1. [MUSDB18 — SigSep](https://sigsep.github.io/datasets/musdb.html)
2. [MUSDB18-HQ — Zenodo](https://zenodo.org/records/3338373)
3. [MoisesDB Paper (arXiv:2307.15913)](https://ar5iv.labs.arxiv.org/html/2307.15913)
4. [MoisesDB GitHub (moises-ai/moises-db)](https://github.com/moises-ai/moises-db)
5. [MedleyDB Description](https://medleydb.weebly.com/description.html)
6. [MedleyDB taxonomy.yaml](https://github.com/marl/medleydb/blob/master/medleydb/resources/taxonomy.yaml)
7. [MedleyDB instrument_f0_type.json](https://github.com/marl/medleydb/blob/master/medleydb/resources/instrument_f0_type.json)
8. [ZFTurbo Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training)
9. [ZFTurbo Dataset Types](https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/main/docs/dataset_types.md)
10. [ACMID Paper](https://arxiv.org/html/2510.07840)
11. [MedleyDB Python Package](https://medleydb.readthedocs.io/en/latest/api.html)
12. [Open-Source Tools & Data for MSS](https://source-separation.github.io/tutorial/data/musdb18.html)
13. [Demucs / python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator)
