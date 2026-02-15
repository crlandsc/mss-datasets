# MSS Data Aggregator — Specification Draft v0.1

## 1. Prior Art Analysis: Nothing Like This Exists

After thorough searching, I can confirm there is **no existing tool** that aggregates multiple MSS datasets into a unified, stem-mapped, normalized output. What does exist are individual dataset parsers and training frameworks:

The `musdb` Python package parses and processes the MUSDB18 dataset only. It was originally developed for the Music Separation task as part of the Signal Separation Evaluation Campaign (SISEC).[[9]](https://github.com/sigsep/sigsep-mus-db) MoisesDB provides its own separate Python library to download, process and use MoisesDB.[[2]](https://www.researchgate.net/publication/372784496_Moisesdb_A_dataset_for_source_separation_beyond_4-stems) MedleyDB has its own `medleydb` Python package with its own API for loading multitracks.

ZFTurbo's Music-Source-Separation-Training repository is the closest adjacent tool. It's a training framework that is easy to modify for experiments.[[2]](https://github.com/ZFTurbo/Music-Source-Separation-Training) It defines two dataset layout types: Type 1 (MUSDB-style), where each folder contains all needed stems as WAV files, and Type 2 (Stems-style), where each folder is a stem name containing wav files of only that stem.[[1]](https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/main/docs/dataset_types.md) Crucially, ZFTurbo's tool assumes you've **already organized your data** into these formats — it does not download, normalize, deduplicate, or map stems from multiple source datasets. Researchers are expected to do all of that manually.

Other tools like Spleeter, Demucs, Open-Unmix, and ByteSep are inference/training tools that operate on pre-existing data but do not aggregate or prepare training datasets.

**Conclusion: This tool would be genuinely novel.** No one has built a unified downloader + normalizer + stem-mapper + deduplicator for MSS datasets. Every lab currently writes bespoke scripts for this.

---

## 2. Source Datasets

### 2.1 MUSDB18-HQ (Primary — Benchmark)

MUSDB18 is a dataset of 150 full length music tracks (~10h total duration) of varying genres.[[3]](https://source-separation.github.io/tutorial/data/musdb18.html) All files from the MUSDB18-HQ dataset are saved as uncompressed wav files. All signals are stereophonic and encoded at 44.1kHz.[[9]](https://zenodo.org/records/3338373) MUSDB18 contains two folders, a folder with a training set: "train", composed of 100 songs, and a folder with a test set: "test", composed of 50 songs.[[5]](https://dagshub.com/kinkusuma/musdb18-dataset)

Native stems: `vocals`, `drums`, `bass`, `other` — already in VDBO format, zero mapping work needed.

The dataset is hosted on Zenodo and requires that users request access, since the tracks can only be used for academic purposes.[[9]](https://github.com/sigsep/sigsep-mus-db) This means our tool cannot fully automate the download — the user will need to authenticate/provide a path to the already-downloaded archive. The tool should detect and ingest it.

### 2.2 MoisesDB (Primary — Volume + Extended Stems)

MoisesDB is introduced as a dataset for musical source separation consisting of 240 tracks from 45 artists, covering twelve musical genres.[[1]](https://ar5iv.labs.arxiv.org/html/2307.15913) Total duration is approximately 14 hours, 24 minutes, and 46 seconds.[[5]](https://www.emergentmind.com/topics/moisesdb-dataset)

There are 11 top-level stems, each further subdivided into specific sub-stems. This structure mirrors the workflow of practical mixing sessions and enables both granular and aggregate source separation experiments.[[5]](https://www.emergentmind.com/topics/moisesdb-dataset) "Vocals," "drums," and "bass" stems are present in nearly all songs. In contrast, stems such as "wind" and "other plucked" are comparatively rare.[[5]](https://www.emergentmind.com/topics/moisesdb-dataset)

The 11 top-level stems are: **vocals, drums, bass, guitar, piano, bowed_strings, wind, other_plucked, other_keys, percussion, other**. Guitar subdivides further into acoustic guitar and electric guitar.

The MoisesDB paper benchmarked stems in groups: vocals, bass, drums, other, guitar, and piano.[[1]](https://ar5iv.labs.arxiv.org/html/2307.15913) This gives us the natural "extended" profile: **VDBO+GP (6-stem)**.

This dataset is offered free of charge for non-commercial research use only.[[1]](https://ar5iv.labs.arxiv.org/html/2307.15913)

**No overlap with MUSDB18 or MedleyDB.** MoisesDB consists of entirely new recordings from distinct artists.

### 2.3 MedleyDB v1 + v2 (Primary — Diversity, with Caveats)

Bittner et al. released the MedleyDB dataset, which comprises 122 songs in multitrack format. It was extended by 74 songs (totalling 196 songs) in 2016, and published as MedleyDB 2.0.[[2]](https://www.researchgate.net/publication/372784496_Moisesdb_A_dataset_for_source_separation_beyond_4-stems) All types of audio files are .wav files with a sample rate of 44.1 kHz and a bit depth of 16. The mix and stems are stereo and the raw audio files are mono.[[4]](https://medleydb.weebly.com/description.html)

**Critical: MedleyDB's stem labeling is free-form, not VDBO.** The shortcoming of MedleyDB for music source separation is the way it organizes tracks into stems. While it provides instrument information for each of them, and functional annotations for stems, stems are not meaningfully labelled, only numbered. As a result, stem 01 of one song may be the drum kit, while stem 01 of another mix is the bassoon. Furthermore, instruments are grouped according to how they physically produce their sound, rather than their role in the mix.[[1]](https://ar5iv.labs.arxiv.org/html/2307.15913)

This means MedleyDB requires the most stem-mapping work. Each stem has instrument metadata in YAML files with labels like "female singer," "clean electric guitar," "drum set," etc. We'll need a comprehensive lookup table to map ~50+ instrument labels to VDBO (or extended) categories.

MedleyDB is offered free of charge for non-commercial research use only under the terms of the Creative Commons Attribution Noncommercial License.[[7]](https://medleydb.weebly.com/downloads.html)

### 2.4 CRITICAL: Cross-Dataset Overlap

This is a showstopper issue that many researchers don't account for:

The data from MUSDB18 is composed of several different sources: 100 tracks are taken from the DSD100 dataset. 46 tracks are taken from the MedleyDB licensed under Creative Commons (BY-NC-SA 4.0). 2 tracks were kindly provided by Native Instruments. 2 tracks are from the Canadian rock band The Easton Ellises.[[4]](https://sigsep.github.io/datasets/musdb.html)

**46 of the 150 MUSDB18 tracks are pulled directly from MedleyDB.** If we naively include both MUSDB18-HQ and MedleyDB, those 46 songs will appear twice. Since MUSDB18-HQ is the benchmark and its versions of these tracks are already in VDBO format, MUSDB18-HQ takes priority for these 46 songs. MedleyDB contributes only the **remaining ~150 tracks** (196 total minus the 46 that are already in MUSDB18).

MoisesDB has zero overlap with either — 46 songs from MedleyDB are also used in MUSDB18,[[1]](https://ar5iv.labs.arxiv.org/html/2307.15913) but MoisesDB is entirely independent.

The tool must maintain a hardcoded overlap registry that prevents duplicates.

### 2.5 Tier 2 Datasets — Recommendation

**Slakh2100**: Slakh consists of 145 hours of mixtures.[[2]](https://www.researchgate.net/publication/372784496_Moisesdb_A_dataset_for_source_separation_beyond_4-stems) However, Slakh, as a synthetic dataset with a different distribution, performed poorly[[5]](https://arxiv.org/html/2510.07840) when used for real-world MSS training in the ACMID paper's experiments. Given your goal of testing architectures on real audio, and given the added complexity of integrating a synthetic dataset, I recommend **excluding Slakh from v1** of this tool. It can be a follow-up.

**ACMID**: As we discussed previously, solo-instrument crawled data without mixtures. **Exclude from v1.**

---

## 3. Stem Profiles

### 3.1 Default: VDBO (4-stem)

| Output Stem | MUSDB18-HQ Source | MoisesDB Source | MedleyDB Source |
|---|---|---|---|
| `vocals` | `vocals` | `vocals` | stems labeled: `male singer`, `female singer`, `vocalist`, `choir`, `vocalists`, etc. |
| `drums` | `drums` | `drums` | stems labeled: `drum set`, `drum machine`, `timpani`, `toms`, etc. |
| `bass` | `bass` | `bass` | stems labeled: `electric bass`, `acoustic bass`, `bass synthesizer`, etc. |
| `other` | `other` | everything else (guitar, piano, bowed_strings, wind, other_plucked, other_keys, percussion, other) summed | everything else summed |

### 3.2 Extended: VDBGPO (6-stem)

This profile splits guitar and piano out of the "other" bucket, matching the MoisesDB benchmark configuration of vocals, bass, drums, other, guitar, and piano.[[1]](https://ar5iv.labs.arxiv.org/html/2307.15913) This is also exactly what the Demucs `htdemucs_6s` model outputs: vocals, drums, bass, guitar, piano, other.[[8]](https://github.com/nomadkaraoke/python-audio-separator)

| Output Stem | MUSDB18-HQ Source | MoisesDB Source | MedleyDB Source |
|---|---|---|---|
| `vocals` | `vocals` | `vocals` | vocal-labeled stems |
| `drums` | `drums` | `drums` | drum-labeled stems |
| `bass` | `bass` | `bass` | bass-labeled stems |
| `guitar` | *extracted from `other`* — see note | `guitar` (acoustic + electric) | stems labeled: `acoustic guitar`, `clean electric guitar`, `distorted electric guitar`, etc. |
| `piano` | *extracted from `other`* — see note | `piano` | stems labeled: `piano`, `electric piano`, `synthesizer`, etc. |
| `other` | *remainder of `other`* — see note | bowed_strings + wind + other_plucked + other_keys + percussion + other | everything else |

**Important MUSDB18-HQ limitation**: MUSDB18-HQ only provides 4 pre-mixed stems. The `other` stem is a single audio file containing guitar + piano + everything else already summed. **It is not possible to extract guitar or piano from MUSDB18-HQ's `other` stem** — the information is permanently mixed. When running in 6-stem mode, MUSDB18-HQ tracks can only contribute to the 4-stem subset (VDBO). The 6-stem profile is populated primarily from MoisesDB and MedleyDB, which have instrument-level stems. The spec should make this explicit and flag these tracks in metadata so training pipelines know which tracks can contribute all 6 stems vs. only 4.

### 3.3 Future Profiles (Out of Scope for v1, but Architecture Should Support)

Things like 7-stem (splitting guitar into acoustic/electric), 8-stem (adding percussion separate from drums), etc. The config-based approach should make adding these trivial, but they're not a v1 priority.

---

## 4. Output Format

Based on your workflow — where you mix stems on-the-fly rather than using pre-baked mixtures — the primary output format should be **ZFTurbo Type 2 (Stems-style)**: one folder per stem category, containing individual WAV files.

```
output/
├── vocals/
│   ├── musdb18hq_train_001_AClassicEducation_NightOwl.wav
│   ├── musdb18hq_train_002_ANightWithMeow_Pillow.wav
│   ├── moisesdb_001_ArtistName_TrackName.wav
│   ├── medleydb_001_LizNelson_Rainfall.wav
│   └── ...
├── drums/
│   ├── musdb18hq_train_001_AClassicEducation_NightOwl.wav
│   └── ...
├── bass/
│   └── ...
├── other/
│   └── ...
├── guitar/          # only present in 6-stem profile
│   └── ...
├── piano/           # only present in 6-stem profile
│   └── ...
└── metadata/
    ├── manifest.json       # per-file provenance, source dataset, license, original name
    ├── splits.json         # train/val/test assignments
    └── overlap_registry.json  # which tracks were deduplicated and from where
```

The naming convention for files encodes provenance: `{source_dataset}_{split}_{index}_{artist}_{title}.wav`. This makes it trivially easy to trace any file back to its origin without needing to look up metadata.

**Mixture files**: Not generated by default. Available via `--include-mixtures` flag, which creates a parallel `mixtures/` folder. When present, each mixture file is generated as the linear sum of all stems for that song (not downloaded from the source, since you want consistency with your on-the-fly mixing approach).

**Format**: All output files are 44.1 kHz, 16-bit, stereo WAV.

---

## 5. Processing Pipeline

### 5.1 Stage 1: Acquire

For each dataset, the tool either downloads from the source or ingests from a user-provided local path. MUSDB18-HQ requires Zenodo access, so the user must provide the path to their downloaded archive. MoisesDB may require an API key or download token. MedleyDB similarly requires a request. The tool detects which datasets are available locally and processes them.

```
mss-aggregate --musdb18hq-path /path/to/musdb18hq \
              --moisesdb-path /path/to/moisesdb \
              --medleydb-path /path/to/medleydb \
              --profile vdbo \
              --output ./data
```

For datasets that can be auto-downloaded, the tool handles it. For those requiring manual access, it provides clear instructions and validates the provided path.

### 5.2 Stage 2: Deduplicate

Before any processing, the tool runs the overlap registry:

The hardcoded overlap list identifies the **46 MedleyDB tracks that exist in MUSDB18-HQ**. When both datasets are present, the MUSDB18-HQ version takes priority (it's already in VDBO format and is the benchmark). The MedleyDB copies of those 46 tracks are skipped entirely.

For safety, the tool also runs an audio fingerprint check (chromaprint) across all remaining tracks to catch any unknown overlaps, logging any flagged pairs for user review.

### 5.3 Stage 3: Stem Map

This is the core logic. For each track in each dataset:

**MUSDB18-HQ**: No mapping needed for VDBO. Files are already named `vocals.wav`, `drums.wav`, `bass.wav`, `other.wav`.

**MoisesDB**: Use the MoisesDB Python library's built-in taxonomy. For VDBO: sum guitar + piano + bowed_strings + wind + other_plucked + other_keys + percussion + other into the `other` stem. For 6-stem: guitar and piano get their own outputs, everything else (bowed_strings, wind, other_plucked, other_keys, percussion, other) sums into `other`.

**MedleyDB**: Read each track's YAML metadata to get instrument labels per stem. Apply the instrument-to-VDBO lookup table. Multiple stems mapping to the same category get summed. This lookup table is a critical artifact that must be auditable (stored as a YAML config, not buried in code). Edge cases like "Main System" (full ensemble recordings) need special handling — these tracks may need to be flagged or excluded.

### 5.4 Stage 4: Normalize

For every output stem file: resample to 44.1 kHz (should be a no-op for most tracks since MUSDB18-HQ, MoisesDB, and MedleyDB are all natively 44.1 kHz), convert to 16-bit PCM, ensure stereo (MedleyDB raw tracks are mono — if we're using raw tracks, pad to dual-mono; if using pre-mixed stems, they're already stereo), and write as WAV.

Optional loudness normalization (EBU R128 to a target LUFS) is **off by default** since your pipeline handles its own mixing dynamics. Available via `--normalize-loudness` flag.

### 5.5 Stage 5: Validate

Post-processing checks for every output file: verify it's valid WAV with correct sample rate, bit depth, and channel count. Check for silent files (any stem that's all zeros or below a noise floor threshold gets flagged in metadata — it's kept in the output since some songs legitimately have no bass, for example, but it's labeled so the training pipeline can handle it). Verify that the total number of output files per stem matches expectations.

For tracks where we have the original mixture available (MUSDB18-HQ), optionally verify that the sum of VDBO stems matches the mixture within floating-point tolerance. This is a sanity check on our mapping, not a required feature for the end user.

### 5.6 Stage 6: Write Metadata

Generate `manifest.json` with per-file records including: source dataset, original track name, artist, split (train/val/test), stem profile used, license identifier, whether the file is a summed composite of multiple original stems or a direct copy, and any flags (e.g., silent, below loudness threshold, bleed detected).

Generate `splits.json` locking the train/val/test assignments.

---

## 6. Split Management

MUSDB18-HQ's splits are canonical and inviolable. MUSDB18 contains a training set of 100 songs and a test set of 50 songs. Supervised approaches should be trained on the training set and tested on both sets.[[3]](https://zenodo.org/records/1117372) The 50-song test set must never appear in training data.

For MoisesDB, we adopt the split used in recent literature: the MoisesDB validation set was constructed by randomly selecting 50 tracks from the MoisesDB dataset, maintaining the same genre distribution as the evaluation split of MUSDB18-HQ.[[4]](https://www.researchgate.net/figure/Distribution-of-stems-in-MoisesDB-from-moisesdbdataset-import-MoisesDB-db_fig1_372784496) We should use this established split (or the 8:1:1 split used in the ACMID paper) rather than inventing our own.

For MedleyDB, the `medleydb` Python library provides an `artist_conditional_split` function to create artist-conditional train-test splits[[3]](https://medleydb.readthedocs.io/_/downloads/en/latest/pdf/) — this ensures the same artist doesn't appear in both train and test, which is important for preventing data leakage. We should use this or adopt the split used in prior work.

All split assignments are locked in `splits.json` on first run and never changed. If a user re-runs the tool, they get identical splits.

---

## 7. CLI Interface

```
# Default: VDBO profile, all available datasets
mss-aggregate \
  --musdb18hq-path /path/to/musdb18hq \
  --moisesdb-path /path/to/moisesdb \
  --medleydb-path /path/to/medleydb \
  --output ./data

# 6-stem extended profile
mss-aggregate \
  --musdb18hq-path /path/to/musdb18hq \
  --moisesdb-path /path/to/moisesdb \
  --medleydb-path /path/to/medleydb \
  --profile vdbo+gp \
  --output ./data

# Only specific datasets
mss-aggregate \
  --musdb18hq-path /path/to/musdb18hq \
  --moisesdb-path /path/to/moisesdb \
  --output ./data

# Include mixture files
mss-aggregate \
  --musdb18hq-path /path/to/musdb18hq \
  --profile vdbo \
  --include-mixtures \
  --output ./data

# Dry run — show what would be processed
mss-aggregate \
  --musdb18hq-path /path/to/musdb18hq \
  --moisesdb-path /path/to/moisesdb \
  --medleydb-path /path/to/medleydb \
  --dry-run

# Validate existing output
mss-aggregate --validate ./data

# Normalize loudness (optional)
mss-aggregate --musdb18hq-path ... --normalize-loudness --loudness-target -14
```

The `--profile` flag defaults to `vdbo`. Options are `vdbo` (4-stem) and `vdbo+gp` (6-stem). Future profiles can be added.

---

## 8. Key Numbers (Expected Output)

For a VDBO run with all three Tier 1 datasets:

| Dataset | Tracks | Est. Duration | Notes |
|---|---|---|---|
| MUSDB18-HQ | 150 | ~10h | Native VDBO, 100 train / 50 test |
| MoisesDB | 240 | ~14.4h | Mapped to VDBO, ~190 train / ~50 val |
| MedleyDB (unique) | ~150 | ~7h | 196 total minus ~46 overlap with MUSDB18 |
| **Total** | **~540** | **~31h** | Deduplicated, unique tracks |

Output: 4 stem folders × ~540 files each = ~2,160 WAV files, plus metadata.

For VDBO+GP (6-stem), MoisesDB and MedleyDB contribute to all 6 stems, while MUSDB18-HQ tracks only contribute to 4 (with guitar/piano metadata flags). Approximately ~390 tracks would have full 6-stem coverage.

---

## 9. Open Questions for Discussion

**Q1**: For the MedleyDB instrument-to-VDBO mapping, there are some genuinely ambiguous cases. For example, should "synthesizer" map to `piano`, `other`, or a new category? Should "drum machine" map to `drums` or `other`? I think we should draft the lookup table as a separate deliverable and have it reviewed.

**Q2**: MedleyDB has a `has_bleed` flag on some tracks — meaning microphone bleed between stems. Should we exclude these tracks, include them with a warning flag, or include them silently? Bleed is common in real recordings and arguably makes training data more realistic, but it technically violates the assumption of clean stems.

**Q3**: For the 6-stem profile, how should we handle MoisesDB tracks that don't have a guitar or piano stem? (Not all 240 tracks have every instrument.) Should those tracks contribute `silent` stems for guitar/piano, or should they be excluded from the 6-stem output?

**Q4**: Should the output filenames be anonymized/hashed, or preserve artist/track name for human readability? I lean toward readable names with provenance encoded, as shown above.

---

That's the spec as it stands. What do you want to dig into or revise first?

---
Learn more:
1. [Datasets — Open-Source Tools & Data for Music Source Separation](https://source-separation.github.io/tutorial/data/datasets.html)
2. [Introduction — Open-Source Tools & Data for Music Source Separation](https://source-separation.github.io/tutorial/data/introduction.html)
3. [\[2307.15913\] MoisesDB: A Dataset for Source Separation beyond 4-Stems](https://ar5iv.labs.arxiv.org/html/2307.15913)
4. [medleydb/medleydb/multitrack.py at master · marl/medleydb](https://github.com/marl/medleydb/blob/master/medleydb/multitrack.py)
5. [Music-Source-Separation-Training/docs/dataset\_types.md at main · ZFTurbo/Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/main/docs/dataset_types.md)
6. [(PDF) Moisesdb: A dataset for source separation beyond 4-stems](https://www.researchgate.net/publication/372784496_Moisesdb_A_dataset_for_source_separation_beyond_4-stems)
7. [GitHub - deezer/spleeter: Deezer source separation library including pretrained models.](https://github.com/deezer/spleeter)
8. [GitHub - ZFTurbo/Music-Source-Separation-Training: Repository for training models for music source separation.](https://github.com/ZFTurbo/Music-Source-Separation-Training)
9. [The MUSDB18 dataset — Open-Source Tools & Data for Music Source Separation](https://source-separation.github.io/tutorial/data/old.musdb18.html)
10. [Multitrack — medleydb 1.3.4 documentation](https://medleydb.readthedocs.io/en/latest/api.html)
11. [The MUSDB18 dataset — Open-Source Tools & Data for Music Source Separation](https://source-separation.github.io/tutorial/data/musdb18.html)
12. [\[2307.15913\] Moisesdb: A dataset for source separation beyond 4-stems](https://arxiv.org/abs/2307.15913)
13. [MUSDB18 - a corpus for music separation](https://zenodo.org/records/1117372)
14. [medleydb Documentation Release 1.3.4 Rachel Bittner Oct 11, 2018](https://medleydb.readthedocs.io/_/downloads/en/latest/pdf/)
15. [Music-Source-Separation-Training/docs/pretrained\_models.md at main · ZFTurbo/Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/main/docs/pretrained_models.md)
16. [Solos: A Dataset for Audio-Visual Music Source Separation and Localization - Solos](https://juanmontesinos.com/Solos/)
17. [MUSDB18 | SigSep](https://sigsep.github.io/datasets/musdb.html)
18. [MUSDB18 - a corpus for music separation](https://inria.hal.science/hal-02190845v1/document)
19. [GitHub - sigsep/open-unmix-pytorch: Open-Unmix - Music Source Separation for PyTorch](https://github.com/sigsep/open-unmix-pytorch)
20. [Distribution of stems in MoisesDB. from moisesdb.dataset import... | Download Scientific Diagram](https://www.researchgate.net/figure/Distribution-of-stems-in-MoisesDB-from-moisesdbdataset-import-MoisesDB-db_fig1_372784496)
21. [Description - MedleyDB](https://medleydb.weebly.com/description.html)
22. [Music-Source-Separation-Training/README.md at main · ZFTurbo/Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/main/README.md)
23. [Cutting Music Source Separation Some Slakh: A Dataset to Study the Impact of Training Data Quality and Quantity | IEEE Conference Publication | IEEE Xplore](https://ieeexplore.ieee.org/document/8937170/)
24. [kinkusuma/musdb18-dataset | DagsHub](https://dagshub.com/kinkusuma/musdb18-dataset)
25. [GitHub - Frikallo/MISST: A local GUI music source separation tool built on Tkinter and demucs serving as a free and open source Stem Player](https://github.com/Frikallo/MISST)
26. [ACMID: Automatic Curation of Musical Instrument Dataset for 7-stem music source separation](https://arxiv.org/html/2510.07840)
27. [MoisesDB Multitrack Music Dataset](https://www.emergentmind.com/topics/moisesdb-dataset)
28. [MedleyDB - Home](https://medleydb.weebly.com/)
29. [Release SCNet XL · ZFTurbo/Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/tag/v1.0.13)
30. [(PDF) MedleyDB: A Multitrack Dataset for Annotation-Intensive MIR Research](https://www.researchgate.net/publication/265508421_MedleyDB_A_Multitrack_Dataset_for_Annotation-Intensive_MIR_Research)
31. [medleydb.multitrack — medleydb 1.3.4 documentation](https://medleydb.readthedocs.io/en/latest/_modules/medleydb/multitrack.html)
32. [ISMIR 2023: MoisesDB: A Dataset for Source Separation Beyond 4 Stems](https://ismir2023program.ismir.net/poster_160.html)
33. [Post your model · Issue #1 · ZFTurbo/Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training/issues/1)
34. [musdb · PyPI](https://pypi.org/project/musdb/)
35. [website/content/datasets/musdb.md at master · sigsep/website](https://github.com/sigsep/website/blob/master/content/datasets/musdb.md)
36. [AIforMusic Toolbox — Resource Library — Spleeter: Deezer’s Open‑Source AI for Music Stem Separation](https://tools.aiformusic.org/knowledgebase/articles/spleeter-deezer-s-open-source-ai-for-music-stem-separation)
37. [ACMID: AUTOMATIC CURATION OF MUSICAL INSTRUMENT DATASET FOR 7-STEM](https://arxiv.org/pdf/2510.07840)
38. [Downloads - MedleyDB](https://medleydb.weebly.com/downloads.html)
39. [Introducing MoisesDB | Music AI](https://music.ai/blog/press/introducing-moisesdb-the-ultimate-multitrack-dataset-for-source-separation-beyond-4-stems/)
40. [Releases · ZFTurbo/Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training/releases)
41. [Ismir](https://archives.ismir.net/ismir2023/paper/000073.pdf)
42. [Music Source Separation with Hybrid Demucs — Torchaudio 0.13.0 documentation](https://docs.pytorch.org/audio/0.13.0/tutorials/hybrid_demucs_tutorial.html)
43. [GitHub - nomadkaraoke/python-audio-separator: Easy to use stem (e.g. instrumental/vocals) separation from CLI or as a python package, using a variety of amazing pre-trained models (primarily from UVR)](https://github.com/nomadkaraoke/python-audio-separator)
44. [Music-Source-Separation-Training/docs/mel\_roformer\_experiments.md at main · ZFTurbo/Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/main/docs/mel_roformer_experiments.md)
45. [GitHub - bytedance/music\_source\_separation](https://github.com/bytedance/music_source_separation)
46. [SigSep](https://sigsep.github.io/)
47. [GitHub - sigsep/sigsep-mus-db: Python parser and tools for MUSDB18 Music Separation Dataset](https://github.com/sigsep/sigsep-mus-db)
48. [MUSDB18-HQ - an uncompressed version of MUSDB18](https://zenodo.org/records/3338373)
49. [The SJTU X-LANCE Lab System for MSR Challenge 2025](https://arxiv.org/html/2602.09042)
50. [MedleyDB: A MULTITRACK DATASET FOR ANNOTATION-INTENSIVE MIR RESEARCH](https://www.justinsalamon.com/uploads/4/3/9/4/4394963/bittner_medleydb_ismir2014.pdf)
51. [Music-Source-Separation-Training/docs/gui.md at main · ZFTurbo/Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/main/docs/gui.md)
52. [Slakh | Demo site for the Synthesized Lakh Dataset (Slakh)](http://www.slakh.com/)
53. [Music Source Separation download | SourceForge.net](https://sourceforge.net/projects/music-source-separation.mirror/)
54. [Instrument taxonomy and classification](https://groups.google.com/g/music-ontology-specification-group/c/9dCGZGLEgVs)
55. [Music Source Separation with Hybrid Demucs — Torchaudio 2.8.0 documentation](https://docs.pytorch.org/audio/main/tutorials/hybrid_demucs_tutorial.html)
56. [medleydb/docs/example.rst at master · marl/medleydb](https://github.com/marl/medleydb/blob/master/docs/example.rst)
57. [PyMusic-Instrument · PyPI](https://pypi.org/project/PyMusic-Instrument/)
58. [Roformer-based Models | ZFTurbo/Music-Source-Separation-Training | DeepWiki](https://deepwiki.com/ZFTurbo/Music-Source-Separation-Training/4.1-roformer-based-models)