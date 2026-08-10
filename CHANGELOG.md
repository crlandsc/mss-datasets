# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
While the major version is 0, breaking changes bump the minor version.

## [0.2.1] - 2026-08-10

Licensing housekeeping. No code changes.

### Changed

- `LICENSE` now contains the MIT text alone. The dataset terms appended to it
  were enough to defeat GitHub's license detection, which left the repo showing
  no license at all. They move to `DATASET-LICENSES.md` unchanged in substance.

## [0.2.0] - 2026-08-09

A correctness release. An audit of the shipped 460-track dataset found it clean,
but surfaced four latent defects that would have corrupted a future reprocess.
All four are closed here, and the guarantees are now checked by command rather
than by hand.

### Breaking

- **Output filenames are now `{source}_{artist}_{title}.wav`.** The split and the
  positional index have both been removed. They previously fed the filename, so
  reassigning splits or changing dataset contents renamed files and stranded the
  old copies — with `--split-output` that left the same audio in both `train/`
  and `val/`. The split is now carried by the directory alone.
- `sanitize_filename(source, artist, title)` no longer takes `split` or `index`.
- `splits.json` and `manifest.json` are keyed on the new filename base rather
  than on a positional index.
- Reading an existing tree still works — both the current and legacy filename
  layouts are parsed. **Reprocess into a fresh output directory**, though: mixing
  the two leaves stale copies behind, and `--audit` reports it as an error.

### Added

- `--audit` checks an output tree for the invariants aggregation is supposed to
  guarantee: no duplicated tracks, no track in more than one split, no song
  arriving from two datasets, no mixing of filename layouts, and metadata that
  agrees with what is on disk. Exits non-zero on any error, so it works as a CI
  gate.
- `--inventory` reports the dataset regrouped for review, with MUSDB18-HQ and
  MoisesDB shown as their complete rosters and MedleyDB as the extras it
  uniquely contributes. Writes `metadata/inventory.md`.
- `--sorted-view PATH` materializes that regrouping as a browsable tree of
  relative symlinks, so no audio is duplicated. `--link-mode` selects
  `symlink`, `hardlink` or `copy`.
- Cross-dataset deduplication now compares all three datasets rather than only
  MUSDB18-HQ against MedleyDB, so a duplicate involving MoisesDB can no longer
  pass through unnoticed. `overlap_registry.json` records every resolved
  duplicate with the dataset that won it.
- The overlap list is verified against vendored copies of the upstream sigsep
  and `marl/medleydb` tracklists, in both directions, so a missing or spurious
  entry fails CI. Previously only its length was asserted.

### Fixed

- **Wheel packaging.** The MedleyDB instrument and override tables are loaded
  relative to the package directory but were never included in the built wheel,
  so any `pip`-installed copy raised `FileNotFoundError` on the first MedleyDB
  track. Present since those tables were introduced; invisible locally because
  editable installs read from the source tree.
- **Resumability.** A track counts as processed only when every file the previous
  run recorded still exists. The old check passed if *any single* stem was
  present, so a run interrupted between stems left that track permanently
  incomplete.
- **Manifest truncation.** `manifest.json` describes the whole dataset again. It
  previously recorded only the tracks processed in the most recent invocation,
  so any resumed run silently shrank it.
- **Stale copies across splits.** A reconcile pass relocates any track sitting in
  a split directory it no longer belongs to, before processing decides what
  needs writing.
- **MoisesDB validation set stability.** Selection now sorts by track name before
  shuffling, so it depends only on which tracks exist rather than on the order
  the `moisesdb` library happens to yield them. It also respects splits already
  locked by `splits.json`, which it previously overwrote unconditionally.
- MedleyDB stems carrying several instrument labels no longer raise
  `AttributeError`. They resolve to a shared target when the labels agree, and
  to `other` when they disagree rather than being attributed to one instrument.
- Added the missing `woodwind section` instrument mapping. No MedleyDB label is
  now unmapped.
- Filename collisions are resolved deterministically instead of silently
  overwriting. The resolver existed but was never called.
- `config*.yaml` and the default download directory are gitignored. Local
  configs hold a real Zenodo token and the download directory holds tens of
  gigabytes; neither was ignored.

### Removed

- The `huggingface-hub` dependency, which nothing imported and nothing else
  depended on.

## [0.1.2] - 2026-02-21

- README rewritten with fuller context for the aggregation decisions.

## [0.1.1] - 2026-02-21

- Corrected the name in the license file.

## [0.1.0] - 2026-02-21

Initial release. Aggregates MUSDB18-HQ, MoisesDB and MedleyDB into unified stem
folders under the `vdbo` and `vdbo+gp` profiles, with cross-dataset
deduplication, MedleyDB bleed and override filtering, deterministic splits,
optional mixtures and stem-sum verification.

[0.2.1]: https://github.com/crlandsc/mss-datasets/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/crlandsc/mss-datasets/compare/v0.1.2...v0.2.0
[0.1.2]: https://github.com/crlandsc/mss-datasets/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/crlandsc/mss-datasets/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/crlandsc/mss-datasets/releases/tag/v0.1.0
