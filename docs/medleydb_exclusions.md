# MedleyDB Exclusions

MedleyDB uses ~121 instrument labels that are mapped to output stems (vocals, drums, bass, other, etc.). Some labels are ambiguous — they describe the tool or effect rather than the audio content. After manual review of every stem with these labels, the following exclusions were applied.

All exclusion rules are defined in [`src/mss_datasets/mapping/medleydb_overrides.yaml`](../src/mss_datasets/mapping/medleydb_overrides.yaml) and applied automatically during processing.

## Excluded Tracks (5)

These tracks have excessive stem bleed — audio from other instruments bleeds into unrelated stems, making them unreliable for source separation training.

| Track | Instrument Reviewed | Issue |
|---|---|---|
| HopsNVinyl_ChickenFriedSteak | vibraphone (S05) | Full mix audible in stem |
| HopsNVinyl_HoneyBrown | vibraphone (S01) | Full mix audible in stem |
| HopsNVinyl_ReignCheck | vibraphone (S05) | Full mix audible in stem |
| HopsNVinyl_WidowsWalk | vibraphone (S05) | Full mix audible in stem |
| Karachacha_Volamos | vibraphone (S03) | Drums bleeding into stem |

## Excluded Stems (22 stems across 15 tracks)

These individual stems contain audio that doesn't match the category they would be routed to. They are skipped during processing while the rest of each track is used normally.

### Sampler (3 stems)

The `sampler` label maps to "other", but the actual audio content is vocals or drums.

| Track | Stem | Content | Would Route To |
|---|---|---|---|
| Grants_PunchDrunk | S01 | Vocal samples | other |
| Grants_PunchDrunk | S08 | Vocal samples | other |
| Snowmine_Curfews | S06 | Sampled drums | other |

### FX / Processed Sound (19 stems)

The `fx/processed sound` label maps to "other", but many stems contain clearly identifiable vocals, drums, bass, or guitar.

| Track | Stem | Content | Would Route To |
|---|---|---|---|
| AClassicEducation_NightOwl | S12 | Voice | other |
| AimeeNorwich_Flying | S02 | Percussive | other |
| AimeeNorwich_Flying | S07 | Bass sounds | other |
| AimeeNorwich_Flying | S12 | Percussive | other |
| Creepoid_OldTree | S03 | Distorted vocals | other |
| DreamersOfTheGhetto_HeavyLove | S03 | Reversed percussion | other |
| EthanHein_1930sSynthAndUprightBass | S02 | Percussive | other |
| Grants_PunchDrunk | S10 | Percussive | other |
| Lushlife_ToynbeeSuite | S15 | Vocal sounds | other |
| MatthewEntwistle_ImpressionsOfSaturn | S02 | Breath sounds | other |
| MatthewEntwistle_TheArch | S08 | Percussion | other |
| MatthewEntwistle_TheArch | S17 | Percussive | other |
| PortStWillow_StayEven | S11 | Vocals | other |
| SecretMountains_HighHorse | S06 | Synth & guitar | other |
| TablaBreakbeatScience_MiloVsMongo | S07 | Phaser guitar | other |
| TablaBreakbeatScience_MoodyPlucks | S02 | Percussive | other |
| TablaBreakbeatScience_MoodyPlucks | S03 | Percussion | other |
| TablaBreakbeatScience_MoodyPlucks | S08 | Other + guitar | other |
| TablaBreakbeatScience_Scorpio | S06 | Percussive | other |

## Labels Not Excluded

The following stems with ambiguous labels were reviewed and found acceptable:

- **Pitched percussion** (vibraphone, glockenspiel, chimes): 13 stems across 13 tracks were reviewed and confirmed to contain only pitched percussion, which correctly routes to "other".
- **FX / Processed Sound**: 24 stems were reviewed and confirmed to contain electronic/synth/ambient content appropriate for "other" (e.g., reversed sounds, static, foley, synth pads).
