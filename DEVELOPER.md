# KeyBand Developer Notes

This file is for developers and internal maintenance.

## Audio Pack Build

Requirements:

- `python3`
- `ffmpeg`

Build the default piano pack:

```bash
python3 scripts/build_piano_samples.py --clean
```

Outputs:

- `docs/assets/audio/piano/*.m4a`
- `docs/assets/audio/piano/manifest.json`
- `reports/piano_alignment_report_<buildId>.md` (plus `.json` and `.csv`)

## Guitar Pack

The guitar pack is rebuilt in-place from its own manifest using the KeyBand alignment logic.

- `docs/assets/audio/guitar/`

Rebuild command:

```bash
python3 scripts/rebuild_pack_from_manifest.py --manifest docs/assets/audio/guitar/manifest.json
```

## Alignment Validation

You can validate any pack against the 8-point gates:

```bash
python3 scripts/validate_audio_pack.py --manifest docs/assets/audio/piano/manifest.json
python3 scripts/validate_audio_pack.py --manifest docs/assets/audio/guitar/manifest.json
```

## Quality Gates ("8 Standards")

The alignment logic uses the same 8-point gates across instruments:

1. onset alignment spread (p10-p90) <= 3ms and max <= 8ms
2. full RMS spread (p10-p90) <= 2.0dB
3. attack RMS spread (p10-p90) <= 3.2dB
4. mid RMS spread (p10-p90) <= 7.0dB
5. max adjacent gain step <= 3.5dB
6. max peak <= 0.90
7. worst mapping error <= 10 cents (Male/Female x Key x Degree 1-7)
8. package size <= 20MB (warns above 10MB)

## Repo Layout

- `scripts/build_piano_samples.py` - alignment + normalization + report generator
- `docs/assets/audio/piano/` - output audio pack + `manifest.json`
- `docs/assets/audio/guitar/` - guitar audio pack + `manifest.json`
- `reports/` - build reports (md/json/csv)
- `docker/` - Nginx config for static hosting
- `docker-compose.yml` - one-command deploy
