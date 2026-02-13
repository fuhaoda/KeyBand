#!/usr/bin/env python3
"""Download and rebuild aligned flute samples (non-vib mf) for KeyBand Studio."""

from __future__ import annotations

import argparse
from array import array
from dataclasses import dataclass
import csv
import json
from math import exp, log10, log2, log
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from urllib.parse import quote
from urllib.request import urlretrieve

SOURCE_BASE_URL = "https://theremin.music.uiowa.edu/sound%20files/MIS/Woodwinds/flute"

SOURCE_FILES = [
    ("Flute.nonvib.mf.B3B4.aiff", "B3", "B4"),
    ("Flute.nonvib.mf.C5B5.aiff", "C5", "B5"),
    ("Flute.nonvib.mf.C6B6.aiff", "C6", "B6"),
    ("Flute.nonvib.mf.C7.aiff", "C7", "C7"),
]

NOTE_NAMES_FLAT = ("C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B")

# Onset detection.
ONSET_THRESHOLD_RATIO = 0.08
ONSET_THRESHOLD_FLOOR = 0.003
ONSET_MIN_STREAK = 96
ONSET_ENVELOPE_SMOOTH_SEC = 0.003
ONSET_MIN_SILENCE_SEC = 0.08
START_PREROLL_SEC = 0.004

FADE_IN_SEC = 0.001
FADE_OUT_SEC = 0.08

# Loudness windows.
ATTACK_ANALYSIS_SEC = 0.35
MID_WINDOW_START_SEC = 0.35
MID_WINDOW_DURATION_SEC = 0.55
FULL_WINDOW_SEC = 1.5

TARGET_PEAK_LINEAR = 0.90
GAIN_CLAMP_MIN = 0.60
GAIN_CLAMP_MAX = 3.00
GAIN_SMOOTHING_LAMBDA = 0.08
GAIN_SMOOTHING_PASSES = 1

QUALITY_SPREAD_LOW_PERCENTILE = 0.10
QUALITY_SPREAD_HIGH_PERCENTILE = 0.90

GATE_ONSET_SPREAD_MS = 3.0
GATE_ONSET_MAX_MS = 8.0
GATE_FULL_RMS_SPREAD_DB = 2.0
GATE_ATTACK_RMS_SPREAD_DB = 3.2
GATE_MID_RMS_SPREAD_DB = 7.0
GATE_MAX_ADJACENT_GAIN_STEP_DB = 3.5
GATE_MAX_MAPPING_ERROR_CENTS = 10.0

MALE_DO_C = 130.8
FEMALE_DO_C = 261.6
KEY_OFFSETS = {
    "C": 0,
    "C#/Db": 1,
    "D": 2,
    "D#/Eb": 3,
    "E": 4,
    "F": 5,
    "F#/Gb": 6,
    "G": 7,
    "G#/Ab": 8,
    "A": 9,
    "A#/Bb": 10,
    "B": 11,
}
MAJOR_DEGREE_TO_SEMITONE = {1: 0, 2: 2, 3: 4, 4: 5, 5: 7, 6: 9, 7: 11}


@dataclass(frozen=True)
class SampleSpec:
    id: str
    midi: int
    note: str
    hz: float
    source_filename: str
    output_filename: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="docs/assets/audio/flute", help="Output directory")
    parser.add_argument("--cache-dir", default=".cache/flute_nonvib_mf", help="Source cache directory")
    parser.add_argument("--report-dir", default="reports", help="Report output directory")
    parser.add_argument("--duration", type=float, default=6.0, help="Output duration in seconds")
    parser.add_argument("--sample-rate", type=int, default=44100, help="Output sample rate")
    parser.add_argument("--bitrate", default="192k", help="AAC bitrate")
    parser.add_argument("--clean", action="store_true", help="Remove output dir before build")
    parser.add_argument("--refresh-sources", action="store_true", help="Re-download sources")
    return parser.parse_args()


def require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is required but not found in PATH")


def midi_to_hz(midi: int) -> float:
    return 440.0 * (2 ** ((midi - 69) / 12))


def note_to_midi(note: str) -> int:
    name = note[:-1]
    octave = int(note[-1])
    semitone_map = {
        "C": 0,
        "C#": 1,
        "Db": 1,
        "D": 2,
        "D#": 3,
        "Eb": 3,
        "E": 4,
        "F": 5,
        "F#": 6,
        "Gb": 6,
        "G": 7,
        "G#": 8,
        "Ab": 8,
        "A": 9,
        "A#": 10,
        "Bb": 10,
        "B": 11,
    }
    semitone = semitone_map[name]
    return (octave + 1) * 12 + semitone


def midi_to_note_name(midi: int) -> str:
    note_name = NOTE_NAMES_FLAT[midi % 12]
    octave = (midi // 12) - 1
    return f"{note_name}{octave}"


def expand_note_range(start_note: str, end_note: str) -> list[int]:
    start_midi = note_to_midi(start_note)
    end_midi = note_to_midi(end_note)
    return list(range(start_midi, end_midi + 1))


def source_url_for_filename(filename: str) -> str:
    return f"{SOURCE_BASE_URL}/{quote(filename)}"


def download_sources(cache_dir: Path, refresh_sources: bool) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    for filename, _, _ in SOURCE_FILES:
        source_path = cache_dir / filename
        if refresh_sources and source_path.exists():
            source_path.unlink()
        if source_path.exists():
            continue
        source_url = source_url_for_filename(filename)
        print(f"Downloading {source_url}")
        urlretrieve(source_url, source_path)


def decode_mono_float_samples(input_path: Path, sample_rate: int) -> array:
    ffmpeg_cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(input_path),
        "-f",
        "f32le",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "pipe:1",
    ]
    proc = subprocess.run(ffmpeg_cmd, capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode("utf-8", errors="ignore").strip())
    samples = array("f")
    samples.frombytes(proc.stdout)
    return samples


def smooth_envelope(samples: array, sample_rate: int) -> list[float]:
    window = max(1, int(sample_rate * ONSET_ENVELOPE_SMOOTH_SEC))
    result: list[float] = []
    acc = 0.0
    for index, value in enumerate(samples):
        acc += abs(value)
        if index >= window:
            acc -= abs(samples[index - window])
        result.append(acc / min(index + 1, window))
    return result


def detect_scale_onsets(samples: array, sample_rate: int, expected_count: int) -> list[int]:
    min_silence = max(1, int(sample_rate * ONSET_MIN_SILENCE_SEC))
    envelope = smooth_envelope(samples, sample_rate)
    peak = max(envelope) if envelope else 0.0

    def run_with_ratio(ratio: float) -> list[int]:
        threshold = max(peak * ratio, ONSET_THRESHOLD_FLOOR)
        onsets: list[int] = []
        streak = 0
        in_note = False
        silence = 0
        for idx, value in enumerate(envelope):
            if value >= threshold:
                silence = 0
                if not in_note:
                    streak += 1
                    if streak >= ONSET_MIN_STREAK:
                        onset_idx = idx - streak + 1
                        onsets.append(onset_idx)
                        in_note = True
                        streak = 0
                continue
            if in_note:
                silence += 1
                if silence >= min_silence:
                    in_note = False
                    silence = 0
            else:
                streak = 0
        return onsets

    for ratio in (0.06, 0.08, 0.10, 0.12):
        onsets = run_with_ratio(ratio)
        if expected_count and len(onsets) == expected_count:
            return onsets
    return run_with_ratio(ONSET_THRESHOLD_RATIO)


def find_onset(samples: array, sample_rate: int) -> float:
    if not samples:
        return 0.0
    envelope = smooth_envelope(samples, sample_rate)
    peak = max(envelope)
    threshold = max(peak * ONSET_THRESHOLD_RATIO, ONSET_THRESHOLD_FLOOR)
    streak = 0
    for idx, value in enumerate(envelope):
        if value >= threshold:
            streak += 1
            if streak >= ONSET_MIN_STREAK:
                onset_index = idx - streak + 1
                return onset_index / sample_rate
        else:
            streak = 0
    return 0.0


def rms(samples: array, start_idx: int, end_idx: int) -> float:
    if end_idx <= start_idx:
        return 0.0
    acc = 0.0
    count = end_idx - start_idx
    for idx in range(start_idx, end_idx):
        value = samples[idx]
        acc += value * value
    return (acc / count) ** 0.5


def percentile(values: list[float], percent: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    index = int(round((len(values) - 1) * percent))
    return values[index]


def spread_db_percentile(values: list[float], low: float, high: float) -> float:
    nonzero = [value for value in values if value > 0]
    if not nonzero:
        return 0.0
    low_val = percentile(nonzero, low)
    high_val = percentile(nonzero, high)
    return 20 * log10(high_val / low_val)


def spread_linear_percentile(values: list[float], low: float, high: float) -> float:
    if not values:
        return 0.0
    low_val = percentile(values, low)
    high_val = percentile(values, high)
    return high_val - low_val


def smooth_gain_map_by_neighbors(ordered_ids: list[str], gain_map: dict[str, float]) -> dict[str, float]:
    log_gain = {sid: log(max(gain_map.get(sid, 1.0), 1e-12)) for sid in ordered_ids}
    for _ in range(GAIN_SMOOTHING_PASSES):
        next_log_gain = dict(log_gain)
        for idx, sid in enumerate(ordered_ids):
            neighbors = []
            if idx > 0:
                neighbors.append(log_gain[ordered_ids[idx - 1]])
            if idx < len(ordered_ids) - 1:
                neighbors.append(log_gain[ordered_ids[idx + 1]])
            if not neighbors:
                continue
            neighbor_avg = sum(neighbors) / len(neighbors)
            next_log_gain[sid] = (1.0 - GAIN_SMOOTHING_LAMBDA) * log_gain[sid] + GAIN_SMOOTHING_LAMBDA * neighbor_avg
        log_gain = next_log_gain

    smoothed: dict[str, float] = {}
    for sid, value in log_gain.items():
        gain = exp(value)
        smoothed[sid] = max(GAIN_CLAMP_MIN, min(GAIN_CLAMP_MAX, gain))
    return smoothed


def compute_gain_map_from_blended_rms(
    specs: list[SampleSpec],
    peak_map: dict[str, float],
    full_rms_map: dict[str, float],
    attack_rms_map: dict[str, float],
    mid_rms_map: dict[str, float],
) -> tuple[dict[str, float], float, float]:
    from statistics import median

    full_values = [value for value in full_rms_map.values() if value > 0]
    attack_values = [value for value in attack_rms_map.values() if value > 0]
    mid_values = [value for value in mid_rms_map.values() if value > 0]
    if not full_values or not attack_values or not mid_values:
        raise SystemExit("Cannot normalize samples: missing RMS values")

    target_full = float(median(full_values))
    gain_map: dict[str, float] = {}
    for spec in specs:
        full_rms = max(full_rms_map.get(spec.id, 0.0), 1e-12)
        gain = target_full / full_rms
        gain_map[spec.id] = max(GAIN_CLAMP_MIN, min(GAIN_CLAMP_MAX, gain))

    ordered_specs = sorted(specs, key=lambda item: item.midi)
    ordered_ids = [spec.id for spec in ordered_specs]
    gain_map = smooth_gain_map_by_neighbors(ordered_ids, gain_map)

    max_step_ratio = pow(10.0, GATE_MAX_ADJACENT_GAIN_STEP_DB / 20.0)
    for index, spec in enumerate(ordered_specs):
        if index == 0:
            continue
        prev_spec = ordered_specs[index - 1]
        prev_gain = max(gain_map.get(prev_spec.id, 1.0), 1e-12)
        current_gain = max(gain_map.get(spec.id, 1.0), 1e-12)
        if current_gain > prev_gain * max_step_ratio:
            gain_map[spec.id] = prev_gain * max_step_ratio
        elif current_gain < prev_gain / max_step_ratio:
            gain_map[spec.id] = prev_gain / max_step_ratio

    max_predicted_peak = 0.0
    for spec in specs:
        predicted_peak = peak_map.get(spec.id, 0.0) * gain_map.get(spec.id, 1.0)
        if predicted_peak > max_predicted_peak:
            max_predicted_peak = predicted_peak

    global_peak_scale = 1.0
    if max_predicted_peak > TARGET_PEAK_LINEAR and max_predicted_peak > 0:
        global_peak_scale = TARGET_PEAK_LINEAR / max_predicted_peak
        for sid in list(gain_map):
            gain_map[sid] *= global_peak_scale

    return gain_map, target_full, global_peak_scale


def compute_max_adjacent_gain_step_db(specs: list[SampleSpec], gain_map: dict[str, float]) -> float:
    ordered = sorted(specs, key=lambda item: item.midi)
    max_step = 0.0
    for idx in range(1, len(ordered)):
        prev_gain = max(gain_map.get(ordered[idx - 1].id, 1.0), 1e-12)
        current_gain = max(gain_map.get(ordered[idx].id, 1.0), 1e-12)
        step_db = abs(20.0 * log10(current_gain / prev_gain))
        if step_db > max_step:
            max_step = step_db
    return max_step


def compute_mapping_error(specs: list[SampleSpec]) -> tuple[float, dict]:
    if not specs:
        return 0.0, {}
    ordered = sorted(specs, key=lambda item: item.hz)
    min_hz = ordered[0].hz
    max_hz = ordered[-1].hz
    worst = {"cents": 0.0}
    for gender, base_do in (("male", MALE_DO_C), ("female", FEMALE_DO_C)):
        for key, key_offset in KEY_OFFSETS.items():
            do_hz = base_do * (2 ** (key_offset / 12))
            for degree, semitone in MAJOR_DEGREE_TO_SEMITONE.items():
                target = do_hz * (2 ** (semitone / 12))
                if target < min_hz or target > max_hz:
                    pass
                nearest = min(ordered, key=lambda spec: abs(1200 * log2(target / spec.hz)))
                cents = 1200 * log2(target / nearest.hz)
                if abs(cents) > worst["cents"]:
                    worst = {
                        "gender": gender,
                        "key": key,
                        "degree": degree,
                        "targetHz": target,
                        "sampleId": nearest.id,
                        "midi": nearest.midi,
                        "sampleHz": nearest.hz,
                        "cents": cents,
                    }
    return abs(worst["cents"]), worst


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    if args.clean and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    require_ffmpeg()
    download_sources(cache_dir, args.refresh_sources)

    specs: list[SampleSpec] = []
    for filename, start_note, end_note in SOURCE_FILES:
        for midi in expand_note_range(start_note, end_note):
            note = midi_to_note_name(midi)
            sample_id = f"m{midi:03d}"
            specs.append(
                SampleSpec(
                    id=sample_id,
                    midi=midi,
                    note=note,
                    hz=midi_to_hz(midi),
                    source_filename=filename,
                    output_filename=f"{sample_id}.m4a",
                )
            )

    specs = sorted(specs, key=lambda item: item.midi)

    before_onset_map: dict[str, float] = {}
    after_onset_map: dict[str, float] = {}
    peak_map: dict[str, float] = {}
    full_rms_map: dict[str, float] = {}
    attack_rms_map: dict[str, float] = {}
    mid_rms_map: dict[str, float] = {}

    temp_wavs: dict[str, Path] = {}

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        for filename, start_note, end_note in SOURCE_FILES:
            midi_range = expand_note_range(start_note, end_note)
            source_path = cache_dir / filename
            samples_raw = decode_mono_float_samples(source_path, args.sample_rate)
            onsets = detect_scale_onsets(samples_raw, args.sample_rate, len(midi_range))
            if len(onsets) != len(midi_range):
                raise SystemExit(
                    f"Onset detection failed for {filename}: expected {len(midi_range)} onsets, got {len(onsets)}"
                )
            for onset_idx, midi in zip(onsets, midi_range):
                note = midi_to_note_name(midi)
                sample_id = f"m{midi:03d}"
                before_onset_map[sample_id] = onset_idx / args.sample_rate
                trim_start = max(0.0, before_onset_map[sample_id] - START_PREROLL_SEC)
                temp_wav = temp_dir_path / f"{sample_id}.wav"
                temp_wavs[sample_id] = temp_wav
                trim_cmd = [
                    "ffmpeg",
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    str(source_path),
                    "-af",
                    (
                        f"atrim=start={trim_start:.6f},asetpts=PTS-STARTPTS,"
                        f"apad,atrim=end={args.duration:.6f},"
                        f"afade=t=in:st=0:d={FADE_IN_SEC:.6f},"
                        f"afade=t=out:st={max(0.0, args.duration - FADE_OUT_SEC):.6f}:d={FADE_OUT_SEC:.6f}"
                    ),
                    "-ac",
                    "1",
                    "-ar",
                    str(args.sample_rate),
                    str(temp_wav),
                ]
                subprocess.run(trim_cmd, check=True)

                aligned_samples = decode_mono_float_samples(temp_wav, args.sample_rate)
                onset_after = START_PREROLL_SEC
                after_onset_map[sample_id] = onset_after

                peak_map[sample_id] = max(abs(value) for value in aligned_samples) if aligned_samples else 0.0
                onset_idx = int(onset_after * args.sample_rate)
                end_idx = min(len(aligned_samples), onset_idx + int(FULL_WINDOW_SEC * args.sample_rate))
                full_rms_map[sample_id] = rms(aligned_samples, onset_idx, end_idx)
                attack_end = onset_idx + int(ATTACK_ANALYSIS_SEC * args.sample_rate)
                attack_rms_map[sample_id] = rms(aligned_samples, onset_idx, min(attack_end, end_idx))
                mid_start = onset_idx + int(MID_WINDOW_START_SEC * args.sample_rate)
                mid_end = mid_start + int(MID_WINDOW_DURATION_SEC * args.sample_rate)
                mid_rms_map[sample_id] = rms(aligned_samples, min(mid_start, end_idx), min(mid_end, end_idx))

        gain_map, target_blended_rms, global_peak_scale = compute_gain_map_from_blended_rms(
            specs, peak_map, full_rms_map, attack_rms_map, mid_rms_map
        )

        for spec in specs:
            gain = gain_map.get(spec.id, 1.0)
            temp_wav = temp_wavs[spec.id]
            output_path = output_dir / spec.output_filename
            ffmpeg_cmd = [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(temp_wav),
                "-af",
                f"volume={gain:.10f}",
                "-ac",
                "1",
                "-ar",
                str(args.sample_rate),
                "-c:a",
                "aac",
                "-b:a",
                args.bitrate,
                str(output_path),
            ]
            subprocess.run(ffmpeg_cmd, check=True)

    after_full_rms = {sid: full_rms_map[sid] * gain_map.get(sid, 1.0) for sid in full_rms_map}
    after_attack_rms = {sid: attack_rms_map[sid] * gain_map.get(sid, 1.0) for sid in attack_rms_map}
    after_mid_rms = {sid: mid_rms_map[sid] * gain_map.get(sid, 1.0) for sid in mid_rms_map}
    after_peak = {sid: peak_map[sid] * gain_map.get(sid, 1.0) for sid in peak_map}

    after_onset_values = [value * 1000.0 for value in after_onset_map.values()]
    onset_spread_ms = spread_linear_percentile(after_onset_values, QUALITY_SPREAD_LOW_PERCENTILE, QUALITY_SPREAD_HIGH_PERCENTILE)
    onset_max_ms = max(after_onset_values) if after_onset_values else 0.0

    full_spread = spread_db_percentile(list(after_full_rms.values()), QUALITY_SPREAD_LOW_PERCENTILE, QUALITY_SPREAD_HIGH_PERCENTILE)
    attack_spread = spread_db_percentile(list(after_attack_rms.values()), QUALITY_SPREAD_LOW_PERCENTILE, QUALITY_SPREAD_HIGH_PERCENTILE)
    mid_spread = spread_db_percentile(list(after_mid_rms.values()), QUALITY_SPREAD_LOW_PERCENTILE, QUALITY_SPREAD_HIGH_PERCENTILE)

    max_adjacent_gain_step = compute_max_adjacent_gain_step_db(specs, gain_map)
    max_peak = max(after_peak.values()) if after_peak else 0.0

    mapping_error, mapping_worst = compute_mapping_error(specs)

    total_bytes = sum(path.stat().st_size for path in output_dir.glob("*.m4a"))
    total_mb = total_bytes / (1024 * 1024)

    epsilon = 1e-6
    gates = {
        "onsetSpreadP10P90Ms": onset_spread_ms <= GATE_ONSET_SPREAD_MS,
        "onsetMaxMs": onset_max_ms <= GATE_ONSET_MAX_MS,
        "fullRmsSpreadDb": full_spread <= GATE_FULL_RMS_SPREAD_DB,
        "attackRmsSpreadDb": attack_spread <= GATE_ATTACK_RMS_SPREAD_DB,
        "midRmsSpreadDb": mid_spread <= GATE_MID_RMS_SPREAD_DB,
        "maxAdjacentGainStepDb": max_adjacent_gain_step <= (GATE_MAX_ADJACENT_GAIN_STEP_DB + epsilon),
        "maxPeakLinear": max_peak <= TARGET_PEAK_LINEAR,
        "maxMappingErrorCents": mapping_error <= GATE_MAX_MAPPING_ERROR_CENTS,
        "maxTotalMb": total_mb <= 20.0,
    }
    passed = all(gates.values())

    build_id = int(time.time())
    report_md = report_dir / f"flute_alignment_report_{build_id}.md"
    report_json = report_dir / f"flute_alignment_report_{build_id}.json"
    report_csv = report_dir / f"flute_alignment_samples_{build_id}.csv"

    rows = []
    for spec in specs:
        rows.append(
            {
                "id": spec.id,
                "midi": spec.midi,
                "note": spec.note,
                "onsetBeforeSec": round(before_onset_map.get(spec.id, 0.0), 6),
                "onsetAfterSec": round(after_onset_map.get(spec.id, 0.0), 6),
                "gainApplied": round(gain_map.get(spec.id, 1.0), 10),
                "peakAfter": round(after_peak.get(spec.id, 0.0), 8),
                "fullRmsAfter": round(after_full_rms.get(spec.id, 0.0), 8),
                "attackRmsAfter": round(after_attack_rms.get(spec.id, 0.0), 8),
                "midRmsAfter": round(after_mid_rms.get(spec.id, 0.0), 8),
            }
        )

    with report_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    report_payload = {
        "instrument": "flute",
        "passed": passed,
        "metrics": {
            "onsetSpreadP10P90Ms": onset_spread_ms,
            "onsetMaxMs": onset_max_ms,
            "fullRmsSpreadDb": full_spread,
            "attackRmsSpreadDb": attack_spread,
            "midRmsSpreadDb": mid_spread,
            "maxAdjacentGainStepDb": max_adjacent_gain_step,
            "maxPeakLinear": max_peak,
            "maxMappingErrorCents": mapping_error,
            "sizeMb": total_mb,
        },
        "gates": gates,
        "mappingWorst": mapping_worst,
    }
    report_json.write_text(json.dumps(report_payload, indent=2))

    report_lines = [
        "# Flute Alignment Report",
        "",
        f"Pass: {'YES' if passed else 'NO'}",
        "",
        "## Metrics",
        f"- onset spread p10-p90: {onset_spread_ms:.3f} ms",
        f"- onset max: {onset_max_ms:.3f} ms",
        f"- full RMS spread: {full_spread:.3f} dB",
        f"- attack RMS spread: {attack_spread:.3f} dB",
        f"- mid RMS spread: {mid_spread:.3f} dB",
        f"- max adjacent gain step: {max_adjacent_gain_step:.3f} dB",
        f"- max peak: {max_peak:.6f}",
        f"- worst mapping error: {mapping_error:.3f} cents",
        f"- pack size: {total_mb:.2f} MB",
        "",
        "## Gates",
    ]
    for key, ok in gates.items():
        report_lines.append(f"- {key}: {'PASS' if ok else 'FAIL'}")
    report_md.write_text("\n".join(report_lines))

    manifest = {
        "version": 1,
        "instrument": "flute",
        "buildId": build_id,
        "source": "University of Iowa MIS Flute (non-vib mf)",
        "sourceUrl": SOURCE_BASE_URL,
        "sourceFiles": [name for name, _, _ in SOURCE_FILES],
        "durationMs": int(args.duration * 1000),
        "sampleRate": args.sample_rate,
        "codec": "aac",
        "bitrate": args.bitrate,
        "sampleHzRange": [round(specs[0].hz, 6), round(specs[-1].hz, 6)],
        "alignment": {
            "method": "peak_ratio_threshold_consecutive_streak_trim_with_preroll",
            "thresholdRatio": ONSET_THRESHOLD_RATIO,
            "thresholdFloor": ONSET_THRESHOLD_FLOOR,
            "minStreakSamples": ONSET_MIN_STREAK,
            "envelopeSmoothMs": int(ONSET_ENVELOPE_SMOOTH_SEC * 1000),
            "startPrerollMs": int(START_PREROLL_SEC * 1000),
            "fadeInMs": int(FADE_IN_SEC * 1000),
            "fadeOutMs": int(FADE_OUT_SEC * 1000),
        },
        "normalization": {
            "method": "full_window_dominant_rms_with_neighbor_smoothing_and_peak_guard",
            "targetBlendedRms": round(target_blended_rms, 10),
            "targetPeakLinear": TARGET_PEAK_LINEAR,
            "globalPeakScale": round(global_peak_scale, 6),
            "gainClamp": [GAIN_CLAMP_MIN, GAIN_CLAMP_MAX],
            "gainSmoothing": {"lambda": GAIN_SMOOTHING_LAMBDA, "passes": GAIN_SMOOTHING_PASSES},
            "gainRange": [round(min(gain_map.values()), 8), round(max(gain_map.values()), 8)],
        },
        "quality": {
            "spreadMethod": {
                "onsetMs": "p10_p90",
                "fullRms": "p10_p90",
                "attackRms": "p10_p90",
                "midRms": "p10_p90",
            },
            "onsetSpreadP10P90Ms": round(onset_spread_ms, 4),
            "onsetMaxMs": round(onset_max_ms, 4),
            "fullRmsSpreadDb": round(full_spread, 4),
            "attackRmsSpreadDb": round(attack_spread, 4),
            "midRmsSpreadDb": round(mid_spread, 4),
            "maxAdjacentGainStepDb": round(max_adjacent_gain_step, 4),
            "maxPeakLinear": round(max_peak, 6),
            "thresholds": {
                "onsetSpreadP10P90Ms": GATE_ONSET_SPREAD_MS,
                "onsetMaxMs": GATE_ONSET_MAX_MS,
                "fullRmsSpreadDb": GATE_FULL_RMS_SPREAD_DB,
                "attackRmsSpreadDb": GATE_ATTACK_RMS_SPREAD_DB,
                "midRmsSpreadDb": GATE_MID_RMS_SPREAD_DB,
                "maxAdjacentGainStepDb": GATE_MAX_ADJACENT_GAIN_STEP_DB,
                "maxPeakLinear": TARGET_PEAK_LINEAR,
                "maxMappingErrorCents": GATE_MAX_MAPPING_ERROR_CENTS,
                "maxTotalMb": 20.0,
            },
        },
        "sizeMb": round(total_mb, 4),
        "mappingWorstAbsCents": round(mapping_error, 6),
        "mappingWorst": mapping_worst,
        "report": {
            "md": str(report_md),
            "json": str(report_json),
            "csv": str(report_csv),
        },
        "samples": [
            {
                "id": spec.id,
                "midi": spec.midi,
                "note": spec.note,
                "hz": spec.hz,
                "durationMs": int(args.duration * 1000),
                "gainApplied": round(gain_map.get(spec.id, 1.0), 10),
                "onsetSecBefore": round(before_onset_map.get(spec.id, 0.0), 6),
                "onsetSecAfter": round(after_onset_map.get(spec.id, 0.0), 6),
                "file": f"assets/audio/flute/{spec.output_filename}",
            }
            for spec in specs
        ],
    }

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"Built flute samples: {len(specs)} files, total {total_mb:.2f}MB")
    print(f"Report: {report_md}")
    if not passed:
        raise SystemExit("Flute alignment gates failed")


if __name__ == "__main__":
    main()
