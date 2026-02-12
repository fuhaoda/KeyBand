#!/usr/bin/env python3
"""Validate an audio pack against the 8-point alignment gates."""

from __future__ import annotations

import argparse
from array import array
import csv
import json
from math import log10
from pathlib import Path
import shutil
import subprocess
import time

GATE_ONSET_SPREAD_MS = 3.0
GATE_ONSET_MAX_MS = 8.0
GATE_FULL_RMS_SPREAD_DB = 2.0
GATE_ATTACK_RMS_SPREAD_DB = 3.2
GATE_MID_RMS_SPREAD_DB = 7.0
GATE_MAX_ADJACENT_GAIN_STEP_DB = 3.5
GATE_MAX_PEAK_LINEAR = 0.90
GATE_MAX_MAPPING_ERROR_CENTS = 10.0
GATE_MAX_TOTAL_MB = 20.0

ONSET_THRESHOLD_RATIO = 0.08
ONSET_THRESHOLD_FLOOR = 0.003
ONSET_MIN_STREAK = 96
ONSET_ENVELOPE_SMOOTH_SEC = 0.003

ATTACK_ANALYSIS_SEC = 0.35
MID_WINDOW_START_SEC = 0.35
MID_WINDOW_DURATION_SEC = 0.55

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default="docs/assets/audio/piano/manifest.json",
        help="Path to manifest.json",
    )
    parser.add_argument("--report-dir", default="reports", help="Report output directory")
    return parser.parse_args()


def require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is required but not found in PATH")


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


def spread_db_percentile(values: list[float], low_p: float, high_p: float) -> float:
    valid = [value for value in values if value > 0]
    if not valid:
        return 0.0
    sorted_values = sorted(valid)
    low_idx = int(round((len(sorted_values) - 1) * low_p))
    high_idx = int(round((len(sorted_values) - 1) * high_p))
    low_val = sorted_values[low_idx]
    high_val = sorted_values[high_idx]
    return 20 * log10(high_val / low_val)


def spread_linear_percentile(values: list[float], low_p: float, high_p: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    low_idx = int(round((len(sorted_values) - 1) * low_p))
    high_idx = int(round((len(sorted_values) - 1) * high_p))
    return sorted_values[high_idx] - sorted_values[low_idx]


def compute_mapping_error(samples: list[dict]) -> tuple[float, dict]:
    if not samples:
        return 0.0, {}
    by_hz = sorted(samples, key=lambda item: item["hz"])

    def map_target(target_hz: float) -> dict:
        best = None
        for sample in by_hz:
            cents = abs(1200 * log10(target_hz / sample["hz"]) / log10(2))
            if best is None or cents < best["cents"]:
                best = {"sample": sample, "cents": cents}
        return best

    worst = {"cents": 0.0}
    for gender, base_do in (("male", MALE_DO_C), ("female", FEMALE_DO_C)):
        for key, key_offset in KEY_OFFSETS.items():
            do_hz = base_do * (2 ** (key_offset / 12))
            for degree, semitone in MAJOR_DEGREE_TO_SEMITONE.items():
                target = do_hz * (2 ** (semitone / 12))
                mapping = map_target(target)
                if mapping["cents"] > worst["cents"]:
                    worst = {
                        "gender": gender,
                        "key": key,
                        "degree": degree,
                        "targetHz": target,
                        "sampleId": mapping["sample"]["id"],
                        "midi": mapping["sample"]["midi"],
                        "sampleHz": mapping["sample"]["hz"],
                        "cents": mapping["cents"],
                    }
    return float(worst["cents"]), worst


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    require_ffmpeg()

    manifest = json.loads(manifest_path.read_text())
    instrument = manifest.get("instrument", "instrument")
    duration_ms = float(manifest.get("durationMs", 1000))
    sample_rate = int(manifest.get("sampleRate", 44100))
    samples = manifest.get("samples", [])

    web_root = manifest_path.parents[3]

    onset_values = []
    peak_values = []
    full_rms_values = []
    attack_rms_values = []
    mid_rms_values = []
    rows = []

    for sample in samples:
        file_path = sample.get("file", "")
        audio_path = web_root / file_path.lstrip("/")
        samples_raw = decode_mono_float_samples(audio_path, sample_rate)
        onset_sec = find_onset(samples_raw, sample_rate)
        peak = max(abs(value) for value in samples_raw) if samples_raw else 0.0

        onset_idx = int(onset_sec * sample_rate)
        total_len = int((duration_ms / 1000) * sample_rate)
        end_idx = min(len(samples_raw), onset_idx + total_len)
        full_rms = rms(samples_raw, onset_idx, end_idx)
        attack_end = onset_idx + int(ATTACK_ANALYSIS_SEC * sample_rate)
        attack_rms = rms(samples_raw, onset_idx, min(attack_end, end_idx))
        mid_start = onset_idx + int(MID_WINDOW_START_SEC * sample_rate)
        mid_end = mid_start + int(MID_WINDOW_DURATION_SEC * sample_rate)
        mid_rms = rms(samples_raw, min(mid_start, end_idx), min(mid_end, end_idx))

        onset_values.append(onset_sec * 1000.0)
        peak_values.append(peak)
        full_rms_values.append(full_rms)
        attack_rms_values.append(attack_rms)
        mid_rms_values.append(mid_rms)

        rows.append(
            {
                "id": sample["id"],
                "midi": sample["midi"],
                "note": sample["note"],
                "onsetMs": onset_sec * 1000.0,
                "peak": peak,
                "fullRms": full_rms,
                "attackRms": attack_rms,
                "midRms": mid_rms,
            }
        )

    onset_spread = spread_linear_percentile(onset_values, 0.10, 0.90)
    onset_max = max(onset_values) if onset_values else 0.0
    full_spread = spread_db_percentile(full_rms_values, 0.10, 0.90)
    attack_spread = spread_db_percentile(attack_rms_values, 0.10, 0.90)
    mid_spread = spread_db_percentile(mid_rms_values, 0.10, 0.90)

    gains = [sample.get("gainApplied", 1.0) for sample in samples]
    max_adjacent_gain_step = 0.0
    for prev, next_val in zip(gains, gains[1:]):
        if prev <= 0 or next_val <= 0:
            continue
        step_db = abs(20 * log10(next_val / prev))
        if step_db > max_adjacent_gain_step:
            max_adjacent_gain_step = step_db

    size_mb = sum((web_root / sample["file"].lstrip("/")).stat().st_size for sample in samples) / (1024 * 1024)

    mapping_error, mapping_worst = compute_mapping_error(samples)
    max_peak = max(peak_values) if peak_values else 0.0

    epsilon = 1e-6
    gates = {
        "onsetSpreadP10P90Ms": onset_spread <= GATE_ONSET_SPREAD_MS,
        "onsetMaxMs": onset_max <= GATE_ONSET_MAX_MS,
        "fullRmsSpreadDb": full_spread <= GATE_FULL_RMS_SPREAD_DB,
        "attackRmsSpreadDb": attack_spread <= GATE_ATTACK_RMS_SPREAD_DB,
        "midRmsSpreadDb": mid_spread <= GATE_MID_RMS_SPREAD_DB,
        "maxAdjacentGainStepDb": max_adjacent_gain_step
        <= (GATE_MAX_ADJACENT_GAIN_STEP_DB + epsilon),
        "maxPeakLinear": max_peak <= GATE_MAX_PEAK_LINEAR,
        "maxMappingErrorCents": mapping_error <= GATE_MAX_MAPPING_ERROR_CENTS,
        "sizeMb": size_mb <= GATE_MAX_TOTAL_MB,
    }
    passed = all(gates.values())

    build_id = int(time.time())
    report_path = report_dir / f"{instrument}_alignment_report_{build_id}.md"
    csv_path = report_dir / f"{instrument}_alignment_samples_{build_id}.csv"

    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        f"# {instrument.capitalize()} Alignment Report",
        "",
        f"Pass: {'YES' if passed else 'NO'}",
        "",
        "## Metrics",
        f"- onset spread p10-p90: {onset_spread:.3f} ms",
        f"- onset max: {onset_max:.3f} ms",
        f"- full RMS spread: {full_spread:.3f} dB",
        f"- attack RMS spread: {attack_spread:.3f} dB",
        f"- mid RMS spread: {mid_spread:.3f} dB",
        f"- max adjacent gain step: {max_adjacent_gain_step:.3f} dB",
        f"- max peak: {max_peak:.6f}",
        f"- worst mapping error: {mapping_error:.3f} cents",
        f"- pack size: {size_mb:.2f} MB",
        "",
        "## Gates",
    ]
    for key, ok in gates.items():
        lines.append(f"- {key}: {'PASS' if ok else 'FAIL'}")

    if mapping_worst:
        lines.extend(
            [
                "",
                "## Worst Mapping",
                f"- gender: {mapping_worst['gender']}",
                f"- key: {mapping_worst['key']}",
                f"- degree: {mapping_worst['degree']}",
                f"- targetHz: {mapping_worst['targetHz']:.3f}",
                f"- sampleId: {mapping_worst['sampleId']}",
                f"- sampleHz: {mapping_worst['sampleHz']:.3f}",
                f"- centsError: {mapping_worst['cents']:.3f}",
            ]
        )

    report_path.write_text("\n".join(lines))
    print(f"Wrote report: {report_path}")
    if not passed:
        raise SystemExit("Alignment gates failed")


if __name__ == "__main__":
    main()
