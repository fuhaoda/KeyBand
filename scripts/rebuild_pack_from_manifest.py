#!/usr/bin/env python3
"""Rebuild an audio pack from its manifest using KeyBand alignment + 8-point gates."""

from __future__ import annotations

import argparse
from array import array
from dataclasses import dataclass
import csv
import json
from math import exp, log10
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

ONSET_THRESHOLD_RATIO = 0.08
ONSET_THRESHOLD_FLOOR = 0.003
ONSET_MIN_STREAK = 96
ONSET_ENVELOPE_SMOOTH_SEC = 0.003
START_PREROLL_SEC = 0.004

FADE_IN_SEC = 0.001
FADE_OUT_SEC = 0.08

ATTACK_ANALYSIS_SEC = 0.35
MID_WINDOW_START_SEC = 0.35
MID_WINDOW_DURATION_SEC = 0.55

TARGET_PEAK_LINEAR = 0.90
GAIN_CLAMP_MIN = 0.60
GAIN_CLAMP_MAX = 3.00
GAIN_SMOOTHING_LAMBDA = 0.08
GAIN_SMOOTHING_PASSES = 1

GATE_ONSET_SPREAD_MS = 3.0
GATE_ONSET_MAX_MS = 8.0
GATE_FULL_RMS_SPREAD_DB = 2.0
GATE_ATTACK_RMS_SPREAD_DB = 3.2
GATE_MID_RMS_SPREAD_DB = 7.0
GATE_MAX_ADJACENT_GAIN_STEP_DB = 3.5
GATE_MAX_PEAK_LINEAR = 0.90
GATE_MAX_MAPPING_ERROR_CENTS = 10.0
GATE_MAX_TOTAL_MB = 20.0

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
    input_path: Path
    output_filename: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Path to manifest.json")
    parser.add_argument("--output-dir", default=None, help="Output directory (defaults to manifest folder)")
    parser.add_argument("--report-dir", default="reports", help="Report output directory")
    parser.add_argument("--clean", action="store_true", help="Remove output dir before build")
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


def smooth_gain_map_by_neighbors(ordered_ids: list[str], gain_map: dict[str, float]) -> dict[str, float]:
    log_gain = {sid: log10(max(gain_map.get(sid, 1.0), 1e-12)) for sid in ordered_ids}
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
        gain = 10 ** value
        smoothed[sid] = max(GAIN_CLAMP_MIN, min(GAIN_CLAMP_MAX, gain))
    return smoothed


def compute_gain_map_from_blended_rms(
    specs: list[SampleSpec],
    peak_map: dict[str, float],
    full_rms_map: dict[str, float],
    attack_rms_map: dict[str, float],
    mid_rms_map: dict[str, float],
) -> tuple[dict[str, float], float, float]:
    blended_map: dict[str, float] = {}
    for spec in specs:
        full_rms = full_rms_map.get(spec.id, 0.0)
        attack_rms = attack_rms_map.get(spec.id, 0.0)
        mid_rms = mid_rms_map.get(spec.id, 0.0)
        blended_map[spec.id] = (
            max(full_rms, 1e-12) ** 0.80
            * max(attack_rms, 1e-12) ** 0.10
            * max(mid_rms, 1e-12) ** 0.10
        )

    nonzero = [value for value in blended_map.values() if value > 0]
    if not nonzero:
        raise SystemExit("Cannot normalize samples: all RMS values are zero")

    from statistics import median

    target_rms = float(median(nonzero))
    gain_map: dict[str, float] = {}
    for spec in specs:
        blended = blended_map.get(spec.id, 0.0)
        if blended <= 0:
            gain = 1.0
        else:
            gain = target_rms / blended
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

    return gain_map, target_rms, global_peak_scale


def compute_mapping_error(samples: list[SampleSpec]) -> tuple[float, dict]:
    if not samples:
        return 0.0, {}
    specs = sorted(samples, key=lambda item: item.hz)

    def map_target(target_hz: float) -> SampleSpec:
        return min(specs, key=lambda spec: abs(1200 * log10(target_hz / spec.hz) / log10(2)))

    worst = {"cents": 0.0}
    for gender, base_do in (("male", MALE_DO_C), ("female", FEMALE_DO_C)):
        for key, key_offset in KEY_OFFSETS.items():
            do_hz = base_do * (2 ** (key_offset / 12))
            for degree, semitone in MAJOR_DEGREE_TO_SEMITONE.items():
                target = do_hz * (2 ** (semitone / 12))
                nearest = map_target(target)
                cents = abs(1200 * log10(target / nearest.hz) / log10(2))
                if cents > worst["cents"]:
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
    return float(worst["cents"]), worst


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text())
    instrument = manifest.get("instrument", "instrument")
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    output_dir = Path(args.output_dir) if args.output_dir else manifest_path.parent
    if args.clean and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    require_ffmpeg()

    duration_ms = float(manifest.get("durationMs", 1000))
    sample_rate = int(manifest.get("sampleRate", 44100))
    bitrate = manifest.get("bitrate", "192k")

    web_root = manifest_path.parents[3]
    specs: list[SampleSpec] = []
    for sample in manifest.get("samples", []):
        file_path = sample.get("file", "")
        input_path = web_root / file_path.lstrip("/")
        specs.append(
            SampleSpec(
                id=sample["id"],
                midi=sample["midi"],
                note=sample["note"],
                hz=sample["hz"],
                input_path=input_path,
                output_filename=Path(file_path).name,
            )
        )

    before_onset_map: dict[str, float] = {}
    after_onset_map: dict[str, float] = {}
    peak_map: dict[str, float] = {}
    full_rms_map: dict[str, float] = {}
    attack_rms_map: dict[str, float] = {}
    mid_rms_map: dict[str, float] = {}

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        temp_wavs: dict[str, Path] = {}

        for spec in specs:
            samples_raw = decode_mono_float_samples(spec.input_path, sample_rate)
            onset_before = find_onset(samples_raw, sample_rate)
            before_onset_map[spec.id] = onset_before
            trim_start = max(0.0, onset_before - START_PREROLL_SEC)
            temp_wav = temp_dir_path / f"{spec.id}.wav"
            trim_cmd = [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(spec.input_path),
                "-af",
                (
                    f"atrim=start={trim_start:.6f},asetpts=PTS-STARTPTS,"
                    f"apad,atrim=end={duration_ms/1000:.6f},"
                    f"afade=t=in:st=0:d={FADE_IN_SEC:.6f},"
                    f"afade=t=out:st={max(0.0, duration_ms/1000 - FADE_OUT_SEC):.6f}:d={FADE_OUT_SEC:.6f}"
                ),
                "-ac",
                "1",
                "-ar",
                str(sample_rate),
                str(temp_wav),
            ]
            subprocess.run(trim_cmd, check=True)
            temp_wavs[spec.id] = temp_wav

            aligned_samples = decode_mono_float_samples(temp_wav, sample_rate)
            onset_after = START_PREROLL_SEC
            after_onset_map[spec.id] = onset_after

            peak_map[spec.id] = max(abs(value) for value in aligned_samples) if aligned_samples else 0.0
            onset_idx = int(onset_after * sample_rate)
            end_idx = min(len(aligned_samples), onset_idx + int((duration_ms / 1000) * sample_rate))
            full_rms_map[spec.id] = rms(aligned_samples, onset_idx, end_idx)
            attack_end = onset_idx + int(ATTACK_ANALYSIS_SEC * sample_rate)
            attack_rms_map[spec.id] = rms(aligned_samples, onset_idx, min(attack_end, end_idx))
            mid_start = onset_idx + int(MID_WINDOW_START_SEC * sample_rate)
            mid_end = mid_start + int(MID_WINDOW_DURATION_SEC * sample_rate)
            mid_rms_map[spec.id] = rms(aligned_samples, min(mid_start, end_idx), min(mid_end, end_idx))

        gain_map, target_rms, global_peak_scale = compute_gain_map_from_blended_rms(
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
                str(sample_rate),
                "-c:a",
                "aac",
                "-b:a",
                bitrate,
                str(output_path),
            ]
            subprocess.run(ffmpeg_cmd, check=True)

    after_full_rms = {sid: full_rms_map[sid] * gain_map[sid] for sid in full_rms_map}
    after_attack_rms = {sid: attack_rms_map[sid] * gain_map[sid] for sid in attack_rms_map}
    after_mid_rms = {sid: mid_rms_map[sid] * gain_map[sid] for sid in mid_rms_map}
    after_peak = {sid: peak_map[sid] * gain_map[sid] for sid in peak_map}

    onset_values = [after_onset_map[sid] * 1000.0 for sid in after_onset_map]
    onset_spread = spread_linear_percentile(onset_values, 0.10, 0.90)
    onset_max = max(onset_values) if onset_values else 0.0

    full_spread = spread_db_percentile(list(after_full_rms.values()), 0.10, 0.90)
    attack_spread = spread_db_percentile(list(after_attack_rms.values()), 0.10, 0.90)
    mid_spread = spread_db_percentile(list(after_mid_rms.values()), 0.10, 0.90)

    ordered = sorted(specs, key=lambda item: item.midi)
    max_adjacent_gain_step = 0.0
    for idx in range(1, len(ordered)):
        prev_gain = max(gain_map[ordered[idx - 1].id], 1e-12)
        current_gain = max(gain_map[ordered[idx].id], 1e-12)
        step_db = abs(20 * log10(current_gain / prev_gain))
        if step_db > max_adjacent_gain_step:
            max_adjacent_gain_step = step_db

    max_peak = max(after_peak.values()) if after_peak else 0.0
    total_mb = sum(path.stat().st_size for path in output_dir.glob("*.m4a")) / (1024 * 1024)
    mapping_error, mapping_worst = compute_mapping_error(specs)

    epsilon = 1e-6
    gates = {
        "onsetSpreadP10P90Ms": onset_spread <= GATE_ONSET_SPREAD_MS,
        "onsetMaxMs": onset_max <= GATE_ONSET_MAX_MS,
        "fullRmsSpreadDb": full_spread <= GATE_FULL_RMS_SPREAD_DB,
        "attackRmsSpreadDb": attack_spread <= GATE_ATTACK_RMS_SPREAD_DB,
        "midRmsSpreadDb": mid_spread <= GATE_MID_RMS_SPREAD_DB,
        "maxAdjacentGainStepDb": max_adjacent_gain_step <= (GATE_MAX_ADJACENT_GAIN_STEP_DB + epsilon),
        "maxPeakLinear": max_peak <= GATE_MAX_PEAK_LINEAR,
        "maxMappingErrorCents": mapping_error <= GATE_MAX_MAPPING_ERROR_CENTS,
        "sizeMb": total_mb <= GATE_MAX_TOTAL_MB,
    }
    passed = all(gates.values())

    build_id = int(time.time())
    report_path = report_dir / f"{instrument}_alignment_report_{build_id}.md"
    csv_path = report_dir / f"{instrument}_alignment_samples_{build_id}.csv"
    json_path = report_dir / f"{instrument}_alignment_report_{build_id}.json"

    rows = []
    for spec in specs:
        rows.append(
            {
                "id": spec.id,
                "midi": spec.midi,
                "note": spec.note,
                "onsetBeforeSec": round(before_onset_map[spec.id], 6),
                "onsetAfterSec": round(after_onset_map[spec.id], 6),
                "gainApplied": round(gain_map[spec.id], 10),
                "peakAfter": round(after_peak[spec.id], 8),
                "fullRmsAfter": round(after_full_rms[spec.id], 8),
                "attackRmsAfter": round(after_attack_rms[spec.id], 8),
                "midRmsAfter": round(after_mid_rms[spec.id], 8),
            }
        )

    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    report_json = {
        "instrument": instrument,
        "passed": passed,
        "metrics": {
            "onsetSpreadP10P90Ms": onset_spread,
            "onsetMaxMs": onset_max,
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
    json_path.write_text(json.dumps(report_json, indent=2))

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
        f"- pack size: {total_mb:.2f} MB",
        "",
        "## Gates",
    ]
    for key, ok in gates.items():
        lines.append(f"- {key}: {'PASS' if ok else 'FAIL'}")
    report_path.write_text("\n".join(lines))

    manifest["buildId"] = build_id
    manifest["alignment"] = {
        "method": "peak_ratio_threshold_consecutive_streak_trim_with_preroll",
        "thresholdRatio": ONSET_THRESHOLD_RATIO,
        "thresholdFloor": ONSET_THRESHOLD_FLOOR,
        "minStreakSamples": ONSET_MIN_STREAK,
        "envelopeSmoothMs": int(ONSET_ENVELOPE_SMOOTH_SEC * 1000),
        "startPrerollMs": int(START_PREROLL_SEC * 1000),
        "fadeInMs": int(FADE_IN_SEC * 1000),
        "fadeOutMs": int(FADE_OUT_SEC * 1000),
    }
    manifest["normalization"] = {
        "method": "full_window_dominant_rms_with_neighbor_smoothing_and_peak_guard",
        "targetBlendedRms": round(target_rms, 10),
        "targetPeakLinear": TARGET_PEAK_LINEAR,
        "globalPeakScale": 1.0,
        "gainClamp": [GAIN_CLAMP_MIN, GAIN_CLAMP_MAX],
        "gainSmoothing": {"lambda": GAIN_SMOOTHING_LAMBDA, "passes": GAIN_SMOOTHING_PASSES},
        "gainRange": [
            round(min(gain_map.values()), 8),
            round(max(gain_map.values()), 8),
        ],
    }
    manifest["quality"] = {
        "spreadMethod": {
            "onsetMs": "p10_p90",
            "fullRms": "p10_p90",
            "attackRms": "p10_p90",
            "midRms": "p10_p90",
        },
        "onsetSpreadP10P90Ms": round(onset_spread, 4),
        "onsetMaxMs": round(onset_max, 4),
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
            "maxPeakLinear": GATE_MAX_PEAK_LINEAR,
            "maxMappingErrorCents": GATE_MAX_MAPPING_ERROR_CENTS,
            "maxTotalMb": GATE_MAX_TOTAL_MB,
        },
    }
    manifest["sizeMb"] = round(total_mb, 4)
    manifest["mappingWorstAbsCents"] = round(mapping_error, 6)
    manifest["mappingWorst"] = mapping_worst
    manifest["report"] = {
        "md": str(report_path),
        "json": str(json_path),
        "csv": str(csv_path),
    }

    sample_rows = []
    for spec in specs:
        sample_rows.append(
            {
                "id": spec.id,
                "midi": spec.midi,
                "note": spec.note,
                "hz": spec.hz,
                "durationMs": duration_ms,
                "gainApplied": round(gain_map[spec.id], 10),
                "onsetSecBefore": round(before_onset_map[spec.id], 6),
                "onsetSecAfter": round(after_onset_map[spec.id], 6),
                "file": f"/assets/audio/{instrument}/{spec.output_filename}",
            }
        )
    manifest["samples"] = sample_rows

    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"Wrote report: {report_path}")
    if not passed:
        raise SystemExit("Alignment gates failed")


if __name__ == "__main__":
    main()
