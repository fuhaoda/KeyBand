#!/usr/bin/env python3
"""Download and rebuild aligned piano samples (C2..B5) for KeyBand.

This script implements:

- onset detection (peak-ratio threshold + consecutive streak)
- trim to onset with preroll (default 4ms)
- fixed-duration rendering to temp WAV
- "quantile alignment" loudness normalization (p10-p90 spread checks)
- neighbor-smoothed gain map + peak guard
- before/after reports (Markdown + JSON + CSV)
- manifest.json with build metadata and quality metrics
"""

from __future__ import annotations

import argparse
from array import array
from dataclasses import dataclass
import csv
import json
from math import exp, log, log10, log2, sqrt
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from urllib.parse import quote
from urllib.request import urlretrieve

SOURCE_BASE_URL = "https://theremin.music.uiowa.edu/sound%20files/MIS/Piano_Other/piano"

MIN_MIDI = 36  # C2
MAX_MIDI = 83  # B5

NOTE_NAMES_FLAT = ("C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B")

# Onset detection + alignment.
ONSET_THRESHOLD_RATIO = 0.08
ONSET_THRESHOLD_FLOOR = 0.003
ONSET_MIN_STREAK = 96
ONSET_ENVELOPE_SMOOTH_SEC = 0.003
START_PREROLL_SEC = 0.004  # 4ms preroll, matches tonic-ear guitar alignment.

FADE_IN_SEC = 0.001
FADE_OUT_SEC = 0.080

# Loudness windows (relative to onset).
ATTACK_ANALYSIS_SEC = 0.35
MID_WINDOW_START_SEC = 0.35
MID_WINDOW_DURATION_SEC = 0.55

# Gain normalization.
TARGET_PEAK_LINEAR = 0.90
GAIN_CLAMP_MIN = 0.60
GAIN_CLAMP_MAX = 3.00
GAIN_SMOOTHING_LAMBDA = 0.08
GAIN_SMOOTHING_PASSES = 1

# "Quantile alignment" spread metrics.
QUALITY_SPREAD_LOW_PERCENTILE = 0.10
QUALITY_SPREAD_HIGH_PERCENTILE = 0.90

# Quality gates (the "8 standards" from the plan).
GATE_ONSET_SPREAD_MS = 3.0
GATE_ONSET_MAX_MS = 8.0
GATE_FULL_RMS_SPREAD_DB = 2.0
GATE_ATTACK_RMS_SPREAD_DB = 3.2
GATE_MID_RMS_SPREAD_DB = 7.0
GATE_MAX_ADJACENT_GAIN_STEP_DB = 3.5
GATE_MAX_MAPPING_ERROR_CENTS = 10.0

# Mapping error targets (same logic as tonic-ear).
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
MAJOR_DEGREE_TO_SEMITONE = {
    1: 0,
    2: 2,
    3: 4,
    4: 5,
    5: 7,
    6: 9,
    7: 11,
}


@dataclass(frozen=True)
class SampleSpec:
    id: str
    midi: int
    note: str
    hz: float
    source_filename: str
    output_filename: str


@dataclass(frozen=True)
class SampleMetrics:
    onset_sec: float
    peak: float
    full_rms: float
    attack_rms: float
    mid_rms: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="web/assets/audio/piano", help="Output directory")
    parser.add_argument("--cache-dir", default=".cache/piano_mis_ff", help="Source cache directory")
    parser.add_argument(
        "--source-dir",
        default=None,
        help="Optional directory containing source aiff files named like Piano.ff.C2.aiff",
    )
    parser.add_argument("--report-dir", default="reports", help="Report output directory")
    parser.add_argument("--duration", type=float, default=8.0, help="Output duration in seconds")
    parser.add_argument("--sample-rate", type=int, default=44100, help="Output sample rate")
    parser.add_argument("--bitrate", default="192k", help="AAC bitrate, for example 160k/192k")
    parser.add_argument("--target-mb", type=float, default=10.0, help="Soft package-size target")
    parser.add_argument("--max-total-mb", type=float, default=20.0, help="Hard package-size cap")
    parser.add_argument("--clean", action="store_true", help="Remove output dir before build")
    parser.add_argument(
        "--refresh-sources",
        action="store_true",
        help="Force re-download of all source files before processing",
    )
    return parser.parse_args()


def require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is required but not found in PATH")


def midi_to_hz(midi: int) -> float:
    return 440.0 * (2 ** ((midi - 69) / 12))


def midi_to_note_name(midi: int) -> str:
    note_name = NOTE_NAMES_FLAT[midi % 12]
    octave = (midi // 12) - 1
    return f"{note_name}{octave}"


def build_sample_specs() -> list[SampleSpec]:
    specs: list[SampleSpec] = []
    for midi in range(MIN_MIDI, MAX_MIDI + 1):
        note = midi_to_note_name(midi)
        sample_id = f"m{midi:03d}"
        specs.append(
            SampleSpec(
                id=sample_id,
                midi=midi,
                note=note,
                hz=midi_to_hz(midi),
                source_filename=f"Piano.ff.{note}.aiff",
                output_filename=f"{sample_id}.m4a",
            )
        )
    return specs


def source_url_for_filename(filename: str) -> str:
    return f"{SOURCE_BASE_URL}/{quote(filename)}"


def download_sources(cache_dir: Path, refresh_sources: bool) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    for spec in build_sample_specs():
        source_path = cache_dir / spec.source_filename
        if refresh_sources and source_path.exists():
            source_path.unlink()
        if source_path.exists():
            continue
        source_url = source_url_for_filename(spec.source_filename)
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
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "f32le",
        "-",
    ]
    proc = subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    samples = array("f")
    samples.frombytes(proc.stdout)
    return samples


def detect_onset_seconds(samples: array, sample_rate: int) -> float:
    if not samples:
        return 0.0
    peak_abs = 0.0
    for value in samples:
        abs_value = abs(value)
        if abs_value > peak_abs:
            peak_abs = abs_value
    if peak_abs <= 0.0:
        return 0.0

    threshold = max(peak_abs * ONSET_THRESHOLD_RATIO, ONSET_THRESHOLD_FLOOR)
    alpha = 0.0
    if ONSET_ENVELOPE_SMOOTH_SEC > 0:
        alpha = exp(-1.0 / (sample_rate * ONSET_ENVELOPE_SMOOTH_SEC))
    envelope = 0.0
    streak = 0
    for index, value in enumerate(samples):
        envelope = alpha * envelope + (1.0 - alpha) * abs(value)
        if envelope >= threshold:
            streak += 1
            if streak >= ONSET_MIN_STREAK:
                onset_index = index - ONSET_MIN_STREAK + 1
                return onset_index / sample_rate
        else:
            streak = 0
    return 0.0


def _window_indices(
    sample_rate: int,
    onset_sec: float,
    start_rel_sec: float,
    duration_sec: float,
) -> tuple[int, int]:
    start_sec = max(0.0, onset_sec + start_rel_sec)
    end_sec = max(start_sec, onset_sec + start_rel_sec + max(0.0, duration_sec))
    start = int(round(start_sec * sample_rate))
    end = int(round(end_sec * sample_rate))
    return start, end


def window_rms_zero_padded(samples: array, start: int, end: int) -> float:
    if end <= start:
        return 0.0
    n_total = end - start
    if n_total <= 0:
        return 0.0
    energy = 0.0
    n_real = 0
    for index in range(start, min(end, len(samples))):
        value = samples[index]
        energy += value * value
        n_real += 1
    if n_real == 0:
        return 0.0
    # zero padding contributes 0 energy; divide by intended window length
    return sqrt(energy / n_total)


def window_peak(samples: array, start: int, end: int) -> float:
    peak = 0.0
    for index in range(start, min(end, len(samples))):
        abs_value = abs(samples[index])
        if abs_value > peak:
            peak = abs_value
    return peak


def measure_metrics(
    samples: array,
    sample_rate: int,
    onset_sec: float,
    duration: float,
) -> SampleMetrics:
    # Full window.
    full_start, full_end = _window_indices(sample_rate, onset_sec, 0.0, duration)
    peak = window_peak(samples, full_start, full_end)
    full_rms = window_rms_zero_padded(samples, full_start, full_end)

    # Attack.
    attack_duration = min(ATTACK_ANALYSIS_SEC, duration)
    att_start, att_end = _window_indices(sample_rate, onset_sec, 0.0, attack_duration)
    attack_rms = window_rms_zero_padded(samples, att_start, att_end)

    # Mid.
    mid_start_rel = min(max(0.0, MID_WINDOW_START_SEC), max(0.0, duration - 0.05))
    mid_duration = min(max(0.05, MID_WINDOW_DURATION_SEC), max(0.05, duration - mid_start_rel))
    mid_start, mid_end = _window_indices(sample_rate, onset_sec, mid_start_rel, mid_duration)
    mid_rms = window_rms_zero_padded(samples, mid_start, mid_end)

    return SampleMetrics(
        onset_sec=float(onset_sec),
        peak=float(peak),
        full_rms=float(full_rms),
        attack_rms=float(attack_rms),
        mid_rms=float(mid_rms),
    )


def render_aligned_wav(
    input_path: Path,
    output_path: Path,
    trim_start_sec: float,
    duration: float,
    sample_rate: int,
) -> None:
    trim_start_sec = max(0.0, trim_start_sec)
    fade_out_start = max(duration - FADE_OUT_SEC, 0.01)
    filter_chain = ",".join(
        [
            "aformat=channel_layouts=mono",
            f"aresample={sample_rate}",
            f"atrim=start={trim_start_sec:.6f}",
            "asetpts=PTS-STARTPTS",
            f"apad=pad_dur={duration:.6f}",
            f"atrim=end={duration:.6f}",
            f"afade=t=in:st=0:d={FADE_IN_SEC:.6f}",
            f"afade=t=out:st={fade_out_start:.6f}:d={FADE_OUT_SEC:.6f}",
        ]
    )
    ffmpeg_cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(input_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-af",
        filter_chain,
        "-c:a",
        "pcm_f32le",
        str(output_path),
    ]
    subprocess.run(ffmpeg_cmd, check=True)


def encode_final_sample(temp_wav_path: Path, output_path: Path, bitrate: str, gain: float) -> None:
    ffmpeg_cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(temp_wav_path),
        "-af",
        f"volume={gain:.8f}",
        "-c:a",
        "aac",
        "-b:a",
        bitrate,
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    subprocess.run(ffmpeg_cmd, check=True)


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    if percentile <= 0:
        return sorted_values[0]
    if percentile >= 1:
        return sorted_values[-1]
    position = percentile * (len(sorted_values) - 1)
    lower_index = int(position)
    upper_index = min(len(sorted_values) - 1, lower_index + 1)
    fraction = position - lower_index
    return sorted_values[lower_index] * (1.0 - fraction) + sorted_values[upper_index] * fraction


def _spread_db_percentile(values: list[float], low_percentile: float, high_percentile: float) -> float:
    nonzero = [value for value in values if value > 0]
    if len(nonzero) < 2:
        return 0.0
    lo = _percentile(nonzero, low_percentile)
    hi = _percentile(nonzero, high_percentile)
    if lo <= 0 or hi <= 0:
        return 0.0
    return 20.0 * log10(hi / lo)


def smooth_gain_map_by_neighbors(ordered_ids: list[str], gain_map: dict[str, float]) -> dict[str, float]:
    log_gain = {sid: log(max(gain_map.get(sid, 1.0), 1e-12)) for sid in ordered_ids}

    for _ in range(GAIN_SMOOTHING_PASSES):
        next_log_gain: dict[str, float] = {}
        for index, sid in enumerate(ordered_ids):
            neighbors: list[float] = []
            if index > 0:
                neighbors.append(log_gain[ordered_ids[index - 1]])
            if index + 1 < len(ordered_ids):
                neighbors.append(log_gain[ordered_ids[index + 1]])
            if not neighbors:
                next_log_gain[sid] = log_gain[sid]
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

    # Use median like tonic-ear guitar alignment.
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

    # Enforce adjacent gain step bound.
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

    # Peak guard (global scale if any predicted peak exceeds target).
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


def enforce_size_budget(output_dir: Path, target_mb: float, max_total_mb: float) -> tuple[int, float]:
    audio_files = sorted(output_dir.glob("*.m4a"))
    total_bytes = sum(path.stat().st_size for path in audio_files)
    total_mb = total_bytes / (1024 * 1024)

    if total_mb > max_total_mb:
        raise SystemExit(
            f"Audio package is {total_mb:.2f}MB which exceeds hard cap {max_total_mb:.2f}MB",
        )
    if total_mb > target_mb:
        print(
            f"WARNING: audio package is {total_mb:.2f}MB, above target {target_mb:.2f}MB "
            f"but within hard cap {max_total_mb:.2f}MB",
        )
    return total_bytes, total_mb


def calculate_do_frequency(gender: str, key_id: str) -> float:
    if gender == "male":
        base_do = MALE_DO_C
    elif gender == "female":
        base_do = FEMALE_DO_C
    else:
        raise ValueError(f"Unknown gender '{gender}'")
    if key_id not in KEY_OFFSETS:
        raise ValueError(f"Unknown key '{key_id}'")
    semitone_shift = KEY_OFFSETS[key_id]
    return base_do * (2 ** (semitone_shift / 12))


def compute_mapping_error_cents(target_hz: float, sample_hz: float) -> float:
    if target_hz <= 0 or sample_hz <= 0:
        return 0.0
    return 1200.0 * log2(target_hz / sample_hz)


def worst_mapping_error(specs: list[SampleSpec]) -> tuple[float, dict]:
    """Return worst absolute cents error over Male/Female × Key × degree(1..7)."""

    hz_values = [spec.hz for spec in specs]
    if not hz_values:
        return 0.0, {}
    min_hz = min(hz_values)
    max_hz = max(hz_values)

    worst_abs = 0.0
    worst: dict = {}

    # Map targetHz to nearest sampleHz.
    for gender in ("male", "female"):
        for key_id in KEY_OFFSETS:
            do_hz = calculate_do_frequency(gender=gender, key_id=key_id)
            for degree, semitone in MAJOR_DEGREE_TO_SEMITONE.items():
                target = do_hz * (2 ** (semitone / 12))
                if target < min_hz or target > max_hz:
                    continue
                nearest = min(specs, key=lambda spec: abs(compute_mapping_error_cents(target, spec.hz)))
                cents = compute_mapping_error_cents(target, nearest.hz)
                abs_cents = abs(cents)
                if abs_cents >= worst_abs:
                    worst_abs = abs_cents
                    worst = {
                        "gender": gender,
                        "key": key_id,
                        "degree": degree,
                        "targetHz": round(target, 6),
                        "sampleId": nearest.id,
                        "midi": nearest.midi,
                        "sampleHz": round(nearest.hz, 6),
                        "centsError": round(cents, 6),
                    }
    return worst_abs, worst


def _fmt_db(value: float) -> str:
    return f"{value:.3f}dB"


def _fmt_ms(value: float) -> str:
    return f"{value:.3f}ms"


def write_reports(
    report_dir: Path,
    build_id: int,
    args: argparse.Namespace,
    specs: list[SampleSpec],
    before_metrics: dict[str, SampleMetrics],
    after_metrics: dict[str, SampleMetrics],
    gain_map: dict[str, float],
    target_blended_rms: float,
    global_peak_scale: float,
    mapping_worst_abs: float,
    mapping_worst: dict,
    total_mb: float,
    quality_summary: dict,
    issues: list[str],
) -> dict[str, str]:
    report_dir.mkdir(parents=True, exist_ok=True)

    md_path = report_dir / f"piano_alignment_report_{build_id}.md"
    json_path = report_dir / f"piano_alignment_report_{build_id}.json"
    csv_path = report_dir / f"piano_alignment_samples_{build_id}.csv"

    def metric_values(metrics: dict[str, SampleMetrics], field: str) -> list[float]:
        return [getattr(m, field) for m in metrics.values()]

    def percentile_summary(values: list[float]) -> dict:
        if not values:
            return {"min": 0.0, "p10": 0.0, "p50": 0.0, "p90": 0.0, "max": 0.0}
        return {
            "min": float(min(values)),
            "p10": float(_percentile(values, 0.10)),
            "p50": float(_percentile(values, 0.50)),
            "p90": float(_percentile(values, 0.90)),
            "max": float(max(values)),
        }

    before_onsets = metric_values(before_metrics, "onset_sec")
    after_onsets = metric_values(after_metrics, "onset_sec")
    before_onset_p10 = _percentile(before_onsets, 0.10) if before_onsets else 0.0
    before_onset_p90 = _percentile(before_onsets, 0.90) if before_onsets else 0.0
    after_onset_p10 = _percentile(after_onsets, 0.10) if after_onsets else 0.0
    after_onset_p90 = _percentile(after_onsets, 0.90) if after_onsets else 0.0

    before_percentiles = {
        "onsetSec": percentile_summary(before_onsets),
        "peak": percentile_summary(metric_values(before_metrics, "peak")),
        "fullRms": percentile_summary(metric_values(before_metrics, "full_rms")),
        "attackRms": percentile_summary(metric_values(before_metrics, "attack_rms")),
        "midRms": percentile_summary(metric_values(before_metrics, "mid_rms")),
    }
    after_percentiles = {
        "onsetSec": percentile_summary(after_onsets),
        "peak": percentile_summary(metric_values(after_metrics, "peak")),
        "fullRms": percentile_summary(metric_values(after_metrics, "full_rms")),
        "attackRms": percentile_summary(metric_values(after_metrics, "attack_rms")),
        "midRms": percentile_summary(metric_values(after_metrics, "mid_rms")),
    }

    before_summary = {
        "onsetP10Ms": round(before_onset_p10 * 1000.0, 4),
        "onsetP90Ms": round(before_onset_p90 * 1000.0, 4),
        "onsetSpreadP10P90Ms": round((before_onset_p90 - before_onset_p10) * 1000.0, 4),
        "onsetMaxMs": round(max(before_onsets) * 1000.0, 4) if before_onsets else 0.0,
        "fullRmsSpreadDb": round(
            _spread_db_percentile(metric_values(before_metrics, "full_rms"), QUALITY_SPREAD_LOW_PERCENTILE, QUALITY_SPREAD_HIGH_PERCENTILE),
            4,
        ),
        "attackRmsSpreadDb": round(
            _spread_db_percentile(metric_values(before_metrics, "attack_rms"), QUALITY_SPREAD_LOW_PERCENTILE, QUALITY_SPREAD_HIGH_PERCENTILE),
            4,
        ),
        "midRmsSpreadDb": round(
            _spread_db_percentile(metric_values(before_metrics, "mid_rms"), QUALITY_SPREAD_LOW_PERCENTILE, QUALITY_SPREAD_HIGH_PERCENTILE),
            4,
        ),
        "maxPeakLinear": round(max((m.peak for m in before_metrics.values()), default=0.0), 6),
    }

    after_summary = {
        "onsetP10Ms": round(after_onset_p10 * 1000.0, 4),
        "onsetP90Ms": round(after_onset_p90 * 1000.0, 4),
        "onsetSpreadP10P90Ms": round((after_onset_p90 - after_onset_p10) * 1000.0, 4),
        "onsetMaxMs": round(max(after_onsets) * 1000.0, 4) if after_onsets else 0.0,
        "fullRmsSpreadDb": round(
            _spread_db_percentile(metric_values(after_metrics, "full_rms"), QUALITY_SPREAD_LOW_PERCENTILE, QUALITY_SPREAD_HIGH_PERCENTILE),
            4,
        ),
        "attackRmsSpreadDb": round(
            _spread_db_percentile(metric_values(after_metrics, "attack_rms"), QUALITY_SPREAD_LOW_PERCENTILE, QUALITY_SPREAD_HIGH_PERCENTILE),
            4,
        ),
        "midRmsSpreadDb": round(
            _spread_db_percentile(metric_values(after_metrics, "mid_rms"), QUALITY_SPREAD_LOW_PERCENTILE, QUALITY_SPREAD_HIGH_PERCENTILE),
            4,
        ),
        "maxPeakLinear": round(max((m.peak for m in after_metrics.values()), default=0.0), 6),
        "sizeMb": round(total_mb, 4),
        "maxAdjacentGainStepDb": round(compute_max_adjacent_gain_step_db(specs, gain_map), 4),
    }

    def top_sample_ids_by(metric_fn, limit: int = 5) -> list[str]:
        ranked = []
        for spec in specs:
            m = after_metrics.get(spec.id)
            if not m:
                continue
            ranked.append((metric_fn(spec, m), spec.id))
        ranked.sort(reverse=True)
        return [sid for _, sid in ranked[:limit]]

    def top_deviation_ids(field: str, limit: int = 5) -> list[str]:
        values = [getattr(after_metrics[sid], field) for sid in after_metrics if getattr(after_metrics[sid], field) > 0]
        if not values:
            return []
        from statistics import median

        med = median(values)
        ranked = []
        for spec in specs:
            m = after_metrics.get(spec.id)
            if not m:
                continue
            v = getattr(m, field)
            if v <= 0 or med <= 0:
                score = 0.0
            else:
                score = abs(log(v / med))
            ranked.append((score, spec.id))
        ranked.sort(reverse=True)
        return [sid for _, sid in ranked[:limit]]

    outliers = {
        "onsetLate": top_sample_ids_by(lambda _spec, m: m.onset_sec),
        "peakHigh": top_sample_ids_by(lambda _spec, m: m.peak),
        "fullRmsOdd": top_deviation_ids("full_rms"),
        "attackRmsOdd": top_deviation_ids("attack_rms"),
        "midRmsOdd": top_deviation_ids("mid_rms"),
    }

    outlier_tags: dict[str, set[str]] = {key: set(value) for key, value in outliers.items()}

    report_json = {
        "buildId": build_id,
        "instrument": "piano",
        "source": "University of Iowa MIS Piano (ff)",
        "sourceUrl": SOURCE_BASE_URL,
        "range": {"minMidi": MIN_MIDI, "maxMidi": MAX_MIDI, "minNote": midi_to_note_name(MIN_MIDI), "maxNote": midi_to_note_name(MAX_MIDI)},
        "args": {
            "duration": args.duration,
            "sampleRate": args.sample_rate,
            "bitrate": args.bitrate,
            "targetMb": args.target_mb,
            "maxTotalMb": args.max_total_mb,
        },
        "alignment": {
            "onsetDetection": {
                "thresholdRatio": ONSET_THRESHOLD_RATIO,
                "thresholdFloor": ONSET_THRESHOLD_FLOOR,
                "minStreakSamples": ONSET_MIN_STREAK,
                "envelopeSmoothMs": int(round(ONSET_ENVELOPE_SMOOTH_SEC * 1000)),
            },
            "startPrerollMs": int(round(START_PREROLL_SEC * 1000)),
            "fadeInMs": int(round(FADE_IN_SEC * 1000)),
            "fadeOutMs": int(round(FADE_OUT_SEC * 1000)),
        },
        "normalization": {
            "method": "full_window_dominant_rms_with_neighbor_smoothing_and_peak_guard",
            "targetBlendedRms": round(target_blended_rms, 10),
            "targetPeakLinear": TARGET_PEAK_LINEAR,
            "globalPeakScale": round(global_peak_scale, 10),
            "gainClamp": [GAIN_CLAMP_MIN, GAIN_CLAMP_MAX],
            "gainSmoothing": {"lambda": GAIN_SMOOTHING_LAMBDA, "passes": GAIN_SMOOTHING_PASSES},
        },
        "qualityGates": {
            "onsetSpreadP10P90Ms": GATE_ONSET_SPREAD_MS,
            "onsetMaxMs": GATE_ONSET_MAX_MS,
            "fullRmsSpreadDb": GATE_FULL_RMS_SPREAD_DB,
            "attackRmsSpreadDb": GATE_ATTACK_RMS_SPREAD_DB,
            "midRmsSpreadDb": GATE_MID_RMS_SPREAD_DB,
            "maxAdjacentGainStepDb": GATE_MAX_ADJACENT_GAIN_STEP_DB,
            "maxPeakLinear": TARGET_PEAK_LINEAR,
            "maxMappingErrorCents": GATE_MAX_MAPPING_ERROR_CENTS,
            "maxTotalMb": args.max_total_mb,
        },
        "before": before_summary,
        "after": after_summary,
        "beforePercentiles": before_percentiles,
        "afterPercentiles": after_percentiles,
        "mappingWorst": mapping_worst,
        "mappingWorstAbsCents": round(mapping_worst_abs, 6),
        "issues": issues,
        "pass": not issues,
        "quality": quality_summary,
        "outliers": outliers,
    }

    json_path.write_text(json.dumps(report_json, indent=2), encoding="utf-8")

    # CSV per sample.
    fieldnames = [
        "id",
        "midi",
        "note",
        "hz",
        "source",
        "gainApplied",
        "onsetBeforeSec",
        "onsetAfterSec",
        "peakBefore",
        "peakAfter",
        "fullRmsBefore",
        "fullRmsAfter",
        "attackRmsBefore",
        "attackRmsAfter",
        "midRmsBefore",
        "midRmsAfter",
        "outliers",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for spec in sorted(specs, key=lambda item: item.midi):
            before = before_metrics.get(spec.id)
            after = after_metrics.get(spec.id)
            tags: list[str] = []
            if spec.id in outlier_tags["onsetLate"]:
                tags.append("ONSET_LATE")
            if spec.id in outlier_tags["peakHigh"]:
                tags.append("PEAK_HIGH")
            if spec.id in outlier_tags["fullRmsOdd"]:
                tags.append("FULL_RMS_ODD")
            if spec.id in outlier_tags["attackRmsOdd"]:
                tags.append("ATTACK_RMS_ODD")
            if spec.id in outlier_tags["midRmsOdd"]:
                tags.append("MID_RMS_ODD")
            writer.writerow(
                {
                    "id": spec.id,
                    "midi": spec.midi,
                    "note": spec.note,
                    "hz": round(spec.hz, 6),
                    "source": spec.source_filename,
                    "gainApplied": round(gain_map.get(spec.id, 1.0), 10),
                    "onsetBeforeSec": round(before.onset_sec, 6) if before else "",
                    "onsetAfterSec": round(after.onset_sec, 6) if after else "",
                    "peakBefore": round(before.peak, 8) if before else "",
                    "peakAfter": round(after.peak, 8) if after else "",
                    "fullRmsBefore": round(before.full_rms, 10) if before else "",
                    "fullRmsAfter": round(after.full_rms, 10) if after else "",
                    "attackRmsBefore": round(before.attack_rms, 10) if before else "",
                    "attackRmsAfter": round(after.attack_rms, 10) if after else "",
                    "midRmsBefore": round(before.mid_rms, 10) if before else "",
                    "midRmsAfter": round(after.mid_rms, 10) if after else "",
                    "outliers": ",".join(tags),
                }
            )

    md_lines: list[str] = []
    md_lines.append(f"# KeyBand Piano Alignment Report ({build_id})")
    md_lines.append("")
    md_lines.append("## Summary")
    md_lines.append(f"- Range: {midi_to_note_name(MIN_MIDI)}..{midi_to_note_name(MAX_MIDI)} (MIDI {MIN_MIDI}..{MAX_MIDI})")
    md_lines.append(f"- Duration: {args.duration:.3f}s | Sample rate: {args.sample_rate}Hz | Bitrate: {args.bitrate}")
    md_lines.append(f"- Output size: {total_mb:.2f}MB")
    md_lines.append(f"- Pass: {'YES' if not issues else 'NO'}")
    if issues:
        md_lines.append("")
        md_lines.append("## Failures")
        for issue in issues:
            md_lines.append(f"- {issue}")
    md_lines.append("")
    md_lines.append("## Gates (after)")
    md_lines.append(f"- Onset spread p10-p90: {after_summary['onsetSpreadP10P90Ms']:.3f}ms (<= {GATE_ONSET_SPREAD_MS:.3f}ms)")
    md_lines.append(f"- Onset max: {after_summary['onsetMaxMs']:.3f}ms (<= {GATE_ONSET_MAX_MS:.3f}ms)")
    md_lines.append(f"- Full RMS spread p10-p90: {after_summary['fullRmsSpreadDb']:.3f}dB (<= {GATE_FULL_RMS_SPREAD_DB:.3f}dB)")
    md_lines.append(f"- Attack RMS spread p10-p90: {after_summary['attackRmsSpreadDb']:.3f}dB (<= {GATE_ATTACK_RMS_SPREAD_DB:.3f}dB)")
    md_lines.append(f"- Mid RMS spread p10-p90: {after_summary['midRmsSpreadDb']:.3f}dB (<= {GATE_MID_RMS_SPREAD_DB:.3f}dB)")
    md_lines.append(f"- Max adjacent gain step: {after_summary['maxAdjacentGainStepDb']:.3f}dB (<= {GATE_MAX_ADJACENT_GAIN_STEP_DB:.3f}dB)")
    md_lines.append(f"- Max peak: {after_summary['maxPeakLinear']:.6f} (<= {TARGET_PEAK_LINEAR:.2f})")
    md_lines.append(f"- Worst mapping error: {mapping_worst_abs:.6f} cents (<= {GATE_MAX_MAPPING_ERROR_CENTS:.2f})")
    md_lines.append("")
    md_lines.append("## Before vs After (headline)")
    md_lines.append(f"- Onset spread p10-p90: {_fmt_ms(before_summary['onsetSpreadP10P90Ms'])} -> {_fmt_ms(after_summary['onsetSpreadP10P90Ms'])}")
    md_lines.append(f"- Full RMS spread p10-p90: {_fmt_db(before_summary['fullRmsSpreadDb'])} -> {_fmt_db(after_summary['fullRmsSpreadDb'])}")
    md_lines.append(f"- Attack RMS spread p10-p90: {_fmt_db(before_summary['attackRmsSpreadDb'])} -> {_fmt_db(after_summary['attackRmsSpreadDb'])}")
    md_lines.append(f"- Mid RMS spread p10-p90: {_fmt_db(before_summary['midRmsSpreadDb'])} -> {_fmt_db(after_summary['midRmsSpreadDb'])}")
    md_lines.append("")
    md_lines.append("## Normalization")
    md_lines.append(f"- target blended RMS (median): {target_blended_rms:.10f}")
    md_lines.append(f"- globalPeakScale: {global_peak_scale:.10f}")
    md_lines.append(f"- gainApplied range: {min(gain_map.values()):.6f}..{max(gain_map.values()):.6f}")
    md_lines.append("")
    md_lines.append("## Outliers (after)")
    md_lines.append(f"- Onset late: {', '.join(outliers['onsetLate']) if outliers['onsetLate'] else '(none)'}")
    md_lines.append(f"- Peak high: {', '.join(outliers['peakHigh']) if outliers['peakHigh'] else '(none)'}")
    md_lines.append(f"- Full RMS odd: {', '.join(outliers['fullRmsOdd']) if outliers['fullRmsOdd'] else '(none)'}")
    md_lines.append(f"- Attack RMS odd: {', '.join(outliers['attackRmsOdd']) if outliers['attackRmsOdd'] else '(none)'}")
    md_lines.append(f"- Mid RMS odd: {', '.join(outliers['midRmsOdd']) if outliers['midRmsOdd'] else '(none)'}")
    md_lines.append("")
    md_lines.append("## Artifacts")
    md_lines.append(f"- JSON: `{json_path}`")
    md_lines.append(f"- CSV: `{csv_path}`")
    md_lines.append("")

    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    return {"md": str(md_path), "json": str(json_path), "csv": str(csv_path)}


def write_manifest(
    output_dir: Path,
    build_id: int,
    args: argparse.Namespace,
    specs: list[SampleSpec],
    before_metrics: dict[str, SampleMetrics],
    after_metrics: dict[str, SampleMetrics],
    gain_map: dict[str, float],
    target_blended_rms: float,
    global_peak_scale: float,
    total_mb: float,
    mapping_worst_abs: float,
    mapping_worst: dict,
    quality_summary: dict,
    report_paths: dict[str, str],
) -> dict:
    hz_values = [spec.hz for spec in specs]
    sample_hz_range = [min(hz_values), max(hz_values)] if hz_values else [0.0, 0.0]

    sample_entries: list[dict] = []
    for spec in sorted(specs, key=lambda item: item.midi):
        before = before_metrics.get(spec.id)
        after = after_metrics.get(spec.id)
        sample_entries.append(
            {
                "id": spec.id,
                "midi": spec.midi,
                "note": spec.note,
                "hz": round(spec.hz, 6),
                "durationMs": int(round(args.duration * 1000)),
                "gainApplied": round(gain_map.get(spec.id, 1.0), 10),
                "onsetSecBefore": round(before.onset_sec, 6) if before else None,
                "onsetSecAfter": round(after.onset_sec, 6) if after else None,
                "file": f"/assets/audio/piano/{spec.output_filename}",
            }
        )

    manifest = {
        "version": 1,
        "instrument": "piano",
        "buildId": build_id,
        "source": "University of Iowa MIS Piano (ff)",
        "sourceUrl": SOURCE_BASE_URL,
        "durationMs": int(round(args.duration * 1000)),
        "sampleRate": args.sample_rate,
        "codec": "aac",
        "bitrate": args.bitrate,
        "sampleHzRange": [round(sample_hz_range[0], 6), round(sample_hz_range[1], 6)],
        "alignment": {
            "method": "peak_ratio_threshold_consecutive_streak_trim_with_preroll",
            "thresholdRatio": ONSET_THRESHOLD_RATIO,
            "thresholdFloor": ONSET_THRESHOLD_FLOOR,
            "minStreakSamples": ONSET_MIN_STREAK,
            "envelopeSmoothMs": int(round(ONSET_ENVELOPE_SMOOTH_SEC * 1000)),
            "startPrerollMs": int(round(START_PREROLL_SEC * 1000)),
            "fadeInMs": int(round(FADE_IN_SEC * 1000)),
            "fadeOutMs": int(round(FADE_OUT_SEC * 1000)),
        },
        "normalization": {
            "method": "full_window_dominant_rms_with_neighbor_smoothing_and_peak_guard",
            "targetBlendedRms": round(target_blended_rms, 10),
            "targetPeakLinear": TARGET_PEAK_LINEAR,
            "globalPeakScale": round(global_peak_scale, 10),
            "gainClamp": [GAIN_CLAMP_MIN, GAIN_CLAMP_MAX],
            "gainSmoothing": {"lambda": GAIN_SMOOTHING_LAMBDA, "passes": GAIN_SMOOTHING_PASSES},
            "gainRange": [round(min(gain_map.values()), 8), round(max(gain_map.values()), 8)] if gain_map else [1.0, 1.0],
        },
        "quality": quality_summary,
        "sizeMb": round(total_mb, 4),
        "mappingWorstAbsCents": round(mapping_worst_abs, 6),
        "mappingWorst": mapping_worst,
        "report": report_paths,
        "sampleCount": len(sample_entries),
        "targetFrequencyCount": len(KEY_OFFSETS) * 2 * len(MAJOR_DEGREE_TO_SEMITONE),
        "samples": sample_entries,
    }

    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    args = parse_args()
    require_ffmpeg()

    output_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir)
    report_dir = Path(args.report_dir)
    source_dir = Path(args.source_dir).expanduser().resolve() if args.source_dir else None

    specs = build_sample_specs()

    if args.clean and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if source_dir is None:
        download_sources(cache_dir, refresh_sources=args.refresh_sources)
        source_root = cache_dir
    else:
        if not source_dir.exists():
            raise SystemExit(f"--source-dir does not exist: {source_dir}")
        source_root = source_dir

    build_id = int(time.time())

    before_metrics: dict[str, SampleMetrics] = {}
    after_metrics: dict[str, SampleMetrics] = {}

    # Temp aligned wavs before applying any gain.
    with tempfile.TemporaryDirectory(prefix="keyband_piano_") as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        temp_wavs: dict[str, Path] = {}

        # 1) Detect onsets on raw sources and render aligned fixed-duration WAVs.
        for spec in specs:
            source_path = source_root / spec.source_filename
            if not source_path.exists():
                raise SystemExit(f"Missing source file: {source_path}")

            decoded_source = decode_mono_float_samples(source_path, sample_rate=args.sample_rate)
            onset_before = detect_onset_seconds(decoded_source, sample_rate=args.sample_rate)
            before_metrics[spec.id] = measure_metrics(
                decoded_source,
                sample_rate=args.sample_rate,
                onset_sec=onset_before,
                duration=args.duration,
            )

            trim_start = max(0.0, onset_before - START_PREROLL_SEC)
            temp_wav_path = temp_dir / f"{spec.id}.wav"
            render_aligned_wav(
                input_path=source_path,
                output_path=temp_wav_path,
                trim_start_sec=trim_start,
                duration=args.duration,
                sample_rate=args.sample_rate,
            )
            temp_wavs[spec.id] = temp_wav_path

        # 2) Measure aligned wav metrics and compute gain map (quantile-aligned).
        peak_map: dict[str, float] = {}
        full_rms_map: dict[str, float] = {}
        attack_rms_map: dict[str, float] = {}
        mid_rms_map: dict[str, float] = {}
        aligned_onset_map: dict[str, float] = {}

        for spec in specs:
            decoded = decode_mono_float_samples(temp_wavs[spec.id], sample_rate=args.sample_rate)
            onset = detect_onset_seconds(decoded, sample_rate=args.sample_rate)
            aligned_onset_map[spec.id] = onset
            metrics = measure_metrics(decoded, sample_rate=args.sample_rate, onset_sec=onset, duration=args.duration)
            peak_map[spec.id] = metrics.peak
            full_rms_map[spec.id] = metrics.full_rms
            attack_rms_map[spec.id] = metrics.attack_rms
            mid_rms_map[spec.id] = metrics.mid_rms

        gain_map, target_blended_rms, global_peak_scale = compute_gain_map_from_blended_rms(
            specs=specs,
            peak_map=peak_map,
            full_rms_map=full_rms_map,
            attack_rms_map=attack_rms_map,
            mid_rms_map=mid_rms_map,
        )

        # 3) Encode final AAC samples with gain applied.
        for spec in specs:
            encode_final_sample(
                temp_wav_path=temp_wavs[spec.id],
                output_path=output_dir / spec.output_filename,
                bitrate=args.bitrate,
                gain=gain_map.get(spec.id, 1.0),
            )

    # 4) Measure after metrics from output m4a.
    for spec in specs:
        output_path = output_dir / spec.output_filename
        decoded = decode_mono_float_samples(output_path, sample_rate=args.sample_rate)
        onset_after = detect_onset_seconds(decoded, sample_rate=args.sample_rate)
        after_metrics[spec.id] = measure_metrics(
            decoded,
            sample_rate=args.sample_rate,
            onset_sec=onset_after,
            duration=args.duration,
        )

    total_bytes, total_mb = enforce_size_budget(output_dir, target_mb=args.target_mb, max_total_mb=args.max_total_mb)
    _ = total_bytes

    mapping_worst_abs, mapping_worst = worst_mapping_error(specs)

    # 5) Compute quality summaries + gates.
    onset_after_values = [m.onset_sec for m in after_metrics.values()]
    onset_p10 = _percentile(onset_after_values, QUALITY_SPREAD_LOW_PERCENTILE) if onset_after_values else 0.0
    onset_p90 = _percentile(onset_after_values, QUALITY_SPREAD_HIGH_PERCENTILE) if onset_after_values else 0.0
    onset_spread_ms = (onset_p90 - onset_p10) * 1000.0
    onset_max_ms = (max(onset_after_values) * 1000.0) if onset_after_values else 0.0

    full_spread_db = _spread_db_percentile(
        [m.full_rms for m in after_metrics.values()],
        QUALITY_SPREAD_LOW_PERCENTILE,
        QUALITY_SPREAD_HIGH_PERCENTILE,
    )
    attack_spread_db = _spread_db_percentile(
        [m.attack_rms for m in after_metrics.values()],
        QUALITY_SPREAD_LOW_PERCENTILE,
        QUALITY_SPREAD_HIGH_PERCENTILE,
    )
    mid_spread_db = _spread_db_percentile(
        [m.mid_rms for m in after_metrics.values()],
        QUALITY_SPREAD_LOW_PERCENTILE,
        QUALITY_SPREAD_HIGH_PERCENTILE,
    )
    max_adjacent_gain_step_db = compute_max_adjacent_gain_step_db(specs, gain_map)
    max_peak_after = max((m.peak for m in after_metrics.values()), default=0.0)

    quality_summary = {
        "spreadMethod": {
            "onsetSec": f"p{int(QUALITY_SPREAD_LOW_PERCENTILE * 100)}_p{int(QUALITY_SPREAD_HIGH_PERCENTILE * 100)}",
            "fullRms": f"p{int(QUALITY_SPREAD_LOW_PERCENTILE * 100)}_p{int(QUALITY_SPREAD_HIGH_PERCENTILE * 100)}",
            "attackRms": f"p{int(QUALITY_SPREAD_LOW_PERCENTILE * 100)}_p{int(QUALITY_SPREAD_HIGH_PERCENTILE * 100)}",
            "midRms": f"p{int(QUALITY_SPREAD_LOW_PERCENTILE * 100)}_p{int(QUALITY_SPREAD_HIGH_PERCENTILE * 100)}",
        },
        "onsetSpreadP10P90Ms": round(onset_spread_ms, 4),
        "onsetMaxMs": round(onset_max_ms, 4),
        "fullRmsSpreadDb": round(full_spread_db, 4),
        "attackRmsSpreadDb": round(attack_spread_db, 4),
        "midRmsSpreadDb": round(mid_spread_db, 4),
        "maxAdjacentGainStepDb": round(max_adjacent_gain_step_db, 4),
        "maxPeakLinear": round(max_peak_after, 6),
        "thresholds": {
            "onsetSpreadP10P90Ms": GATE_ONSET_SPREAD_MS,
            "onsetMaxMs": GATE_ONSET_MAX_MS,
            "fullRmsSpreadDb": GATE_FULL_RMS_SPREAD_DB,
            "attackRmsSpreadDb": GATE_ATTACK_RMS_SPREAD_DB,
            "midRmsSpreadDb": GATE_MID_RMS_SPREAD_DB,
            "maxAdjacentGainStepDb": GATE_MAX_ADJACENT_GAIN_STEP_DB,
            "maxPeakLinear": TARGET_PEAK_LINEAR,
            "maxMappingErrorCents": GATE_MAX_MAPPING_ERROR_CENTS,
            "maxTotalMb": args.max_total_mb,
        },
    }

    issues: list[str] = []
    if onset_spread_ms > GATE_ONSET_SPREAD_MS + 1e-6:
        issues.append(f"onset spread p10-p90 {onset_spread_ms:.3f}ms > {GATE_ONSET_SPREAD_MS:.3f}ms")
    if onset_max_ms > GATE_ONSET_MAX_MS + 1e-6:
        issues.append(f"onset max {onset_max_ms:.3f}ms > {GATE_ONSET_MAX_MS:.3f}ms")
    if full_spread_db > GATE_FULL_RMS_SPREAD_DB + 1e-6:
        issues.append(f"full RMS spread {full_spread_db:.2f}dB > {GATE_FULL_RMS_SPREAD_DB:.2f}dB")
    if attack_spread_db > GATE_ATTACK_RMS_SPREAD_DB + 1e-6:
        issues.append(f"attack RMS spread {attack_spread_db:.2f}dB > {GATE_ATTACK_RMS_SPREAD_DB:.2f}dB")
    if mid_spread_db > GATE_MID_RMS_SPREAD_DB + 1e-6:
        issues.append(f"mid RMS spread {mid_spread_db:.2f}dB > {GATE_MID_RMS_SPREAD_DB:.2f}dB")
    if max_adjacent_gain_step_db > GATE_MAX_ADJACENT_GAIN_STEP_DB + 1e-6:
        issues.append(
            f"adjacent gain step {max_adjacent_gain_step_db:.2f}dB > {GATE_MAX_ADJACENT_GAIN_STEP_DB:.2f}dB",
        )
    if max_peak_after > TARGET_PEAK_LINEAR + 1e-6:
        issues.append(f"max peak {max_peak_after:.6f} > {TARGET_PEAK_LINEAR:.2f}")
    if mapping_worst_abs > GATE_MAX_MAPPING_ERROR_CENTS + 1e-9:
        issues.append(f"mapping worst {mapping_worst_abs:.6f} cents > {GATE_MAX_MAPPING_ERROR_CENTS:.2f} cents")
    if total_mb > args.max_total_mb + 1e-9:
        issues.append(f"package size {total_mb:.2f}MB > {args.max_total_mb:.2f}MB")

    report_paths = write_reports(
        report_dir=report_dir,
        build_id=build_id,
        args=args,
        specs=specs,
        before_metrics=before_metrics,
        after_metrics=after_metrics,
        gain_map=gain_map,
        target_blended_rms=target_blended_rms,
        global_peak_scale=global_peak_scale,
        mapping_worst_abs=mapping_worst_abs,
        mapping_worst=mapping_worst,
        total_mb=total_mb,
        quality_summary=quality_summary,
        issues=issues,
    )

    _ = write_manifest(
        output_dir=output_dir,
        build_id=build_id,
        args=args,
        specs=specs,
        before_metrics=before_metrics,
        after_metrics=after_metrics,
        gain_map=gain_map,
        target_blended_rms=target_blended_rms,
        global_peak_scale=global_peak_scale,
        total_mb=total_mb,
        mapping_worst_abs=mapping_worst_abs,
        mapping_worst=mapping_worst,
        quality_summary=quality_summary,
        report_paths=report_paths,
    )

    print(
        "Built piano samples:",
        f"{len(specs)} files ({midi_to_note_name(MIN_MIDI)}..{midi_to_note_name(MAX_MIDI)})",
        f"total {total_mb:.2f}MB, max_peak={max_peak_after:.6f},",
        f"onset_spread_p10_p90={onset_spread_ms:.3f}ms, onset_max={onset_max_ms:.3f}ms",
    )
    print("Report:", report_paths.get("md"))

    if issues:
        raise SystemExit("Piano alignment quality gate failed: " + "; ".join(issues))


if __name__ == "__main__":
    main()
