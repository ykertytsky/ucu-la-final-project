#!/usr/bin/env python3
"""
Batch evaluation driver mirroring pipeline_v1.ipynb (Hankel + truncated SVD denoising).

For each ``*.wav`` under ``data/input/``, runs the same processing chain as the notebook,
writes optional denoised audio under ``data/output/``, and saves timing + scalar metrics
to CSV and JSON under ``data/output/evaluation/``.

Usage (from repo root):

    python run_pipeline_batch.py
    python run_pipeline_batch.py --input-dir data/input --no-audio
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import scipy.linalg as la
import librosa
import soundfile as sf
from scipy.signal import butter, filtfilt


def define_name(recording_name: str, window_length: int, k_rank: int, is_filtered: bool = False) -> str:
    if is_filtered:
        return f"{recording_name}-wl{window_length}-k{k_rank}-denoised-filtered.wav"
    return f"{recording_name}-wl{window_length}-k{k_rank}-denoised.wav"


def compute_L(signal: np.ndarray, sr: int) -> int:
    N = len(signal)
    L_max_allowed = N // 2
    L_target = int(sr * 0.1)
    if N < sr * 0.5:
        L_target = int(sr * 0.05)
    return min(L_target, L_max_allowed)


def k_from_energy(S: np.ndarray, threshold: float = 0.95) -> int:
    sv_energy = S**2
    total = sv_energy.sum()
    cumulative = np.cumsum(sv_energy) / total
    k = int(np.searchsorted(cumulative, threshold) + 1)
    return int(np.clip(k, 1, len(S)))


def build_hankel(x: np.ndarray, L: int) -> np.ndarray:
    N = len(x)
    K = N - L + 1
    if K < 1:
        raise ValueError(f"Window length L={L} exceeds signal length N={N}.")
    shape = (L, K)
    strides = (x.strides[0], x.strides[0])
    return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)


def hankel_to_signal(Hk: np.ndarray) -> np.ndarray:
    L, K = Hk.shape
    N = L + K - 1
    out = np.zeros(N)
    counts = np.zeros(N)
    for i in range(L):
        out[i : i + K] += Hk[i, :]
        counts[i : i + K] += 1
    return out / counts


def low_rank_approx(U: np.ndarray, s: np.ndarray, Vt: np.ndarray, k: int) -> np.ndarray:
    return (U[:, :k] * s[:k]) @ Vt[:k, :]


def highpass(signal: np.ndarray, sr: int, cutoff: float = 2000) -> np.ndarray:
    b, a = butter(4, cutoff / (sr / 2), btype="high")
    return filtfilt(b, a, signal)


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))


def snr_db(signal: np.ndarray, noise: np.ndarray) -> float:
    ps = np.mean(signal.astype(np.float64) ** 2)
    pn = np.mean(noise.astype(np.float64) ** 2)
    if pn == 0:
        return float("inf")
    return float(10 * np.log10(ps / pn))


def snr_against_reference(estimate: np.ndarray, reference: np.ndarray) -> float:
    """SNR(estimate | reference) = 10*log10(P(reference) / P(reference-estimate))."""
    ref = reference.astype(np.float64)
    err = ref - estimate.astype(np.float64)
    ps = np.mean(ref**2)
    pe = np.mean(err**2)
    if pe == 0:
        return float("inf")
    return float(10 * np.log10(ps / pe))


def spectral_centroid(x: np.ndarray, sr: int) -> float:
    freqs = np.fft.rfftfreq(len(x), d=1 / sr)
    power = np.abs(np.fft.rfft(x.astype(np.float64))) ** 2
    return float(np.sum(freqs * power) / np.sum(power))


def _json_sanitize(x: Any) -> Any:
    if isinstance(x, float):
        if math.isinf(x):
            return "inf"
        if math.isnan(x):
            return None
        return x
    if isinstance(x, dict):
        return {k: _json_sanitize(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_json_sanitize(v) for v in x]
    return x


def run_one_wav(
    wav_path: Path,
    output_dir: Path,
    energy_threshold: float,
    clean_dir: Path | None,
    save_audio: bool,
    verbose: bool,
) -> dict[str, Any]:
    recording_name = wav_path.stem
    row: dict[str, Any] = {
        "recording": recording_name,
        "input_path": str(wav_path.resolve()),
        "error": None,
    }
    t_wall0 = time.perf_counter()

    try:
        t0 = time.perf_counter()
        signal, sr = librosa.load(str(wav_path), sr=None, mono=True)
        row["time_load_s"] = time.perf_counter() - t0
        row["sample_rate_hz"] = int(sr)
        row["duration_s"] = float(len(signal) / sr)
        row["num_samples"] = int(len(signal))

        t0 = time.perf_counter()
        L = compute_L(signal, int(sr))
        row["window_L"] = int(L)
        H = build_hankel(signal, L)
        row["time_hankel_s"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        H_copy = np.array(H, dtype=np.float64)
        try:
            import torch

            use_cuda = torch.cuda.is_available()
        except ImportError:
            torch = None  # type: ignore[assignment]
            use_cuda = False

        if use_cuda:
            dev = torch.device("cuda")
            H_t = torch.as_tensor(H_copy, device=dev, dtype=torch.float64)
            U_t, s_t, Vh_t = torch.linalg.svd(H_t, full_matrices=False)
            U = U_t.detach().cpu().numpy()
            s = s_t.detach().cpu().numpy()
            Vt = Vh_t.detach().cpu().numpy()
            row["svd_device"] = "cuda"
        else:
            U, s, Vt = la.svd(H_copy, full_matrices=False)
            row["svd_device"] = "cpu"
        row["time_svd_s"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        k_rank = k_from_energy(s, threshold=energy_threshold)
        row["k_rank"] = int(k_rank)
        row["num_singular_values"] = int(len(s))

        Hk = low_rank_approx(U, s, Vt, k_rank)
        denoised = hankel_to_signal(Hk)
        denoised = np.clip(denoised, -1.0, 1.0).astype(np.float32)

        hf_original = highpass(signal, sr, cutoff=1500)
        hf_denoised = highpass(denoised, sr, cutoff=1500)
        alpha = 0.7
        output = denoised + alpha * (hf_original - hf_denoised)
        row["time_reconstruct_s"] = time.perf_counter() - t0

        if save_audio:
            t0 = time.perf_counter()
            out_name = define_name(recording_name, L, k_rank, is_filtered=True)
            out_path = output_dir / out_name
            out_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(out_path), output, sr)
            row["output_wav"] = str(out_path.resolve())
            row["time_save_audio_s"] = time.perf_counter() - t0
        else:
            row["output_wav"] = None
            row["time_save_audio_s"] = 0.0

        removed = signal - denoised
        kept_frac = rms(denoised) / rms(signal) * 100 if rms(signal) > 0 else 0.0
        removed_energy_pct = (rms(removed) ** 2) / (rms(signal) ** 2) * 100 if rms(signal) > 0 else 0.0
        snr_kept_vs_removed = snr_db(denoised, removed)

        sc_original = spectral_centroid(signal, sr)
        sc_denoised = spectral_centroid(denoised, sr)
        sc_shift_hz = sc_denoised - sc_original

        total_energy = (s**2).sum()
        kept_sv_energy_pct = (s[:k_rank] ** 2).sum() / total_energy * 100

        row["energy_threshold"] = float(energy_threshold)
        row["kept_sv_energy_pct"] = float(kept_sv_energy_pct)
        row["rms_amplitude_kept_pct"] = float(kept_frac)
        row["removed_signal_energy_pct"] = float(removed_energy_pct)
        row["snr_kept_vs_removed_db"] = float(snr_kept_vs_removed)
        row["spectral_centroid_original_hz"] = float(sc_original)
        row["spectral_centroid_denoised_hz"] = float(sc_denoised)
        row["spectral_centroid_shift_hz"] = float(sc_shift_hz)

        # Optional true SNR metrics when clean reference is available.
        row["clean_ref_path"] = None
        row["input_snr_db"] = None
        row["output_snr_db"] = None
        row["snr_improvement_db"] = None
        if clean_dir is not None:
            clean_path = clean_dir / f"{recording_name}.wav"
            if clean_path.exists():
                clean, _ = librosa.load(str(clean_path), sr=sr, mono=True)
                n = min(len(clean), len(signal), len(output))
                if n > 0:
                    clean = clean[:n]
                    noisy = signal[:n]
                    enhanced = output[:n]
                    in_snr = snr_against_reference(noisy, clean)
                    out_snr = snr_against_reference(enhanced, clean)
                    row["clean_ref_path"] = str(clean_path.resolve())
                    row["input_snr_db"] = float(in_snr)
                    row["output_snr_db"] = float(out_snr)
                    row["snr_improvement_db"] = float(out_snr - in_snr)

    except Exception as e:  # noqa: BLE001 — batch runner should record failures
        row["error"] = f"{type(e).__name__}: {e}"

    row["time_total_s"] = time.perf_counter() - t_wall0

    if verbose and row["error"] is None:
        print(
            f"[ok] {recording_name}  "
            f"k={row['k_rank']}  SNR={row['snr_kept_vs_removed_db']:.1f} dB  "
            f"total={row['time_total_s']:.2f}s"
        )
    elif verbose and row["error"]:
        print(f"[fail] {recording_name}  {row['error']}")

    return row


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def main() -> None:
    root = _repo_root()
    parser = argparse.ArgumentParser(description="Batch-run pipeline_v1 logic on all input WAV files.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=root / "data/input",
        help="Directory containing .wav recordings (default: <repo>/data/input)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "data/output",
        help="Directory for denoised WAV files (default: <repo>/data/output)",
    )
    parser.add_argument(
        "--eval-dir",
        type=Path,
        default=None,
        help="Directory for CSV/JSON metrics (default: <output-dir>/evaluation)",
    )
    parser.add_argument(
        "--energy-threshold",
        type=float,
        default=0.95,
        help="Cumulative singular-value energy threshold for choosing k (default: 0.95)",
    )
    parser.add_argument(
        "--clean-dir",
        type=Path,
        default=None,
        help="Optional directory with clean references named like input stems for true SNR improvement.",
    )
    parser.add_argument("--no-audio", action="store_true", help="Do not write denoised WAV files")
    parser.add_argument("-q", "--quiet", action="store_true", help="Less console output")
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    clean_dir = args.clean_dir.resolve() if args.clean_dir is not None else None
    eval_dir = (args.eval_dir or (output_dir / "evaluation")).resolve()
    eval_dir.mkdir(parents=True, exist_ok=True)

    wav_files = sorted(input_dir.glob("*.wav"))
    if not wav_files:
        print(f"No .wav files found in {input_dir}")
        return

    rows: list[dict[str, Any]] = []
    for wav in wav_files:
        rows.append(
            run_one_wav(
                wav,
                output_dir=output_dir,
                energy_threshold=args.energy_threshold,
                clean_dir=clean_dir,
                save_audio=not args.no_audio,
                verbose=not args.quiet,
            )
        )

    csv_path = eval_dir / "pipeline_eval_summary.csv"
    json_path = eval_dir / "pipeline_eval_summary.json"

    fieldnames = list(rows[0].keys())
    for r in rows[1:]:
        for k in r:
            if k not in fieldnames:
                fieldnames.append(k)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            flat = {}
            for k, v in r.items():
                flat[k] = "" if v is None else v
            writer.writerow(flat)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(_json_sanitize(rows), f, indent=2)

    if not args.quiet:
        print(f"Wrote {csv_path}")
        print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
