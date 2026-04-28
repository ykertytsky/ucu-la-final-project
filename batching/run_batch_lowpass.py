#!/usr/bin/env python3
"""
Batch: load clean WAVs → AWGN → Butterworth low-pass → SNR vs clean → CSV/JSON report.

Usage:

    python run_batch_lowpass.py
    python run_batch_lowpass.py --target-snr-db 0 --cutoff-hz 3500 --no-audio
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np

from batch_noise_eval_common import (
    butter_lowpass,
    load_wav_mono,
    make_awgn,
    repo_root,
    save_wav,
    snr_metrics_vs_clean,
    write_eval_reports,
)


def main() -> None:
    root = repo_root()
    p = argparse.ArgumentParser(description="AWGN + low-pass batch on data/clean")
    p.add_argument("--input-dir", type=Path, default=root / "data/clean")
    p.add_argument("--output-dir", type=Path, default=root / "data/output/batch_lowpass_awgn")
    p.add_argument("--eval-dir", type=Path, default=root / "data/output/evaluation")
    p.add_argument("--target-snr-db", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cutoff-hz", type=float, default=3000.0)
    p.add_argument("--order", type=int, default=4)
    p.add_argument("--no-audio", action="store_true")
    p.add_argument("-q", "--quiet", action="store_true")
    args = p.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    eval_dir = args.eval_dir.resolve()
    wavs = sorted(input_dir.glob("*.wav"))
    if not wavs:
        print(f"No .wav files in {input_dir}")
        return

    rows: list[dict[str, Any]] = []
    for i, wav in enumerate(wavs):
        row: dict[str, Any] = {
            "method": "lowpass",
            "recording": wav.stem,
            "input_path": str(wav),
            "cutoff_hz": args.cutoff_hz,
            "order": args.order,
            "error": None,
        }
        t_wall = time.perf_counter()
        try:
            clean, sr = load_wav_mono(wav)
            nyq = sr / 2.0
            if args.cutoff_hz >= nyq:
                raise ValueError(f"cutoff_hz {args.cutoff_hz} must be < Nyquist {nyq}")
            rng = np.random.default_rng(args.seed + i * 1_000_003)
            noisy = clean + make_awgn(clean, args.target_snr_db, rng)
            t0 = time.perf_counter()
            denoised = butter_lowpass(noisy, sr, args.cutoff_hz, args.order)
            row["time_filter_s"] = time.perf_counter() - t0
            in_snr, out_snr, delta = snr_metrics_vs_clean(clean, noisy, denoised)
            row["input_snr_db"] = in_snr
            row["output_snr_db"] = out_snr
            row["snr_improvement_db"] = delta
            row["sample_rate_hz"] = sr
            row["target_snr_db"] = float(args.target_snr_db)
            if not args.no_audio:
                noisy_path = output_dir / f"{wav.stem}-snr{args.target_snr_db:g}-noisy.wav"
                out_path = output_dir / f"{wav.stem}-snr{args.target_snr_db:g}-lp{args.cutoff_hz:g}Hz-o{args.order}-denoised.wav"
                save_wav(noisy_path, noisy, sr)
                save_wav(out_path, denoised, sr)
                row["output_noisy_wav"] = str(noisy_path)
                row["output_denoised_wav"] = str(out_path)
            else:
                row["output_noisy_wav"] = None
                row["output_denoised_wav"] = None
        except Exception as e:  # noqa: BLE001
            row["error"] = f"{type(e).__name__}: {e}"
        row["time_total_s"] = time.perf_counter() - t_wall
        rows.append(row)
        if not args.quiet:
            if row["error"]:
                print(f"[fail] {wav.stem}  {row['error']}")
            else:
                print(
                    f"[ok] {wav.stem}  in={row['input_snr_db']:.2f} dB  "
                    f"out={row['output_snr_db']:.2f} dB  Δ={row['snr_improvement_db']:+.2f} dB"
                )

    csv_p, json_p = write_eval_reports(rows, eval_dir, "batch_lowpass_awgn_eval")
    if not args.quiet:
        print(f"Wrote {csv_p}\nWrote {json_p}")


if __name__ == "__main__":
    main()
