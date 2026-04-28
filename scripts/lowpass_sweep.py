"""
Compare a Butterworth low-pass filter against the SVD-on-Hankel pipeline
on the same (noisy, clean) pair. Sweeps cutoff frequency and filter order.

Reference SNR convention matches scripts/snr_sweep.py:
    SNR = 10 log10( power(clean) / power(g*x - clean) )
where g is the best-gain alignment of x onto clean.
"""

from __future__ import annotations

import json
from pathlib import Path

import librosa
import numpy as np
import scipy.signal as sp


CLEAN = "data/clean/recording_011_male.wav"
NOISY = "data/input/recording_011_male.wav"


def load_mono(path: str) -> tuple[np.ndarray, int]:
    x, sr = librosa.load(path, sr=None, mono=True)
    return x.astype(np.float64), sr


def best_gain(ref: np.ndarray, x: np.ndarray) -> float:
    denom = float(np.dot(x, x))
    return float(np.dot(ref, x) / denom) if denom > 0 else 1.0


def snr_db(s: np.ndarray, e: np.ndarray) -> float:
    ps = float(np.mean(s**2))
    pe = float(np.mean(e**2))
    return float("inf") if pe == 0 else 10.0 * np.log10(ps / pe)


def evaluate_against_clean(clean: np.ndarray, x: np.ndarray) -> float:
    g = best_gain(clean, x)
    return snr_db(clean, g * x - clean)


def lowpass(x: np.ndarray, sr: int, cutoff_hz: float, order: int) -> np.ndarray:
    sos = sp.butter(order, cutoff_hz, btype="low", fs=sr, output="sos")
    return sp.sosfiltfilt(sos, x)


def highpass(x: np.ndarray, sr: int, cutoff_hz: float, order: int) -> np.ndarray:
    sos = sp.butter(order, cutoff_hz, btype="high", fs=sr, output="sos")
    return sp.sosfiltfilt(sos, x)


def bandpass(
    x: np.ndarray, sr: int, low_hz: float, high_hz: float, order: int
) -> np.ndarray:
    sos = sp.butter(order, [low_hz, high_hz], btype="band", fs=sr, output="sos")
    return sp.sosfiltfilt(sos, x)


def main() -> None:
    clean, sr = load_mono(CLEAN)
    noisy, sr2 = load_mono(NOISY)
    assert sr == sr2
    N = min(len(clean), len(noisy))
    clean, noisy = clean[:N], noisy[:N]

    snr_in = evaluate_against_clean(clean, noisy)
    print(f"sr={sr}  N={N}")
    print(f"input SNR (noisy vs clean) : {snr_in:+.2f} dB\n")

    nyq = sr / 2
    cutoffs = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 5000, 6000, 7000]
    orders = [2, 4, 6, 8]

    print("=== Butterworth LOW-PASS sweep ===")
    print(f"{'cutoff_Hz':>10}  {'order':>5}  {'SNR_out_dB':>10}  {'Δ_dB':>8}")
    lp_rows = []
    for c in cutoffs:
        if c >= nyq:
            continue
        for o in orders:
            y = lowpass(noisy, sr, c, o)
            s_out = evaluate_against_clean(clean, y)
            d = s_out - snr_in
            lp_rows.append({"type": "lowpass", "cutoff_hz": c, "order": o,
                            "snr_out_db": s_out, "delta_db": d})
            print(f"{c:>10}  {o:>5}  {s_out:+10.2f}  {d:+8.2f}")

    print("\n=== Butterworth HIGH-PASS sweep "
          "(residual peaked near 49 Hz, so try removing low rumble) ===")
    print(f"{'cutoff_Hz':>10}  {'order':>5}  {'SNR_out_dB':>10}  {'Δ_dB':>8}")
    hp_rows = []
    for c in [40, 60, 80, 100, 150, 200, 300]:
        for o in [2, 4, 6]:
            y = highpass(noisy, sr, c, o)
            s_out = evaluate_against_clean(clean, y)
            d = s_out - snr_in
            hp_rows.append({"type": "highpass", "cutoff_hz": c, "order": o,
                            "snr_out_db": s_out, "delta_db": d})
            print(f"{c:>10}  {o:>5}  {s_out:+10.2f}  {d:+8.2f}")

    print("\n=== Butterworth BAND-PASS sweep "
          "(speech band, low cutoff = 80 Hz) ===")
    print(f"{'low_Hz':>7}  {'high_Hz':>8}  {'order':>5}  {'SNR_out_dB':>10}  {'Δ_dB':>8}")
    bp_rows = []
    for low in [60, 80, 120, 200]:
        for high in [3000, 3400, 4000, 5000, 6000, 7000]:
            if high >= nyq or low >= high:
                continue
            for o in [2, 4]:
                y = bandpass(noisy, sr, low, high, o)
                s_out = evaluate_against_clean(clean, y)
                d = s_out - snr_in
                bp_rows.append({"type": "bandpass", "low_hz": low, "high_hz": high,
                                "order": o, "snr_out_db": s_out, "delta_db": d})
                print(f"{low:>7}  {high:>8}  {o:>5}  {s_out:+10.2f}  {d:+8.2f}")

    print("\n=== Top 10 across all filters ===")
    all_rows = lp_rows + hp_rows + bp_rows
    all_rows.sort(key=lambda r: r["delta_db"], reverse=True)
    for r in all_rows[:10]:
        print(r)

    out_dir = Path("data/output/evaluation")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "filter_sweep_recording_011_male.json"
    out_path.write_text(json.dumps(
        {"recording": "recording_011_male", "input_snr_db": snr_in,
         "results": all_rows}, indent=2))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
