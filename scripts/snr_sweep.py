"""
Sweep (L, k) for the SVD-on-Hankel pipeline against a clean reference.

For each rank r, the contribution to the reconstructed 1-D signal equals
the linear convolution of U[:, r] and V[r, :] (because anti-diagonals of
u v^T sum to (u * v)). We accumulate these across r once per L, so the
SNR for every k can be evaluated without rebuilding H_k.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import librosa
import numpy as np
import scipy.linalg as la
import scipy.signal as sp


CLEAN = "data/clean/recording_011_male.wav"
NOISY = "data/input/recording_011_male.wav"


def load_mono(path: str) -> tuple[np.ndarray, int]:
    x, sr = librosa.load(path, sr=None, mono=True)
    return x.astype(np.float64), sr


def build_hankel_view(x: np.ndarray, L: int) -> np.ndarray:
    K = len(x) - L + 1
    return np.lib.stride_tricks.as_strided(
        x, shape=(L, K), strides=(x.strides[0], x.strides[0])
    )


def antidiag_counts(L: int, K: int) -> np.ndarray:
    counts = np.zeros(L + K - 1)
    for i in range(L):
        counts[i : i + K] += 1
    return counts


def best_gain(ref: np.ndarray, x: np.ndarray) -> float:
    denom = float(np.dot(x, x))
    return float(np.dot(ref, x) / denom) if denom > 0 else 1.0


def snr_db(signal: np.ndarray, error: np.ndarray) -> float:
    ps = float(np.mean(signal**2))
    pe = float(np.mean(error**2))
    if pe == 0.0:
        return float("inf")
    return 10.0 * np.log10(ps / pe)


def evaluate_against_clean(clean: np.ndarray, x: np.ndarray) -> float:
    g = best_gain(clean, x)
    return snr_db(clean, g * x - clean)


def sweep_for_L(noisy: np.ndarray, clean: np.ndarray, L: int, k_grid: list[int]):
    H = build_hankel_view(noisy, L)
    K = H.shape[1]
    H_dense = np.asarray(H, dtype=np.float64)

    t0 = time.time()
    U, s, Vt = la.svd(H_dense, full_matrices=False)
    t_svd = time.time() - t0

    R = max(k_grid)
    counts = antidiag_counts(L, K)

    accum = np.zeros(L + K - 1)
    out_at_k: dict[int, np.ndarray] = {}
    k_set = set(k_grid)

    t1 = time.time()
    for r in range(R):
        accum += s[r] * sp.fftconvolve(U[:, r], Vt[r, :])
        if (r + 1) in k_set:
            out_at_k[r + 1] = (accum / counts).copy()
    t_recon = time.time() - t1

    rows = []
    for k in sorted(k_set):
        denoised = out_at_k[k][: len(clean)]
        snr_out = evaluate_against_clean(clean, denoised)
        rows.append({"L": L, "k": k, "snr_out_db": snr_out})

    return rows, {"t_svd_s": t_svd, "t_recon_s": t_recon, "K": K, "R": R}


def main() -> None:
    clean, sr_c = load_mono(CLEAN)
    noisy, sr_n = load_mono(NOISY)
    assert sr_c == sr_n, (sr_c, sr_n)
    N = min(len(clean), len(noisy))
    clean, noisy = clean[:N], noisy[:N]

    snr_in = evaluate_against_clean(clean, noisy)
    print(f"signal length N={N}  sr={sr_c}")
    print(f"input SNR (noisy vs clean) : {snr_in:+.2f} dB")
    print()

    L_grid = [400, 800, 1200, 1600, 2400, 3200]
    k_template = [25, 50, 75, 100, 150, 200, 300, 500, 800, 1200, 1600, 2400, 3200]

    all_rows = []
    for L in L_grid:
        if L >= N:
            continue
        k_grid = sorted({k for k in k_template if k <= L})
        print(f"L={L}  K={N - L + 1}  ks={k_grid}")
        rows, meta = sweep_for_L(noisy, clean, L, k_grid)
        for r in rows:
            r["delta_snr_db"] = r["snr_out_db"] - snr_in
            all_rows.append(r)
            print(
                f"  L={r['L']:5d}  k={r['k']:5d}  "
                f"SNR_out={r['snr_out_db']:+7.2f} dB  "
                f"Δ={r['delta_snr_db']:+7.2f} dB"
            )
        print(
            f"  ↳ svd={meta['t_svd_s']:.1f}s  recon={meta['t_recon_s']:.1f}s\n"
        )

    all_rows.sort(key=lambda r: r["delta_snr_db"], reverse=True)
    print("Top 10 (by Δ SNR):")
    for r in all_rows[:10]:
        print(
            f"  L={r['L']:5d}  k={r['k']:5d}  "
            f"SNR_out={r['snr_out_db']:+7.2f} dB  "
            f"Δ={r['delta_snr_db']:+7.2f} dB"
        )

    out_dir = Path("data/output/evaluation")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "snr_sweep_recording_011_male.json"
    out_path.write_text(
        json.dumps(
            {
                "recording": "recording_011_male",
                "input_snr_db": snr_in,
                "results": all_rows,
            },
            indent=2,
        )
    )
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
