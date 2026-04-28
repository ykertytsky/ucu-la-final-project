"""
Generate AWGN-corrupted versions of a clean recording and run the SVD-on-Hankel
pipeline with the best (L, k) found for each input SNR. Save both the noisy
input and the denoised output as WAV files.

Best (L, k) per target taken from scripts/awgn_svd_sweep.py results.
"""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import scipy.linalg as la
import scipy.signal as sp
import soundfile as sf


CLEAN = "data/clean/recording_011_male.wav"
SEED = 0
OUT_DIR = Path("data/output/awgn")

BEST_PER_TARGET = [
    {"target_snr_db": -5.0, "L": 800,  "k": 50},
    {"target_snr_db":  0.0, "L": 1600, "k": 150},
    {"target_snr_db":  5.0, "L": 800,  "k": 150},
    {"target_snr_db": 10.0, "L": 800,  "k": 200},
]


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


def evaluate(clean: np.ndarray, x: np.ndarray) -> float:
    g = best_gain(clean, x)
    return snr_db(clean, g * x - clean)


def add_awgn(clean: np.ndarray, target_snr_db: float,
             rng: np.random.Generator) -> np.ndarray:
    ps = float(np.mean(clean**2))
    pn = ps / (10.0 ** (target_snr_db / 10.0))
    return clean + rng.standard_normal(clean.shape) * np.sqrt(pn)


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


def svd_denoise(noisy: np.ndarray, L: int, k: int) -> np.ndarray:
    H = np.asarray(build_hankel_view(noisy, L), dtype=np.float64)
    K = H.shape[1]
    U, s, Vt = la.svd(H, full_matrices=False)
    accum = np.zeros(L + K - 1)
    for r in range(k):
        accum += s[r] * sp.fftconvolve(U[:, r], Vt[r, :])
    counts = antidiag_counts(L, K)
    return accum / counts


def safe_write(path: Path, x: np.ndarray, sr: int) -> None:
    peak = float(np.max(np.abs(x)))
    y = x / peak * 0.99 if peak > 1.0 else x
    sf.write(str(path), y.astype(np.float32), sr)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    clean, sr = load_mono(CLEAN)
    rng = np.random.default_rng(SEED)
    base = Path(CLEAN).stem

    for cfg in BEST_PER_TARGET:
        target = cfg["target_snr_db"]
        L = cfg["L"]
        k = cfg["k"]
        tag = f"snr{int(target)}"
        print(f"\n── target {target:+.1f} dB    L={L}  k={k} ──")

        noisy = add_awgn(clean, target, rng)
        snr_in = evaluate(clean, noisy)

        denoised_full = svd_denoise(noisy, L, k)
        denoised = denoised_full[: len(clean)]
        snr_out = evaluate(clean, denoised)

        print(f"  input  SNR = {snr_in:+6.2f} dB")
        print(f"  output SNR = {snr_out:+6.2f} dB")
        print(f"  Δ SNR      = {snr_out - snr_in:+6.2f} dB")

        noisy_path    = OUT_DIR / f"{base}-{tag}-noisy.wav"
        denoised_path = OUT_DIR / f"{base}-{tag}-svd-L{L}-k{k}-denoised.wav"
        safe_write(noisy_path,    noisy,    sr)
        safe_write(denoised_path, denoised, sr)
        print(f"  wrote {noisy_path}")
        print(f"  wrote {denoised_path}")


if __name__ == "__main__":
    main()
