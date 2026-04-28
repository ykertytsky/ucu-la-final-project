"""
Batch AWGN + SVD evaluation using only standard SNR.

Pipeline:
  1) load clean recordings from data/clean
  2) add white Gaussian noise at one target input SNR
  3) measure actual noisy-input SNR against clean
  4) run SVD-on-Hankel denoising
  5) measure denoised-output SNR against clean and save metrics

SNR convention:
    SNR = 10 log10(power(clean) / power(x - clean))
"""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import librosa
import numpy as np
import scipy.linalg as la
import scipy.signal as sp


CLEAN_DIR = Path("data/clean")
OUT_PATH = Path("data/output/evaluation/awgn_svd_snr_metrics.json")
SEED = 0
TARGET_INPUT_SNR_DB = 5.0
WINDOW_LENGTH = 800
K_RANK = 150
MAX_RUN_SECONDS = 1_000_000.0


def load_mono(path: str) -> tuple[np.ndarray, int]:
    x, sr = librosa.load(path, sr=None, mono=True)
    return x.astype(np.float64), sr


def snr_db(s: np.ndarray, e: np.ndarray) -> float:
    ps = float(np.mean(s**2))
    pe = float(np.mean(e**2))
    return float("inf") if pe == 0 else 10.0 * np.log10(ps / pe)


def evaluate(clean: np.ndarray, x: np.ndarray) -> float:
    N = min(len(clean), len(x))
    return snr_db(clean[:N], x[:N] - clean[:N])


def make_awgn(clean: np.ndarray, target_snr_db: float, rng: np.random.Generator) -> np.ndarray:
    raw_noise = rng.standard_normal(clean.shape)
    raw_noise -= float(np.mean(raw_noise))
    ps = float(np.mean(clean**2))
    pn = ps / (10.0 ** (target_snr_db / 10.0))
    return raw_noise / np.sqrt(np.mean(raw_noise**2)) * np.sqrt(pn)


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


def svd_denoise(noisy: np.ndarray, L: int, k: int) -> tuple[np.ndarray, dict]:
    H = np.asarray(build_hankel_view(noisy, L), dtype=np.float64)
    K = H.shape[1]
    t0 = time.time()
    U, s, Vt = la.svd(H, full_matrices=False)
    t_svd = time.time() - t0

    counts = antidiag_counts(L, K)
    accum = np.zeros(L + K - 1)
    for r in range(k):
        accum += s[r] * sp.fftconvolve(U[:, r], Vt[r, :])
    return accum / counts, {"K": K, "t_svd_s": t_svd}


def empty_metrics() -> dict:
    return {
        "seed": SEED,
        "target_input_snr_db": TARGET_INPUT_SNR_DB,
        "window_length": WINDOW_LENGTH,
        "k_rank": K_RANK,
        "snr_definition": "10*log10(mean(clean**2) / mean((x-clean)**2))",
        "results": [],
    }


def load_or_create_metrics() -> dict:
    if not OUT_PATH.exists():
        return empty_metrics()

    metrics = json.loads(OUT_PATH.read_text())
    expected = empty_metrics()
    same_config = all(
        metrics.get(key) == expected[key]
        for key in ["seed", "target_input_snr_db", "window_length", "k_rank"]
    )
    return metrics if same_config else expected


def main() -> None:
    clean_paths = sorted(CLEAN_DIR.glob("*.wav"))
    if not clean_paths:
        raise FileNotFoundError(f"No .wav files found in {CLEAN_DIR}")

    rng = np.random.default_rng(SEED)

    metrics = load_or_create_metrics()
    completed = {r["recording"] for r in metrics["results"]}
    t_run_start = time.time()

    for clean_path in clean_paths:
        if clean_path.stem in completed:
            print(f"\n{clean_path.stem}  already complete, skipping", flush=True)
            continue

        if time.time() - t_run_start > MAX_RUN_SECONDS:
            print("\nRun time budget reached; rerun to resume remaining files.", flush=True)
            break

        clean, sr = load_mono(str(clean_path))
        N = len(clean)
        print(f"\n{clean_path.stem}  N={N}  sr={sr}", flush=True)

        print(f"  target input SNR = {TARGET_INPUT_SNR_DB:+.1f} dB", flush=True)
        noise = make_awgn(clean, TARGET_INPUT_SNR_DB, rng)
        noisy = clean + noise
        snr_in = snr_db(clean, noise)
        print(f"    actual input SNR = {snr_in:+.2f} dB", flush=True)

        L = min(WINDOW_LENGTH, max(2, N // 2))
        k = min(K_RANK, L)
        print(f"    SVD L={L}  k={k}", flush=True)
        denoised, meta = svd_denoise(noisy, L, k)
        snr_out = evaluate(clean, denoised)

        row = {
            "recording": clean_path.stem,
            "sample_rate_hz": sr,
            "n_samples": N,
            "target_input_snr_db": TARGET_INPUT_SNR_DB,
            "input_snr_db": snr_in,
            "L": L,
            "k": k,
            "snr_out_db": snr_out,
            "snr_improvement_db": snr_out - snr_in,
            **meta,
        }
        metrics["results"].append(row)
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUT_PATH.write_text(json.dumps(metrics, indent=2, default=float))
        print(
            f"    output SNR={snr_out:+.2f} dB  "
            f"improvement={snr_out - snr_in:+.2f} dB  "
            f"svd={meta['t_svd_s']:.1f}s",
            flush=True,
        )
        del clean, noise, noisy, denoised
        gc.collect()

    metrics["results"].sort(
        key=lambda r: r["recording"]
    )
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(metrics, indent=2, default=float))
    print(f"\nSaved {OUT_PATH}")


if __name__ == "__main__":
    main()
