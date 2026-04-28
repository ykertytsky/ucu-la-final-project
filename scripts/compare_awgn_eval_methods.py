#!/usr/bin/env python3
"""
Load batch AWGN evaluation CSVs, align on ``recording``, and plot SNR metrics.

Default inputs (under repo root unless paths are absolute):
  - batch_lowpass_awgn_eval.csv
  - batch_bandpass_awgn_eval.csv
  - batch_spectral_awgn_eval.csv
  - pipeline_eval_summary.csv (Hankel + truncated SVD)

Usage:
    python scripts/compare_awgn_eval_methods.py
    python scripts/compare_awgn_eval_methods.py --eval-dir data/output/evaluation
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError as e:  # pragma: no cover
    raise SystemExit("matplotlib is required. pip install matplotlib") from e


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _float(row: dict[str, str], key: str) -> float:
    v = row.get(key)
    if v is None or str(v).strip() == "":
        return float("nan")
    return float(v)


def _row_ok(row: dict[str, str]) -> bool:
    err = (row.get("error") or "").strip()
    return err == ""


def load_eval_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def rows_to_by_recording(
    rows: list[dict[str, str]],
    method_label: str,
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for row in rows:
        if not _row_ok(row):
            continue
        rec = row.get("recording")
        if not rec:
            continue
        out[rec] = {
            "method": method_label,
            "input_snr_db": _float(row, "input_snr_db"),
            "output_snr_db": _float(row, "output_snr_db"),
            "snr_improvement_db": _float(row, "snr_improvement_db"),
        }
    return out


def default_csv_specs(eval_dir: Path) -> list[tuple[str, Path]]:
    specs: list[tuple[str, Path]] = [
        ("Lowpass", eval_dir / "batch_lowpass_awgn_eval.csv"),
        ("Bandpass", eval_dir / "batch_bandpass_awgn_eval.csv"),
        ("Spectral subtraction", eval_dir / "batch_spectral_awgn_eval.csv"),
        ("Hankel + SVD", eval_dir / "pipeline_eval_summary.csv"),
    ]
    hp = eval_dir / "batch_highpass_awgn_eval.csv"
    if hp.is_file():
        specs.append(("Highpass", hp))
    return specs


def write_long_csv(path: Path, long_rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["method", "recording", "input_snr_db", "output_snr_db", "snr_improvement_db"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in long_rows:
            w.writerow({k: r.get(k) for k in fieldnames})


def main() -> None:
    root = _repo_root()
    ap = argparse.ArgumentParser(description="Compare AWGN batch eval CSVs and plot SNR metrics.")
    ap.add_argument(
        "--eval-dir",
        type=Path,
        default=root / "data/output/evaluation",
        help="Directory containing evaluation CSVs",
    )
    ap.add_argument(
        "--out-figure",
        type=Path,
        default=None,
        help="Output PNG path (default: <eval-dir>/method_comparison_snr.png)",
    )
    ap.add_argument(
        "--out-long-csv",
        type=Path,
        default=None,
        help="Optional long-format merge CSV (default: <eval-dir>/method_comparison_long.csv)",
    )
    args = ap.parse_args()

    eval_dir = args.eval_dir.resolve()
    out_fig = (args.out_figure or (eval_dir / "method_comparison_snr.png")).resolve()
    out_long = (args.out_long_csv or (eval_dir / "method_comparison_long.csv")).resolve()

    specs = default_csv_specs(eval_dir)
    series: list[tuple[str, dict[str, dict[str, float]]]] = []
    long_rows: list[dict[str, Any]] = []

    for label, csv_path in specs:
        raw = load_eval_csv(csv_path)
        if not raw:
            print(f"[skip] missing or empty: {csv_path}")
            continue
        by_rec = rows_to_by_recording(raw, label)
        if not by_rec:
            print(f"[skip] no valid rows: {csv_path}")
            continue
        series.append((label, by_rec))
        for rec, m in by_rec.items():
            long_rows.append(
                {
                    "method": label,
                    "recording": rec,
                    "input_snr_db": m["input_snr_db"],
                    "output_snr_db": m["output_snr_db"],
                    "snr_improvement_db": m["snr_improvement_db"],
                }
            )
        print(f"[ok] {label}: {len(by_rec)} recordings from {csv_path.name}")

    if len(series) < 2:
        raise SystemExit("Need at least two non-empty CSVs to compare.")

    write_long_csv(out_long, long_rows)
    print(f"Wrote {out_long}")

    methods = [s[0] for s in series]
    improvements = [
        np.array([m["snr_improvement_db"] for m in br.values()], dtype=np.float64) for _, br in series
    ]
    outputs = [np.array([m["output_snr_db"] for m in br.values()], dtype=np.float64) for _, br in series]

    means_imp = [float(np.nanmean(x)) for x in improvements]
    stds_imp = [float(np.nanstd(x, ddof=1)) if np.sum(np.isfinite(x)) > 1 else 0.0 for x in improvements]
    means_out = [float(np.nanmean(x)) for x in outputs]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax0 = axes[0]
    positions = np.arange(1, len(methods) + 1)
    bp = ax0.boxplot(
        improvements,
        positions=positions,
        patch_artist=True,
        showfliers=True,
    )
    ax0.set_xticks(positions)
    ax0.set_xticklabels(methods, rotation=20)
    for patch, color in zip(bp["boxes"], plt.cm.Set3(np.linspace(0, 1, len(bp["boxes"])))):
        patch.set_facecolor(color)
    ax0.axhline(0.0, color="0.5", lw=0.8, ls="--")
    ax0.set_ylabel("SNR improvement (dB)")
    ax0.set_title("Distribution vs clean (per recording)")

    ax1 = axes[1]
    x = np.arange(len(methods))
    colors = plt.cm.Set2(np.linspace(0, 1, len(methods)))
    bars = ax1.bar(x, means_imp, yerr=stds_imp, capsize=4, color=colors, edgecolor="0.3", linewidth=0.6)
    ax1.axhline(0.0, color="0.5", lw=0.8, ls="--")
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=20)
    ax1.set_ylabel("Mean ± std (dB)")
    ax1.set_title("Average SNR improvement")

    # annotate mean output SNR on bars (secondary cue)
    for i, (rect, mo) in enumerate(zip(bars, means_out)):
        h = rect.get_height()
        ax1.text(
            rect.get_x() + rect.get_width() / 2.0,
            h + stds_imp[i] + 0.08,
            f"out≈{mo:.1f} dB",
            ha="center",
            va="bottom",
            fontsize=8,
            color="0.35",
        )

    fig.suptitle(
        "AWGN batch eval — SNR improvement vs clean (target input SNR ≈ 5 dB)",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_fig}")

    # Console summary table
    print("\nMean ± std (SNR improvement, dB) | mean output SNR (dB)")
    print("-" * 72)
    for m, mi, si, mo in zip(methods, means_imp, stds_imp, means_out):
        print(f"  {m:22s}  {mi:+.2f} ± {si:.2f}     {mo:.2f}")


if __name__ == "__main__":
    main()
