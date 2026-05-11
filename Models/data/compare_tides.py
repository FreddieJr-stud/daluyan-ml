"""Compare our harmonic tide model against tidetime.org reference data.

Usage:
    python -m data.compare_tides
"""

from __future__ import annotations

import math
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from features.tide import predict_tide_height

# Reference data from tidetime.org — Feb 2026, Manila (14.604°N, 120.982°E)
# Format: (day, time_str, height_m, type)
REFERENCE = [
    (1, "05:29", -0.43, "low"), (1, "21:15", 1.28, "high"),
    (2, "06:06", -0.39, "low"), (2, "22:08", 1.25, "high"),
    (3, "06:38", -0.32, "low"), (3, "22:56", 1.16, "high"),
    (4, "07:03", -0.22, "low"),
    (5, "07:23", -0.12, "low"),
    (8, "07:50", 0.16, "low"),
    (9, "07:37", 0.20, "low"), (9, "15:04", 0.72, "high"), (9, "23:28", 0.11, "low"),
    (11, "01:34", 0.03, "low"), (11, "16:42", 0.83, "high"),
    (12, "02:52", -0.06, "low"), (12, "17:48", 0.88, "high"),
    (13, "03:42", -0.13, "low"), (13, "18:55", 0.93, "high"),
    (14, "04:22", -0.18, "low"), (14, "19:54", 0.99, "high"),
    (15, "04:57", -0.22, "low"), (15, "20:44", 1.04, "high"),
    (16, "05:28", -0.23, "low"), (16, "21:29", 1.07, "high"),
    (17, "05:54", -0.22, "low"), (17, "22:10", 1.07, "high"),
    (18, "06:17", -0.18, "low"), (18, "22:51", 1.02, "high"),
    (19, "06:34", -0.12, "low"), (19, "23:32", 0.92, "high"),
    (20, "06:48", -0.03, "low"),
    (21, "00:17", 0.77, "high"), (21, "06:56", 0.07, "low"), (21, "12:56", 0.52, "high"),
    (23, "02:10", 0.40, "high"), (23, "06:50", 0.22, "low"),
    (23, "13:45", 0.80, "high"), (23, "21:39", -0.01, "low"),
    (24, "04:11", 0.23, "high"), (24, "05:58", 0.22, "low"),
    (24, "14:25", 0.92, "high"), (24, "23:37", -0.09, "low"),
    (26, "01:35", -0.19, "low"), (26, "16:30", 1.06, "high"),
    (27, "02:52", -0.28, "low"), (27, "17:51", 1.10, "high"),
    (28, "03:46", -0.33, "low"), (28, "19:09", 1.12, "high"),
]


def main():
    print("Comparison: Our harmonic model vs tidetime.org reference")
    print("=" * 72)
    print(f"{'Date':>8s}  {'Time':>6s}  {'Type':>5s}  {'Ref(m)':>7s}  {'Pred(m)':>8s}  {'Err(m)':>7s}")
    print("-" * 72)

    errors = []
    abs_errors = []
    cal_mask = []  # True if calibration day

    for day, time_str, ref_h, tide_type in REFERENCE:
        h, m = map(int, time_str.split(":"))
        dt = datetime(2026, 2, day, h, m)
        pred_h = predict_tide_height(dt)
        err = pred_h - ref_h
        errors.append(err)
        abs_errors.append(abs(err))
        is_cal = day == 23
        cal_mask.append(is_cal)
        tag = " (cal)" if is_cal else ""
        print(
            f"  Feb {day:2d}  {time_str:>6s}  {tide_type:>5s}  "
            f"{ref_h:+7.2f}  {pred_h:+8.3f}  {err:+7.3f}{tag}"
        )

    # Overall stats
    n = len(errors)
    rmse = math.sqrt(sum(e**2 for e in errors) / n)
    mae = sum(abs_errors) / n
    bias = sum(errors) / n
    print("-" * 72)
    print(f"N = {n} tide events")
    print(f"MAE:  {mae:.3f} m")
    print(f"RMSE: {rmse:.3f} m")
    print(f"Bias: {bias:+.3f} m")
    print(f"Max |error|: {max(abs_errors):.3f} m")

    # Exclude calibration day
    nc_err = [e for e, c in zip(errors, cal_mask) if not c]
    nc_abs = [a for a, c in zip(abs_errors, cal_mask) if not c]
    if nc_err:
        rmse_nc = math.sqrt(sum(e**2 for e in nc_err) / len(nc_err))
        mae_nc = sum(nc_abs) / len(nc_abs)
        bias_nc = sum(nc_err) / len(nc_err)
        print(f"\nExcluding calibration day (Feb 23), N={len(nc_err)}:")
        print(f"MAE:  {mae_nc:.3f} m")
        print(f"RMSE: {rmse_nc:.3f} m")
        print(f"Bias: {bias_nc:+.3f} m")
        print(f"Max |error|: {max(nc_abs):.3f} m")

    # By tide type
    for tt in ("high", "low"):
        tt_err = [e for (_, _, _, t), e in zip(REFERENCE, errors) if t == tt]
        tt_abs = [abs(e) for e in tt_err]
        if tt_err:
            print(f"\n{tt.capitalize()} tides only (N={len(tt_err)}):")
            print(f"  MAE:  {sum(tt_abs)/len(tt_abs):.3f} m")
            print(f"  Bias: {sum(tt_err)/len(tt_err):+.3f} m")


if __name__ == "__main__":
    main()
