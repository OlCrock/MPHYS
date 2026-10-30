import h5py
import numpy as np
from pathlib import Path

INPUT_FILE = Path("output/variances.h5")
OUTPUT_FILE = Path("output/summary_statistics.txt")
SELECTED_RADII = [1, 5, 10, 25, 50]

def describe(x):
    return (
        f"min={x.min():.6e}, median={np.median(x):.6e}, "
        f"mean={x.mean():.6e}, max={x.max():.6e}, std={x.std():.6e}"
    )

with h5py.File(INPUT_FILE, "r") as f:
    snapshots = f["snapshots"][:]
    redshifts = f["redshift"][:]
    radii = f["radii"][:]
    variance = f["variance"][:]
    mean_delta = f["mean_delta"][:]

    lines = [
        "LEGACY SIMULATION SUMMARY STATISTICS",
        "=" * 75,
        f"Snapshots: {len(snapshots)}",
        f"Sphere centres per snapshot: {f["counts"][f"snap{int(snapshots[0]):03d}"].shape[0]}",
        f"Radii: {len(radii)}",
        f"Radius range: {radii.min():.2f}-{radii.max():.2f} Mpc/h",
        "",
        "GLOBAL STATISTICS",
        "-" * 75,
        f"Variance (sigma^2): {describe(variance.ravel())}",
        f"Standard deviation (sigma): {describe(np.sqrt(variance).ravel())}",
        f"Mean overdensity: {describe(mean_delta.ravel())}",
        "",
        "PER-SNAPSHOT STATISTICS",
        "-" * 75,
        "Snap   Redshift       Min variance     Max variance     Mean variance    Mean sigma",
    ]

    for i, snap in enumerate(snapshots):
        v = variance[i]
        lines.append(
            f"{snap:4d}   {redshifts[i]:10.5f}   {v.min():14.6e}   "
            f"{v.max():14.6e}   {v.mean():14.6e}   {np.sqrt(v).mean():14.6e}"
        )

    lines.extend([
        "",
        "STATISTICS AT SELECTED RADII",
        "-" * 75,
        "Snap   Redshift   Radius(Mpc/h)   Variance(sigma^2)       Sigma       Mean overdensity",
    ])

    for i, snap in enumerate(snapshots):
        for target in SELECTED_RADII:
            j = np.argmin(np.abs(radii - target))
            lines.append(
                f"{snap:4d}   {redshifts[i]:8.5f}   {radii[j]:12.2f}   "
                f"{variance[i,j]:17.6e}   {np.sqrt(variance[i,j]):10.6e}   "
                f"{mean_delta[i,j]:17.6e}"
            )

    lines.extend(["", "MEAN OVERDENSITY PER SNAPSHOT", "-" * 75])
    lines.append("Snap   Redshift       Min              Mean             Max")

    for i, snap in enumerate(snapshots):
        d = mean_delta[i]
        lines.append(
            f"{snap:4d}   {redshifts[i]:8.5f}   {d.min():14.6e}   "
            f"{d.mean():14.6e}   {d.max():14.6e}"
        )

    lines.extend([
        "",
        "Notes:",
        "- Variance is sigma^2; sigma is its square root.",
        "- Selected radii are matched to the nearest stored radius.",
        "- These are descriptive statistics, not independent validation.",
    ])

OUTPUT_FILE.write_text("\n".join(lines) + "\n")
print(f"Created {OUTPUT_FILE}")
