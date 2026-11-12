"""Append new snapshot results to the existing variances.h5 file."""

import h5py
import numpy as np
import config as c

new_snapshots = np.array(c.SNAPSHOTS, dtype=int)

with h5py.File(c.FINAL_FILE, "r+") as out:

    old_snapshots = out["snapshots"][:].astype(int)

    overlap = np.intersect1d(old_snapshots, new_snapshots)
    if len(overlap):
        raise ValueError(f"Snapshots already exist: {overlap}")

    new_var = []
    new_mean = []
    new_z = []
    new_counts = {}

    for snap in new_snapshots:
        files = sorted(c.COUNTS_DIR.glob(f"snap{snap:03d}_f*_p*.h5"))

        if len(files) != c.N_FILES * c.N_PARTS:
            raise RuntimeError(
                f"Snapshot {snap}: expected {c.N_FILES*c.N_PARTS} "
                f"chunks, found {len(files)}"
            )

        total = 0

        for fn in files:
            with h5py.File(fn, "r") as f:
                total += f["counts"][:]
                redshift = float(f.attrs["redshift"])
                n_total = int(f.attrs["n_total"])

        nbar = n_total / c.BOX**3 * (4 / 3) * np.pi * c.RADII**3
        delta = total / nbar - 1.0

        new_var.append(delta.var(axis=0))
        new_mean.append(delta.mean(axis=0))
        new_z.append(redshift)
        new_counts[f"snap{snap:03d}"] = total

        print(f"Processed snapshot {snap}: z={redshift:.6f}")

    new_var = np.array(new_var)
    new_mean = np.array(new_mean)
    new_z = np.array(new_z)

    # Preserve existing measurements and append new ones.
    combined = {
        "snapshots": np.concatenate((old_snapshots, new_snapshots)),
        "variance": np.concatenate((out["variance"][:], new_var)),
        "mean_delta": np.concatenate((out["mean_delta"][:], new_mean)),
        "redshift": np.concatenate((out["redshift"][:], new_z))
    }

    # Extend the existing datasets without changing their values.
    for name, data in combined.items():
        temp = name + "_new"
        out.create_dataset(temp, data=data)
        del out[name]
        out.move(temp, name)

    # Add the new raw counts.
    for name, data in new_counts.items():
        out.create_dataset(
            f"counts/{name}",
            data=data,
            compression="gzip"
        )

    out.attrs.update(
        n_spheres=c.N_SPHERES,
        seed=c.SEED,
        box=c.BOX
    )

    print("\nAppend complete.")
    print("Total snapshots:", len(combined["snapshots"]))
    print("Snapshots:", combined["snapshots"])
