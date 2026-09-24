import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

base_path = Path("/disk12/legacy/GVD_C700_l100n256_SLEGAC/dm_gadget/data")

snapshot_numbers = [1, 4, 8, 12, 15]
n_files_per_snapshot = 4
slice_width = 2.0  # Mpc/h

fig, axes = plt.subplots(
    1,
    len(snapshot_numbers),
    figsize=(20, 4)
)

for ax, snap in zip(axes, snapshot_numbers):

    snapdir = base_path / f"snapdir_{snap:03d}"
    chunks = []

    for i in range(n_files_per_snapshot):

        file = snapdir / f"snapshot_{snap:03d}.{i}.hdf5"

        if not file.exists():
            print(f"Missing: {file}")
            continue

        with h5py.File(file, "r") as f:

            # Coordinates are in kpc/h, so convert to Mpc/h
            coords = f["PartType1/Coordinates"][:] / 1000.0

            # BoxSize is also originally in kpc/h
            box_size = float(f["Header"].attrs["BoxSize"]) / 1000.0

            # Centre the z-slice on the middle of the simulation box
            z_mid = box_size / 2.0
            z_min = z_mid - slice_width / 2.0
            z_max = z_mid + slice_width / 2.0

            # Select particles inside the z-slice
            mask = (
                (coords[:, 2] >= z_min) &
                (coords[:, 2] <= z_max)
            )

            # Keep x and y positions
            xy_slice = coords[mask, :2]

            if xy_slice.size > 0:
                chunks.append(xy_slice)

    if not chunks:
        print(f"No particles found for snapshot {snap:03d}")
        continue

    positions = np.concatenate(chunks)

    x = positions[:, 0]
    y = positions[:, 1]

    print(
        f"Snapshot {snap:03d}: "
        f"{len(x)} particles in z-slice"
    )

    ax.scatter(
        x,
        y,
        s=0.5,
        alpha=0.4
    )

    ax.set_title(f"Snapshot {snap:03d}")
    ax.set_xlabel("x [Mpc/h]")
    ax.set_ylabel("y [Mpc/h]")

    ax.set_xlim(0, box_size)
    ax.set_ylim(0, box_size)

plt.tight_layout()

plt.savefig(
    "snapshot_evolution.png",
    dpi=200
)
