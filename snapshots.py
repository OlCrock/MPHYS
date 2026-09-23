import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

base_path = Path("/disk12/legacy/GVD_C700_l100n256_SLEGAC/dm_gadget/data")
snapshot_numbers = [5, 6, 7, 8, 9]  # 5 snapshots
n_files_per_snapshot = 4
slice_width = 5.0  # Mpc/h, thickness of the z-slice

fig, ax = plt.subplots(figsize=(8, 8))

for snap in snapshot_numbers:
    snapdir = base_path / f"snapdir_{snap:03d}"
    chunks = []

    for i in range(n_files_per_snapshot):
        file = snapdir / f"snapshot_{snap:03d}.{i}.hdf5"
        if not file.exists():
            continue

        with h5py.File(file, "r") as f:
            coords = f["PartType1/Coordinates"][:]
            box_size = float(f["Header"].attrs["BoxSize"]) / 1000.0  # kpc/h -> Mpc/h

            # select a z slice of width = 10 Mpc, centered on the box centre
            z_mid = box_size / 2.0
            z_min = z_mid - slice_width / 2.0
            z_max = z_mid + slice_width / 2.0

            mask = (coords[:, 2] >= z_min) & (coords[:, 2] <= z_max)
            xy_slice = coords[mask, :2]  # keep only x and y for particles in the z slice

            if xy_slice.size > 0:
                chunks.append(xy_slice)

    if not chunks:
        print(f"No particles found in z-slice for snapshot {snap:03d}")
        continue

    positions = np.concatenate(chunks)
    x = positions[:, 0]
    y = positions[:, 1]

    print(f"Snapshot {snap:03d}: {len(x)} particles in z-slice")
    ax.scatter(x, y, s=0.5, alpha=0.4, label=f"Snap {snap:03d}")

ax.set_xlabel("x [Mpc/h]")
ax.set_ylabel("y [Mpc/h]")
ax.set_title(f"Particle x-y positions for 5 snapshots, z-slice = {slice_width} Mpc/h")
ax.legend(markerscale=2)
plt.tight_layout()
plt.savefig("slice.png", dpi=200)



