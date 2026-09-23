import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

base_path = Path("/disk12/legacy/GVD_C700_l100n256_SLEGAC/dm_gadget/data")
snapshot_numbers = [5, 6, 7, 8, 9]  # 5 snapshots
n_files_per_snapshot = 4

fig, ax = plt.subplots(figsize=(8, 8))

for snap in snapshot_numbers:
    snapdir = base_path / f"snapdir_{snap:03d}"
    chunks = []

    for i in range(n_files_per_snapshot):
        file = snapdir / f"snapshot_{snap:03d}.{i}.hdf5"
        if not file.exists():
            continue

    with h5py.File(file, "r") as f:
        positions.append(f["PartType1/Coordinates"][:])

positions = np.concatenate(positions)

x = positions[:, 0]
y = positions[:, 1]

print("Number of particles:", len(x))
print("x shape:", x.shape)
print("y shape:", y.shape)



