import h5py
import numpy as np

path = "/disk12/legacy/GVD_C700_l100n256_SLEGAC/dm_gadget/data/snapdir_005"

positions = []

for i in range(4):
    file = f"{path}/snapshot_005.{i}.hdf5"

    with h5py.File(file, "r") as f:
        positions.append(f["PartType1/Coordinates"][:])

positions = np.concatenate(positions)

x = positions[:, 0]
y = positions[:, 1]

print("Number of particles:", len(x))
print("x shape:", x.shape)
print("y shape:", y.shape)



