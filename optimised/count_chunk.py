"""Stage 2 (SLURM array task): cumulative counts N_i(<R_k) for one
(snapshot, file, part) chunk. Counts are additive over chunks."""
import argparse, h5py, numpy as np
from scipy.spatial import cKDTree
import config as c

p = argparse.ArgumentParser()
p.add_argument("--task", type=int, required=True)
p.add_argument("--workers", type=int, default=-1)
a = p.parse_args()

# decode flat task id -> (snapshot, file, part)
n_per_snap = c.N_FILES * c.N_PARTS
snap = c.SNAPSHOTS[a.task // n_per_snap]
rem = a.task % n_per_snap
fidx, part = divmod(rem, c.N_PARTS)

fname = c.BASE_PATH / f"snapdir_{snap:03d}" / f"snapshot_{snap:03d}.{fidx}.hdf5"
with h5py.File(fname, "r") as f:
    hdr = f["Header"].attrs
    redshift = float(hdr["Redshift"])
    scale = c.BOX / float(hdr["BoxSize"])            # convert to Mpc/h
    n_total = int(hdr["NumPart_Total"][1]) + (int(hdr["NumPart_Total_HighWord"][1]) << 32)
    ds = f["PartType1/Coordinates"]
    edges = np.linspace(0, ds.shape[0], c.N_PARTS + 1).astype(int)
    pos = ds[edges[part]:edges[part + 1]].astype(np.float64) * scale

pos %= c.BOX                                          # tree needs [0, box)
tree = cKDTree(pos, boxsize=c.BOX, leafsize=64)
with h5py.File(c.CENTRES_FILE, "r") as f:
    centres = f["centres"][:]

counts = np.empty((len(centres), len(c.RADII)), dtype=np.int64)
for k, r in enumerate(c.RADII):
    counts[:, k] = tree.query_ball_point(centres, r, return_length=True, workers=a.workers)

c.COUNTS_DIR.mkdir(parents=True, exist_ok=True)
out = c.COUNTS_DIR / f"snap{snap:03d}_f{fidx}_p{part}.h5"
with h5py.File(out, "w") as f:
    f["counts"] = counts
    f.attrs.update(snapshot=snap, redshift=redshift, n_total=n_total)
print("Wrote", out)