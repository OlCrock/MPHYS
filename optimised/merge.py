"""Stage 3: sum chunks, compute variances for all radii, write final HDF5."""
import h5py, numpy as np
import config as c

with h5py.File(c.FINAL_FILE, "w") as out:
    out["radii"] = c.RADII
    out["snapshots"] = np.array(c.SNAPSHOTS)
    var = np.full((len(c.SNAPSHOTS), len(c.RADII)), np.nan)
    mean_d = np.full_like(var, np.nan)
    zs = np.full(len(c.SNAPSHOTS), np.nan)
    for s, snap in enumerate(c.SNAPSHOTS):
        files = sorted(c.COUNTS_DIR.glob(f"snap{snap:03d}_f*_p*.h5"))
        if len(files) != c.N_FILES * c.N_PARTS:
            print(f"WARNING snap {snap}: {len(files)} chunks found, skipping")
            continue
        total = 0
        for fn in files:
            with h5py.File(fn, "r") as f:
                total = total + f["counts"][:]
                zs[s], n_total = f.attrs["redshift"], f.attrs["n_total"]
        nbar = n_total / c.BOX**3 * (4 / 3) * np.pi * c.RADII**3
        delta = total / nbar - 1.0
        var[s], mean_d[s] = delta.var(axis=0), delta.mean(axis=0)
        out.create_dataset(f"counts/snap{snap:03d}", data=total, compression="gzip")
    out["variance"] = var
    out["mean_delta"] = mean_d
    out["redshift"] = zs
    out.attrs.update(n_spheres=c.N_SPHERES, seed=c.SEED, box=c.BOX)
print("Wrote", c.FINAL_FILE)