"""Stage 1 (run once): save random sphere centres for reuse."""
import h5py, numpy as np
import config as c

c.OUT_DIR.mkdir(exist_ok=True)
rng = np.random.default_rng(c.SEED)
centres = rng.uniform(0, c.BOX, size=(c.N_SPHERES, 3))
with h5py.File(c.CENTRES_FILE, "w") as f:
    f["centres"] = centres
    f.attrs.update(seed=c.SEED, box=c.BOX, n_spheres=c.N_SPHERES)
print("Saved", c.CENTRES_FILE)