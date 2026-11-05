from pathlib import Path
import numpy as np

BASE_PATH = Path("/cephfs2/TMOX/legacy/GVD_C700_l100n2048_SLEGAC/dm_gadget/data")
OUT_DIR = Path("output")              # all products go here
SNAPSHOTS = [55, 60, 65, 70, 75, 78, 81, 88, 95, 101]
N_FILES = 128                           # files per snapshot
N_PARTS = 4                           # sub-chunks per file (raise if RAM is tight)
BOX = 100.0                           # Mpc/h
N_SPHERES = 500
SEED = 42
RADII = np.linspace(1.0, 50.0, 50)    # Mpc/h; max radius = 50 (the "big" run)
CENTRES_FILE = OUT_DIR / "centres.h5"
COUNTS_DIR = OUT_DIR / "counts"
FINAL_FILE = OUT_DIR / "variances.h5"
