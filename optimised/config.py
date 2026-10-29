from pathlib import Path
import numpy as np

BASE_PATH = Path("/cephfs2/TMOX/legacy/GVD_C700_l100n2048_SLEGAC/dm_gadget/data")
OUT_DIR = Path("output")              # all products go here
SNAPSHOTS = [0, 1, 3, 4, 6, 7, 9, 10, 12, 14, 15, 55]
N_FILES = 4                           # files per snapshot
N_PARTS = 4                           # sub-chunks per file (raise if RAM is tight)
BOX = 100.0                           # Mpc/h
N_SPHERES = 500
SEED = 42
RADII = np.linspace(1.0, 50.0, 50)    # Mpc/h; max radius = 50 (the "big" run)
CENTRES_FILE = OUT_DIR / "centres.h5"
COUNTS_DIR = OUT_DIR / "counts"
FINAL_FILE = OUT_DIR / "variances.h5"
