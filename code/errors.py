import numpy as np
import h5py
import matplotlib.pyplot as plt

# Load centres
with h5py.File("output/centres.h5", "r") as f:
    centres = f["centres"][:]

# Load counts and radii
with h5py.File("optimised/output/variances.h5", "r") as f:
    radii = f["radii"][:]
    counts = f["counts/snap068"][:]

# Select R = 8 Mpc/h
radius_index = np.argmin(np.abs(radii - 8.0))
R = radii[radius_index]

# Calculate overdensity
N_total = 2048**3
volume = (4 / 3) * np.pi * R**3
delta = counts[:, radius_index] * 100.0**3 / (N_total * volume) - 1

# Calculate periodic pair separations
i, j = np.triu_indices(len(centres), k=1)

separation = np.abs(centres[i] - centres[j])
separation = np.minimum(separation, 100.0 - separation)
r = np.sqrt(np.sum(separation**2, axis=1))

# Calculate spatial covariance
bins = np.linspace(0, 50, 26)
products = delta[i] * delta[j]

covariance, _ = np.histogram(r, bins=bins, weights=products)
pair_count, _ = np.histogram(r, bins=bins)

covariance = np.divide(
    covariance,
    pair_count,
    out=np.full(len(bins) - 1, np.nan),
    where=pair_count > 0
)

r_mid = (bins[:-1] + bins[1:]) / 2

# Plot
plt.figure(figsize=(8, 5))
plt.plot(r_mid, covariance, marker="o")
plt.axhline(0, color="black", linestyle="--")
plt.xlabel(r"Centre separation $r$ [$\mathrm{Mpc}/h$]")
plt.ylabel(r"$\xi_\delta(r; R=8)$")
plt.title("Spatial covariance of smoothed overdensity at z=99")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
