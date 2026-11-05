import numpy as np
import h5py
import matplotlib.pyplot as plt

# Select redshift
target_z = 0.0

# Load CAMB data
with open("generated_data/CAMBlinear_sigma.csv", "r") as f:
    camb_radii = np.array(f.readline().strip().split(",")[1:], dtype=float)

linear_camb = np.loadtxt(
    "generated_data/CAMBlinear_sigma.csv",
    delimiter=",",
    skiprows=1
)

camb_redshifts = linear_camb[:, 0]
camb_sigma = linear_camb[:, 1:]

# Load LEGACY data
with h5py.File("optimised/output/variances.h5", "r") as f:
    sim_redshifts = f["redshift"][:]
    sim_radii = f["radii"][:]
    sim_sigma = np.sqrt(f["variance"][:])

# Check redshift arrays match
if not np.allclose(sim_redshifts, camb_redshifts):
    raise ValueError("LEGACY and CAMB redshift arrays do not match.")

# Select the matching redshift
idx = np.where(np.isclose(sim_redshifts, target_z))[0][0]

z = sim_redshifts[idx]

# Extract sigma values at this redshift
legacy_sigma = sim_sigma[idx, :]
camb_sigma_z = camb_sigma[idx, :]

# Interpolate CAMB onto LEGACY radii
camb_matched = np.interp(
    sim_radii,
    camb_radii,
    camb_sigma_z
)

# Calculate fractional residuals
residuals = (legacy_sigma - camb_matched) / camb_matched

# Plot residuals
plt.figure(figsize=(8, 5))

plt.plot(
    sim_radii,
    residuals,
    marker="o",
    markersize=4
)

plt.axhline(0, color="black", linestyle="--", linewidth=1)

plt.xlabel(r"$R\ [\mathrm{Mpc}/h]$")
plt.ylabel(r"Fractional residual")
plt.title(f"LEGACY vs CAMB at z = {z:.3f}")

plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig("plots/residuals.png", dpi=300)
plt.show()