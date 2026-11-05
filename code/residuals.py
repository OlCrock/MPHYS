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

# Find selected LEGACY redshift
idx = np.argmin(np.abs(sim_redshifts - target_z))
z = sim_redshifts[idx]

# Find matching CAMB redshift
camb_idx = np.argmin(np.abs(camb_redshifts - z))

if not np.isclose(camb_redshifts[camb_idx], z):
    raise ValueError(
        f"No matching CAMB redshift found for LEGACY z = {z}"
    )

print(f"LEGACY redshift: {z}")
print(f"CAMB redshift:   {camb_redshifts[camb_idx]}")

# Extract sigma values
legacy_sigma = sim_sigma[idx, :]
camb_sigma_z = camb_sigma[camb_idx, :]

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