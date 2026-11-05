import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt

# Load LEGACY simulation data
with h5py.File("optimised/output/variances.h5", "r") as f:
    sim_redshifts = f["redshift"][:]
    sim_radii = f["radii"][:]
    sim_sigma = np.sqrt(f["variance"][:])

# Load theoretical linear sigma
linear_df = pd.read_csv("linear_sigma.csv")

# Select redshift to plot
target_z = 3.031519

# Find nearest LEGACY snapshot
i = np.argmin(np.abs(sim_redshifts - target_z))
z = sim_redshifts[i]

# Select theoretical values at the same redshift
theory = linear_df[
    np.isclose(linear_df["Redshift"], z, atol=1e-5)
].sort_values("Radius (Mpc/h)")

linear_radii = theory["Radius (Mpc/h)"].values
linear_sigma = theory["Sigma"].values

# Interpolate theory onto LEGACY radii
theory_matched = np.interp(sim_radii, linear_radii, linear_sigma)

# Calculate fractional residuals
residuals = (sim_sigma[i] - theory_matched) / theory_matched

# Plot
plt.figure(figsize=(8, 5))
plt.axhline(0, color="black", linestyle="--", linewidth=1)
plt.plot(sim_radii, residuals, "o", markersize=3)

plt.xlabel(r"Radius ($\mathrm{Mpc}/h$)")
plt.ylabel("Fractional residual")
plt.title(f"LEGACY vs linear theory (z = {z:.3f})")
plt.xlim(1, 20)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("residuals.png", dpi=300)
plt.show()