import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt

# Load LEGACY variances
with h5py.File("optimised/output/variances.h5", "r") as f:
    sim_redshifts = f["redshift"][:]
    sim_radii = f["radii"][:]
    sim_variance = f["variance"][:]

# Convert simulation variance to standard deviation
sim_sigma = np.sqrt(sim_variance)

# Load theoretical linear sigma
linear_df = pd.read_csv("linear_sigma.csv")

# Select actual LEGACY redshifts
target_z = [99.000000, 19.908000, 13.989296, 3.031519]

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
axes = axes.flatten()

for ax, target in zip(axes, target_z):

    # Find LEGACY snapshot
    i = np.argmin(np.abs(sim_redshifts - target))
    z = sim_redshifts[i]

    # Extract theoretical values at the same redshift
    theory = linear_df[
        np.isclose(linear_df["Redshift"], z, atol=1e-5)
    ].sort_values("Radius (Mpc/h)")

    linear_radii = theory["Radius (Mpc/h)"].values
    linear_sigma = theory["Sigma"].values

    # Plot LEGACY
    ax.plot(
        sim_radii, sim_sigma[i], "o",
        markersize=3, label="LEGACY simulation"
    )

    # Plot linear theory
    ax.plot(
        linear_radii, linear_sigma, "-",
        label="Linear theory (CAMB)"
    )

    ax.set_title(f"z = {z:.3f}")
    ax.set_xlabel(r"Radius ($\mathrm{Mpc}/h$)")
    ax.set_ylabel(r"Standard deviation ($\sigma$)")
    ax.set_yscale("log")
    ax.set_xlim(1, 20)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

plt.tight_layout()
plt.savefig("linear_theory_comparison_initial.png", dpi=200)
plt.show()