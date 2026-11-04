import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt

# Load LEGACY variances
with h5py.File("optimised/output/variances.h5", "r") as f:
    sim_redshifts = f["redshift"][:]
    sim_radii = f["radii"][:]
    sim_variance = f["variance"][:]

# Load theoretical linear variances
linear_df = pd.read_csv("linear_sigma.csv", index_col=0)
linear_df.index = linear_df.index.astype(float)
linear_df.columns = linear_df.columns.astype(float)

# Select actual LEGACY redshifts
target_z = [99.000000, 19.908000, 13.989296, 3.031519]

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
axes = axes.flatten()

for ax, target in zip(axes, target_z):
    i = np.argmin(np.abs(sim_redshifts - target))
    z = sim_redshifts[i]

    # Match theoretical redshift
    theory_z = linear_df.index[np.argmin(np.abs(linear_df.index - z))]

    ax.plot(sim_radii, sim_variance[i], "o", markersize =2, label="LEGACY simulation")
    ax.plot(linear_df.columns, linear_df.loc[theory_z], "-", label="Linear theory (CAMB)")

    ax.set_title(f"z = {z:.3f}")
    ax.set_xlabel(r"Radius ($\mathrm{Mpc}/h$)")
    ax.set_ylabel(r"Variance ($\sigma^2$)")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.set_xlim(1, 20)
    ax.legend()

plt.tight_layout()
plt.savefig("linear_theory_comparison_initial.png", dpi=200)
plt.show()