
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt

# Load simulation data
with h5py.File("optimised/output/variances.h5", "r") as f:
    sim_redshifts = f["redshift"][:]
    sim_radii = f["radii"][:]
    sim_variance = f["variance"][:]
    sim_variance_error = f["variance_error_gaussian"][:]

# Extract sigma8 and its propagated error
radius_index = np.argmin(np.abs(sim_radii - 8.0))

sim_sigma8 = np.sqrt(sim_variance[:, radius_index])
sim_sigma8_error = (
    sim_variance_error[:, radius_index] / (2 * sim_sigma8)
)

# Remove first redshift point
sim_sigma8 = sim_sigma8[1:]
sim_sigma8_error = sim_sigma8_error[1:]
sim_redshifts = sim_redshifts[1:]

# Load linear theory
sigma_lin = np.loadtxt("generated_data/linear_sigma.csv", delimiter=",", skiprows=1)

camb_redshifts = sigma_lin[sigma_lin[:, 1] == 8.0, 0]
camb_sigma8 = sigma_lin[sigma_lin[:, 1] == 8.0, 2]

# Match to simulation redshifts
matched_sigma8 = np.array([
    camb_sigma8[np.argmin(np.abs(camb_redshifts - z))]
    for z in sim_redshifts])

sort_idx = np.argsort(sim_redshifts)

sim_redshifts = sim_redshifts[sort_idx]
sim_sigma8 = sim_sigma8[sort_idx]
sim_sigma8_error = sim_sigma8_error[sort_idx]
matched_sigma8 = matched_sigma8[sort_idx]

# Plot
plt.figure(figsize=(8, 5))
plt.errorbar(sim_redshifts, sim_sigma8, yerr=sim_sigma8_error, fmt="o", capsize=1, markersize = 3,label="Simulation")
plt.xlabel("Redshift (z)")
plt.ylabel(r"$\sigma_8$")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

