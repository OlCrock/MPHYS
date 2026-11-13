# plot sigma 8 as a function of z

import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt

with h5py.File("optimised/output/variances.h5", "r") as f:
    sim_redshifts = f["redshift"][:]
    sim_radii = f["radii"][:]
    sim_variance = f["variance"][:]

sim_sigma8 = np.sqrt(sim_variance[:, sim_radii == 8.0].flatten())

sim_sigma8 = sim_sigma8[1:]
sim_redshifts = sim_redshifts[1:]

sigma_lin = np.loadtxt("linear_sigma.csv", delimiter=",", skiprows=1)

camb_redshifts = sigma_lin[sigma_lin[:, 1] == 8.0, 0]
camb_sigma8 = sigma_lin[sigma_lin[:, 1] == 8.0, 2]

# Match to simulation redshifts
matched_sigma8 = np.array([
    camb_sigma8[np.argmin(np.abs(camb_redshifts - z))]
    for z in sim_redshifts
])


plt.scatter(sim_redshifts, sim_sigma8, label="Simulation", color="blue")
plt.scatter(sim_redshifts, matched_sigma8, label="Linear Theory", color="red", marker="x")
plt.axhline(0, color="black", linestyle="--", label="Linear Theory")
plt.xlabel("Redshift (z)")
plt.ylabel(r"$\sigma_8$")
plt.show()

# print ratio alonf with redshifts

for z, ratio in zip(sim_redshifts, sim_sigma8/matched_sigma8):
    print(f"z = {z:.2f}, ratio = {ratio:.6f}")
