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

sim_sigma8 = sim_sigma8[:-1]
sim_redshifts = sim_redshifts[:-1]

plt.scatter(sim_redshifts, sim_sigma8, label="Simulation", color="blue")
plt.xlabel("Redshift (z)")
plt.ylabel(r"$\sigma_8$")
plt.show()
