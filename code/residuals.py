import numpy as np
import h5py
import matplotlib.pyplot as plt

# Load CAMB data
linear_camb = np.loadtxt(
    "generated_data/CAMBlinear_sigma.csv",
    delimiter=",",
    skiprows=1
)

camb_radii = np.loadtxt(
    "generated_data/CAMBlinear_sigma.csv",
    delimiter=",",
    skiprows=1,
    max_rows=0
)

camb_redshifts = linear_camb[:, 0]
camb_sigma = linear_camb[:, 1:]

# Load LEGACY data
with h5py.File("optimised/output/variances.h5", "r") as f:
    sim_redshifts = f["redshift"][:]
    sim_radii = f["radii"][:]
    sim_sigma = np.sqrt(f["variance"][:])

# Interpolate CAMB onto the LEGACY redshifts and radii
camb_matched = np.zeros_like(sim_sigma)

for j in range(len(camb_radii)):
    camb_at_sim_z = np.interp(
        sim_redshifts,
        camb_redshifts,
        camb_sigma[:, j]
    )

    camb_matched[:, :] = camb_matched[:, :]  # no change to LEGACY data

    if j == 0:
        camb_at_sim_z_all = np.zeros((len(sim_redshifts), len(camb_radii)))

    camb_at_sim_z_all[:, j] = camb_at_sim_z

for i in range(len(sim_redshifts)):
    camb_matched[i, :] = np.interp(
        sim_radii,
        camb_radii,
        camb_at_sim_z_all[i, :]
    )

# Calculate fractional residuals
residuals = (sim_sigma - camb_matched) / camb_matched

# Plot residuals
fig, ax = plt.subplots(figsize=(9, 6))

for i, z in enumerate(sim_redshifts):
    ax.plot(
        sim_radii,
        residuals[i],
        marker="o",
        markersize=3,
        label=f"z = {z:.2f}"
    )

ax.axhline(0, color="black", linestyle="--", linewidth=1)

ax.set_xlabel(r"$R\ [\mathrm{Mpc}/h]$")
ax.set_ylabel(r"Fractional residual $(\sigma_{\mathrm{LEGACY}}-\sigma_{\mathrm{CAMB}})/\sigma_{\mathrm{CAMB}}$")
ax.set_title("LEGACY vs CAMB: Fractional Residuals")
ax.legend(fontsize=8, ncol=2)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("plots/CAMB_LEGACY_residuals.png", dpi=300)
plt.show()