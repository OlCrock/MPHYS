import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt

# Load LEGACY data
with h5py.File("optimised/output/variances.h5", "r") as f:
    sim_redshifts = f["redshift"][:]
    sim_snapshots = f["snapshots"][:]
    sim_radii = f["radii"][:]
    sim_variance = f["variance"][:]
    counts_by_snap = {
        name: f["counts"][name][:] for name in f["counts"].keys()
    }
    n_spheres = f.attrs["n_spheres"]
    box = f.attrs["box"]

# Convert variance to standard deviation
sim_sigma = np.sqrt(sim_variance)

# Load theoretical linear sigma
linear_df = pd.read_csv("linear_sigma.csv")


def poisson_errors(counts, radii, n_total, box, n_boot=2000):

    rng = np.random.default_rng(42)

    # Expected number of particles per sphere
    nbar = (
        n_total / box**3
        * (4 / 3) * np.pi * radii**3
    )

    # Poisson bootstrap
    boot_sigma = np.empty((n_boot, len(radii)))

    for i in range(n_boot):

        samples = rng.poisson(lam=nbar, size=counts.shape)

        delta = samples / nbar - 1.0

        boot_sigma[i] = np.sqrt(delta.var(axis=0))

    # Asymmetric 16th and 84th percentile uncertainties
    lower = np.percentile(boot_sigma, 16, axis=0)
    upper = np.percentile(boot_sigma, 84, axis=0)

    return lower, upper


# Select actual LEGACY redshifts
target_z = [99.000000, 3.031519, 2.0, 1.5, 1.0, 0.0]

fig, axes = plt.subplots(3, 2, figsize=(12, 9))
axes = axes.flatten()

for ax, target in zip(axes, target_z):

    # Find LEGACY snapshot
    i = np.argmin(np.abs(sim_redshifts - target))
    z = sim_redshifts[i]

    # Get snapshot counts
    snapshot_id = sim_snapshots[i]
    counts = counts_by_snap[f"snap{snapshot_id:03d}"]

    # Recover total particle count from the mean sphere counts
    nbar = (
        counts.mean(axis=0)
        / (1 + sim_sigma[i] * 0)  # Placeholder removed below
    )

    # Calculate expected counts using the simulation's total particle count
    # This is recovered from the stored counts and sphere volumes.
    n_total = (
        counts.mean(axis=0)[0]
        * box**3
        / ((4 / 3) * np.pi * sim_radii[0]**3)
    )

    err_low, err_high = poisson_errors(
        counts, sim_radii, n_total, box
    )

    # Convert percentile bounds into distances from measured sigma
    err_down = np.maximum(sim_sigma[i] - err_low, 0)
    err_up = np.maximum(err_high - sim_sigma[i], 0)

    # Extract theoretical values at the same redshift
    theory = linear_df[
        np.isclose(linear_df["Redshift"], z, atol=1e-5)
    ].sort_values("Radius (Mpc/h)")

    linear_radii = theory["Radius (Mpc/h)"].values
    linear_sigma = theory["Sigma"].values

    # Plot LEGACY with error bars
    ax.errorbar(
        sim_radii,
        sim_sigma[i],
        yerr=[err_down, err_up],
        fmt="o",
        markersize=3,
        label="LEGACY simulation",
        capsize=2,
        color="#2C5F8A",
    )

    # Plot linear theory
    ax.plot(
        linear_radii,
        linear_sigma,
        "-",
        label="Linear theory (CAMB)",
        color="#C76A4A",
        linewidth=2,
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
