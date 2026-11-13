import numpy as np
import h5py
import matplotlib.pyplot as plt

# Load geometry and metadata while both HDF5 files remain open
with h5py.File("optimised/output/centres.h5", "r") as centres_file, \
     h5py.File("optimised/output/variances.h5", "r") as variances_file:
    centres = centres_file["centres"][:]
    radii = variances_file["radii"][:]
    redshifts = variances_file["redshift"][:]
    snapshots = variances_file["snapshots"][:]
    counts_group = variances_file["counts"]

    # Pair separations between all centres in the box
    box_size = 100.0
    pair_i, pair_j = np.triu_indices(len(centres), k=1)
    separation = np.abs(centres[pair_i] - centres[pair_j])
    separation = np.minimum(separation, box_size - separation)
    r = np.sqrt(np.sum(separation**2, axis=1))

    # Radial-bin definition for the spatial covariance curve
    bins = np.linspace(0.0, 50.0, 26)
    r_mid = (bins[:-1] + bins[1:]) / 2.0

    # Store covariance as [redshift, radius, separation_bin]
    N_total = 2048**3
    n_redshifts = len(redshifts)
    n_radii = len(radii)
    full_covariance = np.full((n_redshifts, n_radii, len(r_mid)), np.nan)

    for z_index, snapshot in enumerate(snapshots):
        snapshot_name = f"snap{int(snapshot):03d}"
        counts = counts_group[snapshot_name][:]

        for radius_index, R in enumerate(radii):
            volume = (4.0 / 3.0) * np.pi * R**3
            delta = counts[:, radius_index] * 100.0**3 / (N_total * volume) - 1.0
            products = delta[pair_i] * delta[pair_j]

            covariance_hist, _ = np.histogram(r, bins=bins, weights=products)
            pair_count, _ = np.histogram(r, bins=bins)

            full_covariance[z_index, radius_index, :] = np.divide(
                covariance_hist,
                pair_count,
                out=np.full(len(r_mid), np.nan),
                where=pair_count > 0,
            )

# Plot the representative case at z ~ 99 and R = 8 Mpc/h
reference_snapshot_index = np.argmin(np.abs(redshifts - 99.0))
reference_radius_index = np.argmin(np.abs(radii - 8.0))
reference_cov = full_covariance[reference_snapshot_index, reference_radius_index, :]

# calculate the error on the variance

# Calculate the Gaussian error on the variance

def variance_error(counts, N_total, volume, r, pair_i, pair_j, bins):

    delta = counts * 100.0**3 / (N_total * volume) - 1.0

    N = len(delta)

    # Two-point covariance
    products = delta[pair_i] * delta[pair_j]

    xi_sum, _ = np.histogram(r, bins=bins, weights=products)
    pair_count, _ = np.histogram(r, bins=bins)

    xi = np.divide(
        xi_sum,
        pair_count,
        out=np.full(len(bins) - 1, np.nan),
        where=pair_count > 0
    )

    r_mid = (bins[:-1] + bins[1:]) / 2

    valid = np.isfinite(xi)

    xi_pairs = np.interp(r, r_mid[valid], xi[valid])

    # Gaussian approximation to variance of the variance
    diagonal = np.var(delta)

    covariance_sum = N * diagonal**2 + 2 * np.sum(xi_pairs**2)

    return np.sqrt(2 * covariance_sum / N**2)


# Select z ~ 99 and R = 8 Mpc/h
reference_snapshot_index = np.argmin(np.abs(redshifts - 99.0))
reference_radius_index = np.argmin(np.abs(radii - 8.0))

snapshot = int(snapshots[reference_snapshot_index])
R = radii[reference_radius_index]
volume = (4.0 / 3.0) * np.pi * R**3

# Load counts for the correct snapshot
with h5py.File("optimised/output/variances.h5", "r") as f:
    counts = f[f"counts/snap{snapshot:03d}"][:]

error = variance_error(
    counts[:, reference_radius_index],
    N_total,
    volume,
    r,
    pair_i,
    pair_j,
    bins
)

print(f"Snapshot: {snapshot}")
print(f"Redshift: {redshifts[reference_snapshot_index]:.4f}")
print(f"Radius: {R:.2f} Mpc/h")
print(f"Gaussian error on variance: {error:.6e}")