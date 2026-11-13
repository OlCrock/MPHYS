import numpy as np
import h5py

# --------------------------------------------------
# Configuration
# --------------------------------------------------

CENTRES_FILE = "optimised/output/centres.h5"
VARIANCES_FILE = "optimised/output/variances.h5"

BOX = 100.0
N_TOTAL = 2048**3
N_BINS = 30


# --------------------------------------------------
# Gaussian error on the variance
# --------------------------------------------------

def variance_error(counts, N_total, volume, r, pair_i, pair_j, bins):

    delta = counts * BOX**3 / (N_total * volume) - 1.0

    N = len(delta)

    # Calculate spatial covariance
    products = delta[pair_i] * delta[pair_j]

    xi_sum, _ = np.histogram(
        r, bins=bins, weights=products
    )

    pair_count, _ = np.histogram(r, bins=bins)

    xi = np.divide(
        xi_sum,
        pair_count,
        out=np.full(len(bins) - 1, np.nan),
        where=pair_count > 0
    )

    r_mid = (bins[:-1] + bins[1:]) / 2

    valid = np.isfinite(xi)

    if np.sum(valid) < 2:
        return np.nan

    # Interpolate covariance to individual pair separations
    xi_pairs = np.interp(
        r, r_mid[valid], xi[valid]
    )

    # Construct covariance matrix
    C = np.zeros((N, N))

    C[pair_i, pair_j] = xi_pairs
    C[pair_j, pair_i] = xi_pairs

    # Diagonal terms
    C[np.diag_indices(N)] = np.var(delta)

    # Account for subtraction of the sample mean
    row_sums = np.sum(C, axis=1)

    trace_C2 = np.sum(C**2)
    ones_C2_ones = np.sum(row_sums**2)
    ones_C_ones = np.sum(C)

    trace_PCP_C = (
        trace_C2
        - 2.0 * ones_C2_ones / N
        + ones_C_ones**2 / N**2
    )

    # Gaussian approximation to variance of the variance estimator
    error = np.sqrt(
        2.0 * max(trace_PCP_C, 0.0) / N**2
    )

    return error


# --------------------------------------------------
# Load centres and calculate pair separations
# --------------------------------------------------

with h5py.File(CENTRES_FILE, "r") as f:
    centres = f["centres"][:]

pair_i, pair_j = np.triu_indices(len(centres), k=1)

separation = np.abs(centres[pair_i] - centres[pair_j])

separation = np.minimum(
    separation,
    BOX - separation
)

r = np.sqrt(np.sum(separation**2, axis=1))

# Cover the full range of periodic minimum-image separations
bins = np.linspace(
    0.0,
    np.sqrt(3.0) * BOX / 2.0 + 1e-8,
    N_BINS + 1
)


# --------------------------------------------------
# Calculate errors for every snapshot and radius
# --------------------------------------------------

with h5py.File(VARIANCES_FILE, "r+") as f:

    radii = f["radii"][:]
    redshifts = f["redshift"][:]
    snapshots = f["snapshots"][:]

    counts_group = f["counts"]

    errors = np.full(
        (len(snapshots), len(radii)),
        np.nan
    )

    for z_index, snapshot in enumerate(snapshots):

        snapshot_name = f"snap{int(snapshot):03d}"
        counts = counts_group[snapshot_name][:]

        for radius_index, R in enumerate(radii):

            volume = (4.0 / 3.0) * np.pi * R**3

            errors[z_index, radius_index] = variance_error(
                counts[:, radius_index],
                N_TOTAL,
                volume,
                r,
                pair_i,
                pair_j,
                bins
            )

        print(
            f"Finished snapshot {int(snapshot)} "
            f"(z = {redshifts[z_index]:.4f})"
        )

    # --------------------------------------------------
    # Save to HDF5
    # --------------------------------------------------

    if "variance_error_gaussian" in f:
        del f["variance_error_gaussian"]

    dataset = f.create_dataset(
        "variance_error_gaussian",
        data=errors
    )

    dataset.attrs["description"] = (
        "Gaussian approximation to the standard error "
        "of the variance, accounting for spatial covariance "
        "between sphere centres."
    )

    dataset.attrs["method"] = (
        "Two-point covariance with Gaussian four-point approximation."
    )

print("\nAll errors calculated and saved.")
print("Dataset: variance_error_gaussian")
print("Shape:", errors.shape)