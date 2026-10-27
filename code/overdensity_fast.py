
import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial import cKDTree
from scipy.stats import norm

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "figure.figsize": (8, 6),
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "axes.linewidth": 1.2,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "-",
    "legend.frameon": True,
    "legend.framealpha": 0.95,
    "legend.fancybox": True,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
    "axes.facecolor": "white",
    "xtick.direction": "out",
    "ytick.direction": "out",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# LEGACY GADGET simulation data
base_path = Path(
    "/cephfs2/TMOX/legacy/GVD_C700_l100n2048_SLEGAC/dm_gadget/data"
)

snapshot_numbers = [0, 1, 3, 4, 6, 7, 9, 10, 12, 14, 15]
sphere_radius = 10.0  # Mpc/h
n_spheres = 100
plot_dir = Path("plots")
plot_dir.mkdir(exist_ok=True)

# Reproducible sphere centres, reused across snapshots
rng = np.random.default_rng(42)


def get_snapshot_files(snapdir):
    """Return all HDF5 chunks for a snapshot."""
    files = sorted(snapdir.glob("snapshot_*.hdf5"))
    if not files:
        raise FileNotFoundError(f"No HDF5 files found in {snapdir}")
    return files


def get_snapshot_metadata(first_file):
    """Read snapshot metadata without loading particle coordinates."""
    with h5py.File(first_file, "r") as h:
        header = h["Header"].attrs
        box_size = float(header["BoxSize"])
        scale_factor = float(header["Time"])
        redshift = 1.0 / scale_factor - 1.0
        hubble_param = float(header["HubbleParam"])

    return box_size, redshift, hubble_param


def calculate_overdensities(snapdir, sphere_radius, centres):
    """
    Count particles in periodic spheres, processing one HDF5 file at a time.

    Each file's coordinates are indexed temporarily. Counts are accumulated
    across all files, so every particle is included exactly once.
    """
    files = get_snapshot_files(snapdir)
    box_size, redshift, hubble_param = get_snapshot_metadata(files[0])

    counts = np.zeros(len(centres), dtype=np.int64)
    n_total = 0

    print(f"Redshift: {redshift:.3f}")
    print(f"Box size: {box_size:.3f} Mpc/h")
    print(f"Hubble parameter h: {hubble_param:.3f}")
    print(f"HDF5 files: {len(files)}")

    for file_num, filename in enumerate(files, start=1):
        print(f"Processing file {file_num}/{len(files)}: {filename.name}")

        with h5py.File(filename, "r") as h:
            coords_ds = h["PartType1/Coordinates"]
            n_file = coords_ds.shape[0]

            # Read only this file's coordinates into memory.
            coords = np.asarray(coords_ds[:], dtype=np.float64)
            coords %= box_size

            # Build a periodic spatial index for this file only.
            tree = cKDTree(coords, boxsize=box_size)

            # Count particles from this file inside every sphere.
            counts += tree.query_ball_point(
                centres,
                r=sphere_radius,
                return_length=True,
            )

            n_total += n_file

            del tree, coords

    box_volume = box_size**3
    sphere_volume = (4.0 / 3.0) * np.pi * sphere_radius**3
    expected_count = n_total * sphere_volume / box_volume

    overdensities = counts / expected_count - 1.0

    print(f"Total particles: {n_total:,}")
    print(f"Mean count per sphere: {expected_count:,.1f}")

    return overdensities, redshift


def plot_overdensity_histogram(overdensities, redshift, snap):
    """Save the overdensity histogram and Gaussian fit."""
    mu, sigma = norm.fit(overdensities)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(
        overdensities,
        bins=min(100, max(5, n_spheres)),
        density=True,
        alpha=0.65,
        color="#4C72B0",
        edgecolor="white",
        linewidth=0.6,
        label="Sphere measurements",
    )

    delta_range = np.linspace(
        overdensities.min() - 0.1,
        overdensities.max() + 0.1,
        500,
    )

    if sigma > 0:
        ax.plot(
            delta_range,
            norm.pdf(delta_range, mu, sigma),
            linewidth=2.2,
            color="#C44E52",
            label=rf"Gaussian fit: $\mu={mu:.3f}$, $\sigma={sigma:.3f}$",
        )

    ax.axvline(0, linestyle="--", linewidth=1.2, color="#2E2E2E")
    ax.set_xlabel(r"Overdensity $\delta$")
    ax.set_ylabel("Probability density")
    ax.set_title(
        rf"$R={sphere_radius}\,h^{{-1}}$ Mpc, $z={redshift:.2f}$"
    )
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.savefig(
        plot_dir / f"snapshot_{snap:03d}_overdensity_GADGET2048_HDF5.png",
        dpi=200,
    )
    plt.close(fig)

    return mu, sigma


def plot_sigma_by_z(redshifts, sigmas):
    """Plot measured overdensity standard deviation against redshift."""
    z = np.asarray(redshifts, dtype=float)
    sigma_sim = np.asarray(sigmas, dtype=float)
    order = np.argsort(z)

    fig, ax = plt.subplots(figsize=(8.5, 6.0))
    ax.plot(
        z[order],
        sigma_sim[order],
        marker="o",
        linestyle="-",
        linewidth=2.2,
        markersize=6,
        color="#2C5F8A",
        label="Simulation",
    )

    ax.set_xlabel("Redshift z")
    ax.set_ylabel(r"Standard deviation $\sigma_\delta$")
    ax.set_title(
        rf"Overdensity amplitude for $R={sphere_radius}\,h^{{-1}}$ Mpc"
    )
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    plt.tight_layout()
    plt.savefig(
        plot_dir / f"sigma_by_z_R{sphere_radius}_GADGET2048_HDF5.png",
        dpi=200,
    )
    plt.close(fig)


def save_variance_csv(redshifts, sigmas):
    """Save redshift, standard deviation and variance to CSV."""
    sigmas = np.asarray(sigmas)
    variance = sigmas**2
    data = np.column_stack((redshifts, sigmas, variance))

    csv_path = (
        plot_dir / f"variance_by_z_R{sphere_radius}_GADGET2048_HDF5.csv"
    )
    np.savetxt(
        csv_path,
        data,
        delimiter=",",
        header="redshift,sigma_delta,variance",
        comments="",
        fmt="%.8e",
    )
    print("Variance CSV saved to:", csv_path.resolve())


# Generate the same sphere centres for every snapshot.
# This assumes all snapshots share the same 100 Mpc/h box.
first_snapdir = base_path / f"snapdir_{snapshot_numbers[0]:03d}"
first_files = get_snapshot_files(first_snapdir)
box_size, _, _ = get_snapshot_metadata(first_files[0])
centres = rng.uniform(0.0, box_size, size=(n_spheres, 3))

redshifts = []
means = []
stds = []
sigmas = []

for snap in snapshot_numbers:
    print()
    print("=" * 60)
    print(f"SNAPSHOT {snap:03d}")
    print("=" * 60)

    snapdir = base_path / f"snapdir_{snap:03d}"

    overdensities, redshift = calculate_overdensities(
        snapdir,
        sphere_radius=sphere_radius,
        centres=centres,
    )

    redshifts.append(redshift)

    mean_delta = np.mean(overdensities)
    std_delta = np.std(overdensities)
    mu, sigma = plot_overdensity_histogram(
        overdensities, redshift, snap
    )

    means.append(mean_delta)
    stds.append(std_delta)
    sigmas.append(sigma)

    print(f"Mean delta:      {mean_delta:.4f}")
    print(f"Std delta:       {std_delta:.4f}")
    print(f"Gaussian mu:     {mu:.4f}")
    print(f"Gaussian sigma:  {sigma:.4f}")

plot_sigma_by_z(redshifts, sigmas)
save_variance_csv(redshifts, sigmas)

print("\nFinished. Plots saved to:", plot_dir.resolve())

