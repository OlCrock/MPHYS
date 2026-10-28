import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import yt
from scipy.stats import norm
from scipy.integrate import quad
import sys

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

base_path = Path(
    "/cephfs2/TMOX/legacy/GVD_C700_l100n2048_SLEGAC/dm_gadget/data"
)

snapshot_numbers = [0, 1, 3, 4, 6, 7, 9, 10, 12, 14, 15]
all_snapshot_numbers = snapshot_numbers.copy()
batch_mode = len(sys.argv) > 1

if batch_mode:
    requested_snap = int(sys.argv[1])
    if requested_snap not in all_snapshot_numbers:
        raise ValueError(f"Snapshot {requested_snap} is not in {all_snapshot_numbers}")
    snapshot_numbers = [requested_snap]
    snapshot_index = all_snapshot_numbers.index(requested_snap)

n_files_per_snapshot = 4
sphere_radius = 8
n_spheres = 1

plot_dir = Path("plots")
plot_dir.mkdir(exist_ok=True)

np.random.seed(42)


def calculate_overdensities(ds, sphere_radius, n_spheres=1):
    """Calculate density contrast in randomly placed spheres."""

    box_size = ds.domain_width[0].to("Mpc/h").value
    all_data = ds.all_data()
    n_total = all_data["PartType1", "particle_position_x"].size

    box_volume = box_size**3
    mean_density = n_total / box_volume
    sphere_volume = (4.0 / 3.0) * np.pi * sphere_radius**3

    print(f"Box size:       {box_size:.3f} Mpc/h")
    print(f"Total particles: {n_total}")
    print(f"Mean density:   {mean_density:.3e} particles/(Mpc/h)^3")

    centres = np.random.uniform(0, box_size, size=(n_spheres, 3))
    overdensities = []

    for i, centre in enumerate(centres):
        sp = ds.sphere(centre, (sphere_radius, "Mpc/h"))
        n_sphere = sp["PartType1", "particle_position_x"].size

        density = n_sphere / sphere_volume
        delta = density / mean_density - 1.0
        overdensities.append(delta)

        if (i + 1) % 100 == 0:
            print(f"Calculated {i + 1}/{n_spheres} spheres")

    return np.asarray(overdensities)


def linear_sigma(H0, Omega_m, Omega_L, z, sigma0_sq):
    """Return linear-theory variance evolved from z=0."""

    def H(a):
        return H0 * np.sqrt(Omega_m / a**3 + Omega_L)

    def D(a):
        integrand = lambda x: 1.0 / (x**3 * H(x)**3)
        return H(a) * quad(integrand, 0, a)[0]

    z = np.atleast_1d(z).astype(float)
    a = 1.0 / (1.0 + z)
    D_ratio = np.array([D(ai) for ai in a]) / D(1.0)

    return D_ratio**2 * sigma0_sq


def plot_sigma_by_z(redshifts, sigmas, sigma_theory=None):
    fig, ax = plt.subplots(figsize=(8.5, 6.0))

    z = np.asarray(redshifts, dtype=float)
    sigma_sim = np.asarray(sigmas, dtype=float)
    order = np.argsort(z)

    ax.plot(
        z[order], sigma_sim[order],
        marker="o", linestyle="-", linewidth=2.2, markersize=6,
        color="#2C5F8A", label="Simulation",
    )

    if sigma_theory is not None:
        sigma_theory = np.asarray(sigma_theory, dtype=float)
        ax.plot(
            z[order], sigma_theory[order],
            marker="s", linestyle="--", linewidth=2.0, markersize=5,
            color="#C76A4A",
            label="Linear growth reference (normalised to sigma8)",
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
        plot_dir / f"sigma_by_z_R{sphere_radius}_GADGET2048.png",
        dpi=200,
    )
    plt.close(fig)


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
    snapshot_file = snapdir / f"snapshot_{snap:03d}.0.hdf5"

    ds = yt.load(
        str(snapshot_file),
        unit_base={"length": (1.0, "Mpc/h")},
        bounding_box=np.array([[0, 100], [0, 100], [0, 100]]),
    )

    ds.force_periodicity()

    redshift = float(ds.current_redshift)
    redshifts.append(redshift)

    print(ds)
    print("Redshift:", redshift)
    print("Code length:", ds.length_unit)
    print("Domain width:", ds.domain_width)
    print(
        "4 Mpc/h in code units:",
        ds.quan(4.0, "Mpc/h").to("code_length"),
    )

    # Advance the RNG to the same state as the original sequential run.
    # Each previous snapshot consumed n_spheres * 3 uniform draws.
    if batch_mode:
        np.random.uniform(
            0, 1, size=(snapshot_index * n_spheres, 3)
        )

    overdensities = calculate_overdensities(
        ds, sphere_radius=sphere_radius, n_spheres=n_spheres
    )

    mean_delta = np.mean(overdensities)
    std_delta = np.std(overdensities)

    means.append(mean_delta)
    stds.append(std_delta)

    mu, sigma = norm.fit(overdensities)
    sigmas.append(sigma)

    print(f"Mean delta:      {mean_delta:.4f}")
    print(f"Std delta:       {std_delta:.4f}")
    print(f"Gaussian mu:     {mu:.4f}")
    print(f"Gaussian sigma:  {sigma:.4f}")

    fig, ax = plt.subplots(figsize=(8.0, 6.0))

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
        gaussian = norm.pdf(delta_range, mu, sigma)
        ax.plot(
            delta_range, gaussian,
            linewidth=2.2, color="#C44E52",
            label=(
                rf"Gaussian fit: $\mu={mu:.3f}$, "
                rf"$\sigma={sigma:.3f}$, $z={redshift:.2f}$"
            ),
        )

    ax.axvline(0, linestyle="--", linewidth=1.2, color="#2E2E2E")
    ax.set_xlabel(r"Overdensity $\delta$")
    ax.set_ylabel("Probability density")
    ax.set_title(rf"$R={sphere_radius}\,h^{{-1}}$ Mpc")
    ax.legend(loc="best", frameon=True, fontsize=9)
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.savefig(
        plot_dir / f"snapshot_{snap:03d}_overdensity_GADGET2048_LARGETEST.png",
        dpi=200,
    )
    plt.close(fig)

    # Each batch task writes its own result file.
    if batch_mode:
        result_path = plot_dir / f"variance_snapshot_{snap:03d}.csv"
        np.savetxt(
            result_path,
            [[snap, redshift, sigma, sigma**2]],
            delimiter=",",
            header="snapshot,redshift,sigma_delta,variance",
            comments="",
            fmt=["%d", "%.8e", "%.8e", "%.8e"],
        )
        print("Snapshot result saved to:", result_path.resolve())


if not batch_mode:
    varr_theory = linear_sigma(
        H0=67.4,
        Omega_m=0.315,
        Omega_L=0.685,
        z=np.array(redshifts),
        sigma0_sq=0.811**2,
    )

    sigma_theory = np.sqrt(varr_theory)

    plot_sigma_by_z(redshifts, sigmas, sigma_theory=sigma_theory)

    variance = np.asarray(sigmas)**2
    data = np.column_stack((redshifts, sigmas, variance))
    header = "redshift,sigma_delta,variance"
    csv_path = plot_dir / f"variance_by_z_R{sphere_radius}_GADGET2048.csv"
    np.savetxt(
        csv_path, data, delimiter=",", header=header,
        comments="", fmt="%.8e",
    )
    print("Variance CSV saved to:", csv_path.resolve())

print("\nFinished. Plots saved to:", plot_dir.resolve())
