import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import yt
from scipy.stats import norm
from scipy.integrate import quad

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "figure.figsize": (8, 6),
    "figure.dpi": 200,
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
})

# LEGACY simulation
base_path = Path(
    "/cephfs2/TMOX/legacy/GVD_C700_l100n2048_SLEGAC/dm_gadget/data"
)

# Start with one snapshot and a small test
snapshot_numbers = [0, 1, 12, 25, 49, 62, 77, 83, 87, 101]
sphere_radius = 8.0  # Mpc/h
n_spheres = 100

plot_dir = Path("plots")
plot_dir.mkdir(exist_ok=True)
np.random.seed(42)


def get_total_particle_count(snapshot_file):
    """Read the global dark-matter particle count from the GADGET header."""
    with h5py.File(snapshot_file, "r") as f:
        header = f["Header"].attrs
        n_total = int(header["NumPart_Total"][1])
        high_word = header.get("NumPart_Total_HighWord")

        if high_word is not None:
            n_total += int(high_word[1]) << 32

    return n_total


def calculate_overdensities(ds, sphere_radius, n_total, n_spheres=1):
    """Calculate density contrasts in randomly placed spheres."""
    box_size = ds.domain_width[0].to("Mpc/h").value
    box_volume = box_size**3
    mean_density = n_total / box_volume
    sphere_volume = (4.0 / 3.0) * np.pi * sphere_radius**3

    print(f"Box size:        {box_size:.3f} Mpc/h", flush=True)
    print(f"Total particles: {n_total:,}", flush=True)
    print(
        f"Mean density:    {mean_density:.3e} particles/(Mpc/h)^3",
        flush=True,
    )
    print(f"Sphere radius:   {sphere_radius:.3f} Mpc/h", flush=True)

    centres = np.random.uniform(0, box_size, size=(n_spheres, 3))
    overdensities = np.empty(n_spheres)

    for i, centre in enumerate(centres):
        sp = ds.sphere(centre, (sphere_radius, "Mpc/h"))
        n_sphere = sp["PartType1", "particle_position_x"].size

        density = n_sphere / sphere_volume
        overdensities[i] = density / mean_density - 1.0

        if (i + 1) % 10 == 0 or i + 1 == n_spheres:
            print(
                f"Calculated {i + 1}/{n_spheres} spheres",
                flush=True,
            )

    return overdensities


def linear_sigma(H0, Omega_m, Omega_L, z, sigma0_sq):
    """Return linear-theory variance evolved from redshift zero."""

    def H(a):
        return H0 * np.sqrt(Omega_m / a**3 + Omega_L)

    def D(a):
        integrand = lambda x: 1.0 / (x**3 * H(x)**3)
        return H(a) * quad(integrand, 0, a)[0]

    z = np.atleast_1d(z).astype(float)
    a = 1.0 / (1.0 + z)
    D_ratio = np.array([D(ai) for ai in a]) / D(1.0)

    return D_ratio**2 * sigma0_sq


def plot_overdensity_histogram(overdensities, snapshot_number, redshift):
    """Plot the overdensity distribution for one snapshot."""
    mu, sigma = norm.fit(overdensities)

    fig, ax = plt.subplots()
    ax.hist(
        overdensities,
        bins="auto",
        density=True,
        alpha=0.65,
        label="Measured overdensities",
    )

    x_min, x_max = ax.get_xlim()
    x = np.linspace(x_min, x_max, 500)
    ax.plot(
        x,
        norm.pdf(x, mu, sigma),
        linewidth=2,
        label=fr"Gaussian fit: $\mu={mu:.3f}$, $\sigma={sigma:.3f}$",
    )

    ax.set_xlabel(r"Density contrast $\delta$")
    ax.set_ylabel("Probability density")
    ax.set_title(
        f"LEGACY snapshot {snapshot_number:03d}, z={redshift:.3f}"
    )
    ax.legend()
    fig.tight_layout()

    output_file = plot_dir / f"overdensity_snapshot_{snapshot_number:03d}.png"
    fig.savefig(output_file)
    plt.close(fig)

    print(f"Saved {output_file}", flush=True)
    return mu, sigma


def plot_sigma_by_redshift(redshifts, stds):
    """Plot measured overdensity scatter against redshift."""
    redshifts = np.asarray(redshifts)
    stds = np.asarray(stds)

    order = np.argsort(redshifts)
    redshifts = redshifts[order]
    stds = stds[order]

    fig, ax = plt.subplots()
    ax.plot(
        redshifts,
        stds,
        "o-",
        label="LEGACY measured standard deviation",
    )
    ax.set_xlabel("Redshift")
    ax.set_ylabel(r"Standard deviation of $\delta$")
    ax.set_title(
        f"Overdensity scatter, sphere radius = {sphere_radius:g} Mpc/h"
    )
    ax.legend()
    fig.tight_layout()

    output_file = plot_dir / "sigma_by_redshift.png"
    fig.savefig(output_file)
    plt.close(fig)

    print(f"Saved {output_file}", flush=True)


def main():
    redshifts = []
    means = []
    stds = []
    fitted_sigmas = []

    for snap in snapshot_numbers:
        print("\n" + "=" * 60, flush=True)
        print(f"SNAPSHOT {snap:03d}", flush=True)
        print("=" * 60, flush=True)

        snapdir = base_path / f"snapdir_{snap:03d}"
        snapshot_file = snapdir / f"snapshot_{snap:03d}.0.hdf5"

        if not snapshot_file.is_file():
            raise FileNotFoundError(
                f"Snapshot file not found: {snapshot_file}"
            )

        # Read the particle count without loading all particle coordinates.
        n_total = get_total_particle_count(snapshot_file)
        print(f"Global particle count: {n_total:,}", flush=True)

        print("Loading snapshot with yt...", flush=True)
        ds = yt.load(
            str(snapshot_file),
            unit_base={"length": (1.0, "Mpc/h")},
            bounding_box=np.array([
                [0.0, 100.0],
                [0.0, 100.0],
                [0.0, 100.0],
            ]),
        )
        ds.force_periodicity()

        redshift = float(ds.current_redshift)
        redshifts.append(redshift)

        print(f"Redshift: {redshift:.6f}", flush=True)
        print(f"Domain width: {ds.domain_width}", flush=True)
        print("Starting sphere calculations...", flush=True)

        overdensities = calculate_overdensities(
            ds,
            sphere_radius=sphere_radius,
            n_total=n_total,
            n_spheres=n_spheres,
        )

        mean_delta = np.mean(overdensities)
        std_delta = np.std(overdensities)
        mu, sigma = plot_overdensity_histogram(
            overdensities, snap, redshift
        )

        means.append(mean_delta)
        stds.append(std_delta)
        fitted_sigmas.append(sigma)

        print(f"Mean delta:       {mean_delta:.6f}", flush=True)
        print(f"Std delta:        {std_delta:.6f}", flush=True)
        print(f"Gaussian mu:      {mu:.6f}", flush=True)
        print(f"Gaussian sigma:   {sigma:.6f}", flush=True)

        del ds
        print(f"Finished snapshot {snap:03d}", flush=True)

    plot_sigma_by_redshift(redshifts, stds)

    # Save the measured summary statistics.
    results_file = plot_dir / "overdensity_summary.txt"
    results = np.column_stack([
        snapshot_numbers,
        redshifts,
        means,
        stds,
        fitted_sigmas,
    ])

    np.savetxt(
        results_file,
        results,
        header=(
            "snapshot redshift mean_delta std_delta gaussian_sigma"
        ),
        fmt=["%d", "%.8e", "%.8e", "%.8e", "%.8e"],
    )

    print(f"Saved {results_file}", flush=True)
    print("Analysis complete.", flush=True)


if __name__ == "__main__":
    main()