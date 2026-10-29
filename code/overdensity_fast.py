import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import yt
import os
import sys
from scipy.stats import norm
from scipy.integrate import quad

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "figure.figsize": (8, 6),
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "axes.linewidth": 1.2,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "legend.frameon": False,
    "savefig.dpi": 200,
    "savefig.bbox": "tight"
})

base_path = Path("/cephfs2/TMOX/legacy/GVD_C700_l100n2048_SLEGAC/dm_gadget/data")
snapshot_numbers = [0, 1, 3, 4, 6, 7, 9, 10, 12, 14, 15]
n_files_per_snapshot = 4
sphere_radius = 10.0
n_spheres = 100

plot_dir = Path("plots")
plot_dir.mkdir(parents=True, exist_ok=True)
results_dir = plot_dir / ".overdensity_results"
results_dir.mkdir(parents=True, exist_ok=True)

np.random.seed(42)


def calculate_overdensities(ds, sphere_radius, n_spheres=1):
    box_size = ds.domain_width[0].to("Mpc/h").value
    ad = ds.all_data()
    n_particles = ad[("PartType1", "particle_position_x")].size
    mean_density = n_particles / box_size**3
    sphere_volume = (4.0 / 3.0) * np.pi * sphere_radius**3
    overdensities = []

    centres = np.random.uniform(0, box_size, size=(n_spheres, 3))

    for i, centre in enumerate(centres):
        sphere = ds.sphere(centre, (sphere_radius, "Mpc/h"))
        n_in_sphere = sphere[("PartType1", "particle_position_x")].size
        density = n_in_sphere / sphere_volume
        overdensity = density / mean_density - 1.0
        overdensities.append(overdensity)

        if (i + 1) % 100 == 0:
            print(f"Completed {i + 1}/{n_spheres} spheres", flush=True)

    return np.asarray(overdensities)


def linear_sigma(H0, Omega_m, Omega_L, z, sigma0_sq):
    def H(a):
        return H0 * np.sqrt(Omega_m / a**3 + Omega_L)

    def growth(a):
        integral = quad(
            lambda x: 1.0 / (x**3 * H(x)**3),
            0.0,
            a,
            limit=200
        )[0]
        return H(a) * integral

    a = 1.0 / (1.0 + z)
    D = growth(a)
    D0 = growth(1.0)
    return (D / D0)**2 * sigma0_sq


def plot_sigma_by_z(redshifts, sigmas, sigma_theory=None):
    fig, ax = plt.subplots()

    ax.plot(
        redshifts, sigmas,
        marker="o", linestyle="-",
        label="LEGACY simulation"
    )

    if sigma_theory is not None:
        ax.plot(
            redshifts, sigma_theory,
            linestyle="--",
            label="Linear theory"
        )

    ax.set_xlabel("Redshift, z")
    ax.set_ylabel(r"Variance, $\sigma^2$")
    ax.set_title(
        f"Overdensity variance vs redshift "
        f"(R = {sphere_radius} Mpc/h)"
    )
    ax.legend()
    fig.savefig(
        plot_dir / f"sigma_by_z_R{sphere_radius}_GADGET2048.png"
    )
    plt.close(fig)


def process_snapshot(snap):
    print(f"Processing snapshot {snap}", flush=True)

    snapshot_path = (
        base_path
        / f"snapdir_{snap:03d}"
        / f"snapshot_{snap:03d}.0.hdf5"
    )

    ds = yt.load(
        str(snapshot_path),
        unit_base={"length": (1.0, "Mpc/h")},
        bounding_box=np.array([
            [0.0, 100.0],
            [0.0, 100.0],
            [0.0, 100.0]
        ])
    )
    ds.force_periodicity()

    redshift = float(ds.current_redshift)
    overdensities = calculate_overdensities(
        ds, sphere_radius, n_spheres
    )

    mean_delta = float(np.mean(overdensities))
    std_delta = float(np.std(overdensities, ddof=1))
    fit_mean, fit_sigma = norm.fit(overdensities)
    variance = float(fit_sigma**2)

    result_path = results_dir / f"snapshot_{snap:03d}.npz"
    np.savez(
        result_path,
        snapshot=snap,
        redshift=redshift,
        mean_delta=mean_delta,
        std_delta=std_delta,
        fit_mean=float(fit_mean),
        fit_sigma=float(fit_sigma),
        variance=variance
    )

    fig, ax = plt.subplots()
    ax.hist(
        overdensities,
        bins=20,
        density=True,
        alpha=0.65,
        label="Simulated overdensities"
    )

    x = np.linspace(
        np.min(overdensities),
        np.max(overdensities),
        300
    )
    ax.plot(
        x, norm.pdf(x, fit_mean, fit_sigma),
        linestyle="--",
        label="Gaussian fit"
    )
    ax.set_xlabel(r"Overdensity, $\delta$")
    ax.set_ylabel("Probability density")
    ax.set_title(f"Snapshot {snap:03d}, z = {redshift:.3f}")
    ax.legend()
    fig.savefig(
        plot_dir
        / f"snapshot_{snap:03d}_overdensity_GADGET2048_LARGETEST.png"
    )
    plt.close(fig)

    print(
        f"Snapshot {snap}: z={redshift:.4f}, "
        f"mean={mean_delta:.5f}, "
        f"std={std_delta:.5f}, variance={variance:.5f}",
        flush=True
    )


def combine_results():
    result_files = [
        results_dir / f"snapshot_{snap:03d}.npz"
        for snap in snapshot_numbers
    ]

    missing = [path.name for path in result_files if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing snapshot results: " + ", ".join(missing)
        )

    results = [np.load(path) for path in result_files]
    redshifts = np.array([r["redshift"].item() for r in results])
    means = np.array([r["mean_delta"].item() for r in results])
    stds = np.array([r["std_delta"].item() for r in results])
    variances = np.array([r["variance"].item() for r in results])

    order = np.argsort(redshifts)
    redshifts = redshifts[order]
    means = means[order]
    stds = stds[order]
    variances = variances[order]

    sigma_theory = np.array([
        linear_sigma(
            H0=67.4,
            Omega_m=0.315,
            Omega_L=0.685,
            z=z,
            sigma0_sq=0.811**2
        )
        for z in redshifts
    ])

    plot_sigma_by_z(redshifts, variances, sigma_theory)

    output_csv = plot_dir / (
        f"variance_by_z_R{sphere_radius}_GADGET2048.csv"
    )
    np.savetxt(
        output_csv,
        np.column_stack((redshifts, means, stds, variances, sigma_theory)),
        delimiter=",",
        header="redshift,mean_delta,std_delta,variance,theory_variance",
        comments=""
    )

    print(f"Saved combined plot to plots/", flush=True)
    print(f"Saved results to {output_csv}", flush=True)


if __name__ == "__main__":
    if "--combine" in sys.argv:
        combine_results()
    else:
        array_task = os.environ.get("SLURM_ARRAY_TASK_ID")

        if array_task is not None:
            task_index = int(array_task)
            if task_index < 0 or task_index >= len(snapshot_numbers):
                raise IndexError(f"Invalid array task index: {task_index}")
            process_snapshot(snapshot_numbers[task_index])
        else:
            for snap in snapshot_numbers:
                process_snapshot(snap)
            combine_results()
