import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import norm

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "figure.figsize": (8, 6),
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "legend.frameon": True,
        "legend.framealpha": 0.95,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
    }
)

plot_dir = Path("plots")
plot_dir.mkdir(exist_ok=True)


def generate_random_field(box_size=100.0, n_particles=20000):
    """Generate a homogeneous random point distribution in a cubic box."""
    positions = np.random.uniform(0.0, box_size, size=(n_particles, 3))
    return positions


def overdensity_histogram_random_field(
    positions,
    sphere_radius=8.0,
    n_spheres=1000,
    box_size=100.0,
):
    """Compute the overdensity distribution from random spheres in a uniform field."""
    sphere_volume = (4.0 / 3.0) * np.pi * sphere_radius**3
    mean_density = positions.shape[0] / box_size**3

    centres = np.random.uniform(0.0, box_size, size=(n_spheres, 3))
    overdensities = []

    for centre in centres:
        disp = positions - centre
        dist_sq = np.sum(disp**2, axis=1)
        n_in_sphere = np.count_nonzero(dist_sq <= sphere_radius**2)
        density = n_in_sphere / sphere_volume
        delta = density / mean_density - 1.0
        overdensities.append(delta)

    overdensities = np.asarray(overdensities)
    return overdensities


def plot_random_field_histogram():
    box_size = 100.0
    sphere_radius = 8.0
    n_particles = 25000
    n_spheres = 1500

    positions = generate_random_field(box_size=box_size, n_particles=n_particles)
    overdensities = overdensity_histogram_random_field(
        positions,
        sphere_radius=sphere_radius,
        n_spheres=n_spheres,
        box_size=box_size,
    )

    mu, sigma = norm.fit(overdensities)
    x = np.linspace(overdensities.min(), overdensities.max(), 500)
    pdf = norm.pdf(x, mu, sigma)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(
        overdensities,
        bins=50,
        density=True,
        alpha=0.75,
        color="#4C72B0",
        edgecolor="white",
        linewidth=0.6,
        label="Random field",
    )
    ax.plot(x, pdf, color="#C44E52", linewidth=2.2, label=rf"Gaussian fit: $\mu={mu:.3f}$, $\sigma={sigma:.3f}$")
    ax.axvline(0.0, linestyle="--", color="#2E2E2E", linewidth=1.2)
    ax.set_xlabel(r"Overdensity $\delta$")
    ax.set_ylabel("Probability density")
    ax.set_title(rf"Random field overdensity histogram for $R={sphere_radius}$ Mpc/h")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(plot_dir / "random_field_histogram.png", dpi=200)
    plt.show()

    print(f"Random field mean overdensity: {np.mean(overdensities):.6f}")
    print(f"Random field std overdensity: {np.std(overdensities):.6f}")
    print(f"Gaussian fit: mu={mu:.6f}, sigma={sigma:.6f}")


if __name__ == "__main__":
    plot_random_field_histogram()


