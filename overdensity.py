import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import yt
from scipy.stats import norm

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
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
    }
)

plot_dir = Path("plots")
plot_dir.mkdir(exist_ok=True)

"""
--------------------------------
base_path = Path("/disk12/legacy/GVD_C700_l100n256_SLEGAC/dm_gadget/data")

snapshot_numbers = [1, 8, 12]
#[1, 4, 8, 12, 15]

n_files_per_snapshot = 4

sphere_radius = 4.0     # Mpc/h
n_spheres = 10
--------------------------------
"""
# -------------------------------
base_path = Path("/cephfs2/brs/bw_mw/100Mpc_256/dm-only-L0")

snapshot_numbers = [0, 1, 2, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64]

sphere_radius = 8     # Mpc/h
n_spheres = 1000

# --------------------------------

def calculate_overdensities(ds, sphere_radius, n_spheres=1):

    """
    Place random spheres throughout the simulation box and
    calculate the density contrast delta in each sphere.
    """
    """
     --------------------------------
    # Box size in Mpc/h
    box_size = ds.domain_width[0].to("Mpc/h").value

    # Total number of particles
    all_data = ds.all_data()
    n_total = all_data["PartType1", "particle_position_x"].size

    # Volume of simulation box
    box_volume = box_size**3
    mean_density = n_total / box_volume

    print(f"Box size:       {box_size:.3f} Mpc/h")
    print(f"Total particles: {n_total}")
    print(f"Mean density:   {mean_density:.3e} particles/(Mpc/h)^3")

    # Random sphere centres
    centres = np.random.uniform(0, box_size, size=(n_spheres, 3))
    ----------------------- edit back when data is back 
    """ 

    # --------------------
    # Box size in Mpc/h
    box_size = 100.0

    # Total number of particles
    all_data = ds.all_data()
    n_total = all_data["all", "particle_position_x"].size

    # Volume of simulation box
    box_volume = box_size**3
    mean_density = n_total / box_volume

    print(f"Box size:       {box_size:.3f} Mpc/h")
    print(f"Total particles: {n_total}")
    print(f"Mean density:   {mean_density:.3e} particles/(Mpc/h)^3")

    # Random sphere centres
    centres = np.random.uniform(0, 1, size=(n_spheres, 3))
    # --------------------


    # Sphere volume
    sphere_volume = ((4.0 / 3.0) * np.pi * sphere_radius**3)

    overdensities = []

    # Loop over spheres
    for i, centre in enumerate(centres):

        sp = ds.sphere(centre, (sphere_radius/100.0, "code_length"))
        #n_sphere = sp["PartType1", "particle_position_x"].size
        n_sphere = sp["all", "particle_position_x"].size

        density = n_sphere / sphere_volume

        # Overdensity
        delta = density / mean_density - 1.0

        overdensities.append(delta)

        if (i + 1) % 100 == 0:
            print(f"Calculated {i + 1}/{n_spheres} spheres")

    overdensities = np.array(overdensities)

    return overdensities

redshifts = []
means = []
stds = []
sigmas = []

def plot_sigma_by_z():
    fig, ax = plt.subplots(figsize=(8.5, 6.0))

    sigma_values = sigmas if sigmas else stds
    ax.plot(
        redshifts,
        sigma_values,
        marker="o",
        linestyle="-",
        linewidth=2.2,
        markersize=6,
        color="#2C5F8A",
    )

    ax.set_xlabel("Redshift z")
    ax.set_ylabel(r"Standard deviation $\sigma$")
    ax.set_title(rf"Overdensity amplitude $\sigma_\delta$ for $R={sphere_radius}\,h^{{-1}}$ Mpc")
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.savefig(plot_dir / f"sigma_by_z_R{sphere_radius}_ENZOtest.png", dpi=200)

for snap in snapshot_numbers:

    print()
    print("=" * 60)
    print(f"SNAPSHOT {snap:03d}")
    print("=" * 60)
    """
    --------------------------------
    snapdir = base_path / f"snapdir_{snap:03d}"

    snapshot_file = (snapdir / f"snapshot_{snap:03d}.0.hdf5")

    with h5py.File(snapshot_file, "r") as f:
        print(f["Header"].attrs.keys())

    ds = yt.load(
    str(snapshot_file),
    unit_base={ "length": (1.0, "Mpc/h")},
    bounding_box=np.array([[0, 100], [0, 100], [0, 100]]))

    # Force periodicity on the dataset 
    ds.force_periodicity()
    --------------------------------
    """
    #-------------------------------
    snapshot_file = base_path / f"DD{snap:04d}" / f"DD{snap:04d}"

    ds = yt.load(str(snapshot_file))

    redshift = ds.current_redshift
    redshifts.append(redshift)

    ds.force_periodicity()
    print(ds)
    #-------------------------------
    print("code length:", ds.length_unit)
    print("domain width:", ds.domain_width)
    print("4 Mpc/h in code units:", ds.quan(4.0, "Mpc/h").to("code_length"))
    print("1 code_length in Mpc/h:", ds.quan(1.0, "code_length").to("Mpc/h"))
    overdensities = calculate_overdensities(ds, sphere_radius=sphere_radius, n_spheres=n_spheres)

    print(f"Mean delta:   "f"{np.mean(overdensities):.4f}")
    print(f"Std delta:    "f"{np.std(overdensities):.4f}")

    means.append(np.mean(overdensities))
    stds.append(np.std(overdensities))
    # FIT GAUSSIAN
    mu, sigma = norm.fit(overdensities)
    sigmas.append(sigma)

    print(f"Gaussian mu:      {mu:.4f}")
    print(f"Gaussian sigma:   {sigma:.4f}")


    # CREATE FIGURE
    fig, ax = plt.subplots(figsize=(8.0, 6.0))

    hist_color = "#4C72B0"
    fit_color = "#C44E52"
    ax.hist(
        overdensities,
        bins=100,
        density=True,
        alpha=0.65,
        color=hist_color,
        edgecolor="white",
        linewidth=0.6,
        label="Sphere measurements",
    )

    # Fit Gaussian curve
    delta_range = np.linspace(overdensities.min(), overdensities.max(), 500)

    gaussian = norm.pdf(delta_range, mu, sigma)

    ax.plot(
        delta_range,
        gaussian,
        linewidth=2.2,
        color=fit_color,
        label=rf"Gaussian fit: $\mu={mu:.3f}$, $\sigma={sigma:.3f}$, $z={redshift:.2f}$",
    )

    ax.axvline(0, linestyle="--", linewidth=1.2, color="#2E2E2E")

    ax.set_xlabel(r"Overdensity $\delta$")
    ax.set_ylabel("Probability density")
    ax.set_title(rf"$R={sphere_radius}\,h^{{-1}}$ Mpc")
    ax.legend(loc="best", frameon=True, fontsize=9)
    ax.grid(True, alpha=0.25)

    # SAVE
    plt.tight_layout()
    output_name = (f"snapshot_{snap:03d}_overdensity_ENZOtest.png")
    plt.savefig(plot_dir / output_name, dpi=200)

# PLOT SIGMA VS REDSHIFT
plot_sigma_by_z()

