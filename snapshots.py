import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import yt
from scipy.stats import norm

base_path = Path("/disk12/legacy/GVD_C700_l100n256_SLEGAC/dm_gadget/data")

snapshot_numbers = [15]
#[1, 4, 8, 12, 15]

n_files_per_snapshot = 4

slice_width = 2.0       # Mpc/h
sphere_radius = 5.0     # Mpc/h
n_spheres = 1000

def calculate_overdensities(ds, sphere_radius, n_spheres=1):

    """
    Place random spheres throughout the simulation box and
    calculate the density contrast delta in each sphere.
    """

    # Box size in Mpc/h
    box_size = ds.domain_width[0].to("Mpc/h").value

    # Total mass in the simulation
    all_data = ds.all_data()

    total_mass = all_data[
        "PartType1",
        "particle_mass"
    ].sum()

    total_mass = total_mass.to("Msun/h").value

    # Volume of simulation box
    box_volume = box_size**3

    mean_density = total_mass / box_volume

    print(f"Box size:       {box_size:.3f} Mpc/h")
    print(f"Total mass:     {total_mass:.3e} Msun/h")
    print(f"Mean density:   {mean_density:.3e} Msun/h/(Mpc/h)^3")

    # Random sphere centres

    centres = np.random.uniform(0, 100, size=(n_spheres, 3))

    # Sphere volume
    sphere_volume = (
        (4.0 / 3.0)
        * np.pi
        * sphere_radius**3)

    overdensities = []

    # Loop over spheres

    for i, centre in enumerate(centres):

        # yt sphere
        sp = ds.sphere(centre, (sphere_radius, "code_length"))

        # Total mass inside sphere
        sphere_mass = sp.quantities.total_mass()

        sphere_mass = sphere_mass.to("Msun/h").value

        # Density inside sphere
        density = sphere_mass / sphere_volume

        # Overdensity
        delta = density / mean_density - 1.0

        overdensities.append(delta)

        if (i + 1) % 100 == 0:
            print(
                f"Calculated {i + 1}/{n_spheres} spheres")

    return np.array(overdensities)


for snap in snapshot_numbers:

    print()
    print("=" * 60)
    print(f"SNAPSHOT {snap:03d}")
    print("=" * 60)

    snapdir = base_path / f"snapdir_{snap:03d}"

    snapshot_file = (
        snapdir / f"snapshot_{snap:03d}.0.hdf5")

    ds = yt.load(
    str(snapshot_file),
    bounding_box=np.array([[0, 100], [0, 100], [0, 100]])
)

    print(ds)

    chunks = []

    for i in range(n_files_per_snapshot):

        file = (
            snapdir
            / f"snapshot_{snap:03d}.{i}.hdf5")

        if not file.exists():
            continue

        with h5py.File(file, "r") as f:

            coords = f["PartType1/Coordinates"][:]

            # Header BoxSize is in kpc/h
            box_size = (
                float(f["Header"].attrs["BoxSize"])
                / 1000.0)

            # Centre z-slice on middle of box
            z_mid = box_size / 2.0

            z_min = (
                z_mid
                - slice_width / 2.0)

            z_max = (
                z_mid
                + slice_width / 2.0)

            # IMPORTANT:
            # coords are apparently in kpc/h,
            # so convert z coordinates to Mpc/h
            z = coords[:, 2] / 1000.0

            mask = (
                (z >= z_min)
                &
                (z <= z_max))

            # x and y converted to Mpc/h
            xy_slice = (
                coords[mask, :2]
                / 1000.0)

            if xy_slice.size > 0:
                chunks.append(xy_slice)


    if not chunks:

        print(
            f"No particles found for snapshot "
            f"{snap:03d}")

        continue

    positions = np.concatenate(chunks)

    x = positions[:, 0]
    y = positions[:, 1]

    print(f"Particles in z-slice: {len(x)}")


    # CALCULATE 3D OVERDENSITIES

    overdensities = calculate_overdensities(
        ds,
        sphere_radius=sphere_radius,
        n_spheres=n_spheres
    )


    print(
        f"Mean delta:   "
        f"{np.mean(overdensities):.4f}")

    print(
        f"Std delta:    "
        f"{np.std(overdensities):.4f}")


    # FIT GAUSSIAN

    mu, sigma = norm.fit(overdensities)

    print(f"Gaussian mu:      {mu:.4f}")

    print(f"Gaussian sigma:   {sigma:.4f}")


    # CREATE FIGURE

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 6))


    ax = axes[0]

    ax.scatter(
        x,
        y,
        s=0.01,
        alpha=0.1,
        rasterized=True
    )

    ax.set_title(f"Snapshot {snap:03d}")

    ax.set_xlabel("x [Mpc/h]")

    ax.set_ylabel("y [Mpc/h]")

    ax.set_xlim(0,box_size)

    ax.set_ylim(0,box_size)

    ax.set_aspect("equal")


    ax = axes[1]

    ax.hist(
        overdensities,
        bins=50,
        density=True,
        alpha=0.6,
        label="Sphere measurements"
    )

    # Gaussian curve
    delta_range = np.linspace(
        overdensities.min(),
        overdensities.max(),
        500
    )

    gaussian = norm.pdf(
        delta_range,
        mu,
        sigma
    )

    ax.plot(
        delta_range,
        gaussian,
        linewidth=2,
        label=(
            rf"Gaussian "
            rf"$\mu={mu:.3f}$, "
            rf"$\sigma={sigma:.3f}$"
        )
    )

    ax.axvline(
        0,
        linestyle="--",
        linewidth=1
    )

    ax.set_xlabel(
        r"Overdensity $\delta$"
    )

    ax.set_ylabel(
        "Probability density"
    )

    ax.set_title(
        rf"$R={sphere_radius}\,h^{{-1}}$ Mpc"
    )

    ax.legend()


    # ========================================================
    # SAVE
    # ========================================================

    plt.tight_layout()

    output_name = (
        f"snapshot_{snap:03d}_overdensity.png"
    )

    plt.savefig(
        output_name,
        dpi=200
    )
