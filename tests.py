import yt
import numpy as np

path = "/cephfs2/brs/bw_mw/100Mpc_256/dm-only-L0/DD0000/DD0000"

ds = yt.load(path)

print("\n=== DATASET ===")
print(ds)
print("Redshift:", ds.current_redshift)
print("Cosmological:", ds.cosmological_simulation)

print("\n=== BOX ===")
print("domain_dimensions:", ds.domain_dimensions)
print("domain_left_edge:", ds.domain_left_edge)
print("domain_right_edge:", ds.domain_right_edge)
print("domain_width:", ds.domain_width)

print("\n=== UNITS ===")
print("length_unit:", ds.length_unit)
print("1 code_length:", ds.quan(1, "code_length"))
print("1 code_length in Mpc/h:", ds.quan(1, "code_length").to("Mpc/h"))

print("\n=== PARTICLES ===")
ad = ds.all_data()

x = ad["all", "particle_position_x"]
y = ad["all", "particle_position_y"]
z = ad["all", "particle_position_z"]

print("Number of particles:", len(x))
print("Expected:", 256**3)

print("\n=== POSITION RANGES ===")
print("x:", x.min(), "to", x.max())
print("y:", y.min(), "to", y.max())
print("z:", z.min(), "to", z.max())

print("\n=== PARTICLE MASS ===")
m = ad["all", "particle_mass"]
print("Particle mass:", m.min(), "to", m.max())
print("Unique masses:", np.unique(m).size)
