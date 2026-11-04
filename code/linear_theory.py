import numpy as np
import pandas as pd
import h5py
from scipy.integrate import quad


def window_function(k, R):
    x = k * R
    return 3 * (np.sin(x) - x * np.cos(x)) / x**3


def growth_factor(z, Omega_m=0.285, Omega_L=0.715):
    z = np.atleast_1d(np.asarray(z, dtype=float))

    def H(a):
        return np.sqrt(Omega_m / a**3 + Omega_L)

    def D_of_a(a):
        integrand = lambda x: 1.0 / (x**3 * H(x)**3)
        return H(a) * quad(integrand, 1e-12, a, limit=200)[0]

    a = 1.0 / (1.0 + z)
    D = np.array([D_of_a(ai) for ai in a])
    D0 = D_of_a(1.0)
    out = D / D0
    return float(out[0]) if out.size == 1 else out


def linear_sigma(pk, k, R):
    integrand = pk * k**2 * window_function(k, R)**2
    sigma_squared = np.trapezoid(integrand, k) / (2 * np.pi**2)
    return np.sqrt(sigma_squared)


# Load LEGACY snapshot redshifts and radii
with h5py.File("optimised/output/variances.h5", "r") as f:
    redshifts = f["redshift"][:]
    R = f["radii"][:]

# Load CDM power spectrum at z=99
power_spectrum = np.loadtxt("generated_data/input_powerspec 2.txt")
k = power_spectrum[:, 0]
pk = power_spectrum[:, 1]  # P_cdm

# Calculate sigma(R) at z=99
sigma_ref = np.array([
    linear_sigma(pk, k, r) for r in R
])

# Evolve to every LEGACY snapshot redshift
growth_ratios = growth_factor(redshifts) / growth_factor(99.0)
sigma_all = growth_ratios[:, None] * sigma_ref[None, :]

# Save results
df = pd.DataFrame({
    "Redshift": np.repeat(redshifts, len(R)),
    "Radius (Mpc/h)": np.tile(R, len(redshifts)),
    "Sigma": sigma_all.flatten()
})

df.to_csv("linear_sigma.csv", index=False)

# Report results
print("Power spectrum: P_cdm")
print("Initial-condition redshift: z = 99")
print(f"Growth factor D(0)/D(99): {1/growth_factor(99.0):.6f}")
print(f"Number of LEGACY snapshots: {len(redshifts)}")
print(f"Number of radii: {len(R)}")
print("\nCDM sigma8 at each snapshot:")

for z, sigma in zip(redshifts, sigma_all):
    print(f"z = {z:10.6f}  sigma8 = {np.interp(8, R, sigma):.6f}")

print("\nSaved linear_sigma.csv")


