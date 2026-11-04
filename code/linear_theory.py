import numpy as np
import pandas as pd
from scipy.integrate import quad


def window_function(k, R):
    x = k * R
    return 3 * (np.sin(x) - x * np.cos(x)) / x**3


def growth_factor(z, Omega_m=0.285, Omega_L=0.715):
    """Linear growth factor D(z), normalized so D(0)=1 in an LCDM cosmology."""
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


def linear_sigma(pk, k, R, z=0.0, z_ref=99.0, Omega_m=0.3, Omega_L=0.7):
    """Return sigma(R) at redshift z, assuming pk is the power spectrum at z_ref."""
    integrand = pk * k**2 * window_function(k, R)**2
    sigma_squared = np.trapezoid(integrand, k) / (2 * np.pi**2)
    sigma = np.sqrt(sigma_squared)

    if np.isclose(z, z_ref):
        return sigma

    growth_ratio = growth_factor(z, Omega_m, Omega_L) / growth_factor(z_ref, Omega_m, Omega_L)
    return sigma * growth_ratio


# Load initial-condition power spectrum (z=99)
power_spectrum = np.loadtxt("generated_data/input_powerspec 2.txt")

k = power_spectrum[:, 0]
pk = power_spectrum[:, 1]

# Smoothing radii
R = np.linspace(1, 51, 101)

# Calculate sigma at the reference redshift z=99
sigma_ref = np.array([
    linear_sigma(pk, k, r, z=99.0, z_ref=99.0) for r in R
], dtype=float)

# Evolve to z=0 using linear growth, assuming the loaded power spectrum is at z=99
sigma_z0 = np.array([
    linear_sigma(pk, k, r, z=0.0, z_ref=99.0) for r in R
], dtype=float)

# Save theoretical standard deviation at z=0
# (sigma at z=99 is used internally as the initial spectrum normalization)
df = pd.DataFrame({
    "Radius (Mpc/h)": R,
    "Sigma_z0": sigma_z0,
    "Sigma_z99": sigma_ref,
})

df.to_csv("linear_sigma.csv", index=False)

print(f"Initial-condition redshift: z = 99")
print(f"Growth factor D(z=0) / D(z=99): {growth_factor(0.0) / growth_factor(99.0):.6f}")
print(f"Sigma8 at z=0: {np.interp(8, R, sigma_z0):.6f}")
print(f"Sigma8 at z=99: {np.interp(8, R, sigma_ref):.6f}")
