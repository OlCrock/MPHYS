import numpy as np 
import matplotlib.pyplot as plt
import camb
import pandas as pd 

def get_linear_power(
    Omega_m=0.285,
    Omega_b=0.045,
    h=0.695,
    sigma8=0.828,
    ns=0.9632,
    redshifts=np.linspace(0, 99, 100),
    kmax=100,
    npoints=5000):
    
    pars = camb.CAMBparams()

    pars.set_cosmology( H0=100*h, ombh2=Omega_b*h**2,
        omch2=(Omega_m-Omega_b)*h**2, YHe=0.248, nnu=3.04)

    pars.InitPower.set_params(As=2e-9, ns=ns)
    pars.NonLinear = camb.model.NonLinear_none

    pars.set_matter_power(redshifts=list(redshifts), kmax=kmax)

    results = camb.get_results(pars)

    As = 2e-9 * (sigma8/results.get_sigma8_0())**2
    pars.InitPower.set_params(As=As, ns=ns)

    results = camb.get_results(pars)

    k, z, pk = results.get_matter_power_spectrum(minkh=1e-5, maxkh=kmax, npoints=npoints)

    return k, z, pk

def window_function(k, R):
    return 3 * (np.sin(k*R) - k*R*np.cos(k*R)) / (k*R)**3

def linear_sigma(pK, k, R):
    integrand = pK * k**2 * window_function(k, R)**2
    sigma_squared = np.trapezoid(integrand, k) / (2 * np.pi**2)
    return np.sqrt(sigma_squared)


k, redshifts, pk = get_linear_power()

R = np.linspace(1, 51, 101)

sigma_arr = np.zeros((len(redshifts), len(R)))

for i in range(len(redshifts)):
    for j, r in enumerate(R):
        sigma_arr[i, j] = linear_sigma(pk[i], k, r)

df = pd.DataFrame(sigma_arr, index=redshifts, columns=R)
df.index.name = "Redshift"
df.columns.name = "Radius (Mpc/h)"

df.to_csv("linear_sigma.csv")