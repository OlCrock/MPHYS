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
    redshifts = np.array([
    99.000000, 19.908000, 19.351375, 18.809568, 18.109530,
    17.600784, 17.105582, 16.465760, 16.000776, 15.399996,
    14.963385, 14.538398, 13.989296, 13.590242, 13.074647,
    12.699943, 12.335214, 11.863969, 11.521497, 11.079007,
    10.757433, 10.444419, 10.039992, 9.746078, 9.366329,
    9.090350, 8.821719, 8.474635, 8.222396, 7.896491,
    7.659643, 7.429101, 7.131230, 6.914755, 6.635061,
    6.431795, 6.233942, 5.978306, 5.792525, 5.552488,
    5.378044, 5.208244, 4.988854, 4.829415, 4.623413,
    4.473703, 4.327979, 4.139697, 4.002865, 3.826071,
    3.697589, 3.572527, 3.410941, 3.293510, 3.141785,
    3.031519, 2.924190, 2.785515, 2.684735, 2.586638,
    2.459892, 2.367780, 2.248768, 2.162278, 2.078090,
    1.969315, 1.890264, 1.788127, 1.713899, 1.641648,
    1.548297, 1.480454, 1.392799, 1.329097, 1.267090,
    1.186975, 1.128752, 1.053525, 0.998855, 0.822301,
    0.769305, 0.717850, 0.667892, 0.619387, 0.572292,
    0.511623, 0.467662, 0.424979, 0.383538, 0.343303,
    0.304237, 0.266307, 0.229481, 0.193725, 0.147663,
    0.114287, 0.084546, 0.055598, 0.020498, 0.000000
]),
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