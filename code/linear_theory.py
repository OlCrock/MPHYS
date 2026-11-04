import numpy as np 
import matplotlib.pyplot as plt
import camb
import pandas as pd 

def window_function(k, R):
    return 3 * (np.sin(k*R) - k*R*np.cos(k*R)) / (k*R)**3

def linear_sigma(pK, k, R):
    integrand = pK * k**2 * window_function(k, R)**2
    sigma_squared = np.trapezoid(integrand, k) / (2 * np.pi**2)
    return np.sqrt(sigma_squared)

power_spectrum = np.loadtxt("generated_data/input_powerspec 2.txt")

k = power_spectrum[:, 0]
pk = power_spectrum[:, 5]

R = np.linspace(1, 51, 101)

sigma_arr = np.zeros((len(k), len(R)))

for i in range(len(k)):
    for j, r in enumerate(R):
        sigma_arr[i, j] = linear_sigma(pk[i], k, r)

df = pd.DataFrame(sigma_arr, index=k, columns=R)
df.index.name = "Redshift"
df.columns.name = "Radius (Mpc/h)"

df.to_csv("linear_sigma.csv")