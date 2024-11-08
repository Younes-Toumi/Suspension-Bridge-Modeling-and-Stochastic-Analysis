import numpy as np
from scipy.stats import norm
import multiprocessing
import time
from Header.PARAMETERS import *
from Header.model_solver import solve_sample




N_samples = 500



N_space = 51
space_points = np.linspace(0, L, N_space)

dx = space_points[1] - space_points[0]

dt = 0.025
period = 200
time_points = np.arange(0, period, dt)
N_time = len(time_points)

def g(rv_E, rv_m1, rv_m2, rv_F):
    
    with multiprocessing.Pool() as pool:
        # Parallelize the loop
        u = pool.starmap(solve_sample, [(i, T, rv_E, rv_m1, b1, rv_m2, b2, k_coupling, rv_F, dt, N_time, time_points, period, N_space, space_points, L, I, 'adapted') for i in range(N_samples)])

    u = np.array(u)
    return u + 3.3

T_mu, T_std = T, 0.05*T
E_mu, E_std = E, 0.05*E
m1_mu, m1_std = m1, 0.05*m1
b1_mu, b1_std = b1, 0.05*b1
m2_mu, m2_std = m2, 0.05*m2
b2_mu, b2_std = b2, 0.05*b2
k_mu, k_std = k_coupling, 0.05*k_coupling
F_mu, F_std = F, 0.05*F


x_E = lambda z1: np.exp( np.sqrt(np.log(E_std**2/E_mu**2 + 1))*z1  + np.log( E_mu**2 / (np.sqrt(E_std**2 + E_mu**2))  )         )
x_m1 = lambda z2: m1_mu + m1_std*z2 
x_m2 = lambda z3: m2_mu + m2_std*z3 
x_F = lambda z4: (F_mu - 0.57721 * (np.sqrt(6) * F_std / np.pi)) - (np.sqrt(6) * F_std / np.pi)*np.log( -np.log( norm.cdf(z4) ) )



# Previous results obtained from the FORM.py file.
beta = 4.48
alpha = np.array([-0.822,  0.079,  0.525,  0.206])

# ISPUD procedure
Z_sns = np.random.randn(N_samples, len(alpha))

# Projecting samples along important direction
Z_norm = Z_sns - (Z_sns * alpha.T) *alpha
Z_norm = Z_sns - np.dot(Z_sns, alpha.reshape(-1, 1)) * alpha

# Computing the parameters of the forced normal
b = np.exp(-beta**2/2) / (norm.cdf(-beta) * np.sqrt(2*np.pi)) 
v = 2*(b - beta) # corresponds to c*(b-beta) | till now used c = 20

# Generating forced samples according to a normal distribution N ~ (b, v)
Z_forced = np.random.normal(b, v, N_samples) # We can display a histogram

# Projecting the forced samples along important direction
Z_parallel = Z_forced[:, np.newaxis] * alpha # Broadcasting
# Uniting the two projected samples to get in m-dimensional space
Z_is = Z_norm + Z_parallel


# Identifying failed samples

rv_E_sns, rv_m1_sns, rv_m2_sns, rv_F_sns = x_E(Z_is[:, 0]), x_m1(Z_is[:, 1]), x_m2(Z_is[:, 2]), x_F(Z_is[:, 3])

if __name__ == '__main__':
    t1 = time.time()
    h_z = g(rv_E_sns, rv_m1_sns, rv_m2_sns, rv_F_sns)
    failure = h_z < 0
    failed_samples = np.sum(failure)

    is_weight = norm.pdf(Z_forced, 0, 1) / norm.pdf(Z_forced, b, v)

    Pf = np.sum(failure*is_weight) / N_samples
    var_Pf = 1/N_samples * (np.sum(failure*is_weight**2) / N_samples - Pf**2)
    cov_Pf = np.sqrt(var_Pf) / Pf

    N_required_MC = 1/cov_Pf**2 /Pf
    t2 = time.time()

    print(f"Results of Important Sampling:")
    print(f"Probability of failure: Pf: {Pf:.3e} | var_Pf: {var_Pf:.3e} | cov_Pf: {cov_Pf:.3e}")
    print(f"Required Monte Carlo samples: {int(N_required_MC)}")
    print(f"Solved the model in: {(t2-t1):.3f} [s]")
    print(f"Failed samples: {np.sum(failure*is_weight)} out of {N_samples}")