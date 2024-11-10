import numpy as np
import time
from scipy.stats import norm
from scipy.optimize import fsolve

from Header.PARAMETERS import *
from Header.model_solver import solve_sample

# Setting number of lines
N_lines = 8000



N_space = 51
space_points = np.linspace(0, L, N_space)

dx = space_points[1] - space_points[0]

dt = 0.025
period = 200
time_points = np.arange(0, period, dt)
N_time = len(time_points)

def g(rv_E, rv_m1, rv_m2, rv_F):
    

    u = solve_sample(0, T, rv_E, rv_m1, b1, rv_m2, b2, k_coupling, rv_F, dt, N_time, time_points, period, N_space, space_points, L, I, 'single')

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



# -------------------------- #
alpha = np.array([-0.82,  0.08,  0.52,  0.21])

# Line Sampling Procedure ----------------------------------------------------------------------

# Initializing matrices to store line starting and Pf of each line
Z_norm = np.zeros((4, N_lines))
Pf_line = np.zeros(N_lines)

# Generating samples in the SNS
Z_sns = np.random.randn(4, N_lines)

# Projecting samples in the direction normal to alpha to get starting point
for i in range(N_lines):
    Z_norm[:, i] = Z_sns[:, i] - np.dot(Z_sns[:, i], alpha) * alpha # Previously was Z_sns[:, i].T * alpha * alpha

# Computing the points on the line at fixed linear coordinate beta
coef_a, coef_b, coef_c = 2, 4.5, 6

Z_1 = Z_norm + coef_a*np.tile(alpha, (N_lines, 1)).T
Z_2 = Z_norm + coef_b*np.tile(alpha, (N_lines, 1)).T
Z_3 = Z_norm + coef_c*np.tile(alpha, (N_lines, 1)).T

t1 = time.time()
for i in range(N_lines):
    h_1 = g(x_E(Z_1[0, i]), x_m1(Z_1[1, i]), x_m2(Z_1[2, i]), x_F(Z_1[3, i]))
    h_2 = g(x_E(Z_2[0, i]), x_m1(Z_2[1, i]), x_m2(Z_2[2, i]), x_F(Z_2[3, i]))
    h_3 = g(x_E(Z_3[0, i]), x_m1(Z_3[1, i]), x_m2(Z_3[2, i]), x_F(Z_3[3, i]))

    A = np.array([
        [coef_a**2, coef_a, 1],
        [coef_b**2, coef_b, 1],
        [coef_c**2, coef_c, 1]
        ])
    
    b = np.array([h_1, h_2, h_3])
    coeff = np.linalg.inv(A).dot(b)

    # Intersection with limit state function
    b_star = fsolve(lambda beta: coeff[0] * beta**2 + coeff[1] * beta + coeff[2], 2)[0]

    # Compute probability of failure on the line
    Pf_line[i] = 1 - norm.cdf(b_star)

t2 = time.time()

Pf = np.sum(Pf_line) / N_lines
var_Pf = np.sum((Pf_line - Pf)**2) / (N_lines * (N_lines-1))
cov_Pf = np.sqrt(var_Pf)/Pf

N_required_MC = 1/cov_Pf**2 /Pf


print(f"Results of Line Sampling:")
print(f"Probability of failure: Pf: {Pf:.3e} | cov_Pf: {cov_Pf:.3e}")
print(f"Required Monte Carlo samples: {int(N_required_MC)}")
print(f"Solved the model in: {(t2-t1):.3f} [s]")
