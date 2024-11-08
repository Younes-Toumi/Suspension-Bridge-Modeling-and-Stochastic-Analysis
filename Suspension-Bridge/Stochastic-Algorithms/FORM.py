from Header.model_solver import solve_sample
from Header.PARAMETERS import *
import time
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

np.set_printoptions(precision=3)

# ---------------------------------------------------------------------------------------------------------------------- # 
# 1. Defining our random variables ----------------------------------------------------------------- # 
# ---------------------------------------------------------------------------------------------------------------------- # 

T_mu, T_std = T, 0.05*T
E_mu, E_std = E, 0.05*E
m1_mu, m1_std = m1, 0.05*m1
b1_mu, b1_std = b1, 0.05*b1
m2_mu, m2_std = m2, 0.05*m2
b2_mu, b2_std = b2, 0.05*b2
k_mu, k_std = k_coupling, 0.05*k_coupling
F_mu, F_std = F, 0.05*F


x_E = lambda z1: np.exp( np.sqrt(np.log(E_std**2/E_mu**2 + 1))*z1  + np.log( E_mu**2 / (np.sqrt(E_std**2 + E_mu**2))  )         ) # x = e^(vz + m)
x_m1 = lambda z2: m1_mu + m1_std*z2 
x_m2 = lambda z3: m2_mu + m2_std*z3 
x_F = lambda z4: (F_mu - 0.57721 * (np.sqrt(6) * F_std / np.pi)) - (np.sqrt(6) * F_std / np.pi)*np.log( -np.log( norm.cdf(z4) ) )

# ---------------------------------------------------------------------------------------------------------------------- # 
# 2. Descritizing our bridge & specifying parameters for the solver --------------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------------------------------------------------- # 

N_space = 51
space_points = np.linspace(0, L, N_space)

dx = space_points[1] - space_points[0]

dt = 0.025
period = 200
time_points = np.arange(0, period, dt)
N_time = len(time_points)


def g(rv_E, rv_m1, rv_m2, rv_F):

    u = solve_sample(0, T, rv_E, rv_m1, b1, rv_m2, b2, k_coupling, rv_F, dt, N_time, time_points, period, N_space, space_points, L, I, mode='single')
    
    return u + 3.3


# Initial guess and parameters for FORM
z = np.array([0, 0, 0, 0]) # z: [-3.534  0.513  2.471  0.99 ] | alpha: [-0.793  0.115  0.555  0.222]

z_array = [z]
beta = [np.inf]
delta_z = np.array([0.05, 0.05, 0.05, 0.05])

t1 = time.time()
for i in range(20):

    rv_E_sns, rv_m1_sns, rv_m2_sns, rv_F_sns = x_E(z[0]), x_m1(z[1]), x_m2(z[2]), x_F(z[3])
    rv_E_sns_delta_p, rv_m1_sns_delta_p, rv_m2_sns_delta_p, rv_F_sns_delta_p = x_E(z[0]+delta_z[0]), x_m1(z[1]+delta_z[1]), x_m2(z[2]+delta_z[2]), x_F(z[3]+delta_z[3])
    rv_E_sns_delta_m, rv_m1_sns_delta_m, rv_m2_sns_delta_m, rv_F_sns_delta_m = x_E(z[0]-delta_z[0]), x_m1(z[1]-delta_z[1]), x_m2(z[2]-delta_z[2]), x_F(z[3]-delta_z[3])
        
    h_z = g(rv_E_sns, rv_m1_sns, rv_m2_sns, rv_F_sns)

    h_z1_delta_p = g(rv_E_sns_delta_p,  rv_m1_sns,          rv_m2_sns,          rv_F_sns)
    h_z2_delta_p = g(rv_E_sns,        rv_m1_sns_delta_p,    rv_m2_sns,          rv_F_sns)
    h_z3_delta_p = g(rv_E_sns,        rv_m1_sns,          rv_m2_sns_delta_p,    rv_F_sns)
    h_z4_delta_p = g(rv_E_sns,        rv_m1_sns,          rv_m2_sns,          rv_F_sns_delta_p)


    h_z1_delta_m = g(rv_E_sns_delta_m,  rv_m1_sns,          rv_m2_sns,          rv_F_sns)
    h_z2_delta_m = g(rv_E_sns,        rv_m1_sns_delta_m,    rv_m2_sns,          rv_F_sns)
    h_z3_delta_m = g(rv_E_sns,        rv_m1_sns,          rv_m2_sns_delta_m,    rv_F_sns)
    h_z4_delta_m = g(rv_E_sns,        rv_m1_sns,          rv_m2_sns,          rv_F_sns_delta_m)

    dh_dz1 = (h_z1_delta_p - h_z1_delta_m) / (2*delta_z[0])
    dh_dz2 = (h_z2_delta_p - h_z2_delta_m) / (2*delta_z[1])
    dh_dz3 = (h_z3_delta_p - h_z3_delta_m) / (2*delta_z[2])
    dh_dz4 = (h_z4_delta_p - h_z4_delta_m) / (2*delta_z[3])

    dh_dz = np.array([dh_dz1, dh_dz2, dh_dz3, dh_dz4])

    # Reliability index (beta) update
    beta_new = (h_z - np.dot(dh_dz, z)) / np.linalg.norm(dh_dz)

    # Important direction (alpha)
    alpha_new = - dh_dz / np.linalg.norm(dh_dz)

    # Update design point (z)
    z_new = alpha_new * beta_new

    
    z = z_new
    z_array.append(z_new)
    beta.append(beta_new)
    i += 1
    print(f"Iteration {i}: z: {z} | Alpha: {alpha_new} | Pf={1 - norm.cdf(beta[-1]):.3e}")

t2 = time.time()
# Probability of failure using FORM
Pf_FORM = 1 - norm.cdf(beta[-1])
print(f"\nProbability of failure using FORM is: {Pf_FORM:.3e}")
print(f"Final results:\nbeta:{beta[-1]:.2f} | z: {z} | alpha: {alpha_new}")
print(f"Solved the model in: {(t2-t1):.3f} [s]")

fig, axis_beta = plt.subplots()
axis_beta.plot(range(i+1), beta, 'x--', color='blue', lw=2)
axis_beta.set_xlabel("Number of FORM iterations")
axis_beta.set_ylabel("Reliability index β")
axis_beta.set_title(f"Evolution of reliability index β with number of iterations | Final value -> β = {beta[-1]:.2f}")
plt.grid()




# z_array = np.array(z_array)
# z_x = z_array[:, 0]
# z_y = z_array[:, 1]
# z_z = z_array[:, 2]

# fig, axis_z = plt.subplots(subplot_kw={'projection':'3d'})
# axis_z.plot(z_x, z_y, z_z, 'x--')
# axis_z.set_xlabel("E in SNS")
# axis_z.set_ylabel("m2 in SNS")
# axis_z.set_zlabel("F in SNS")
# plt.grid()


plt.show()
