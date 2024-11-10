import numpy as np
from scipy.stats import norm
import time
import multiprocessing
import math
from Header.PARAMETERS import *
from Header.model_solver import solve_sample

N_initial_MC_samples = 9_000 # MC samples at the first level


N_space = 51
space_points = np.linspace(0, L, N_space)

dx = space_points[1] - space_points[0]

dt = 0.025
period = 200
time_points = np.arange(0, period, dt)
N_time = len(time_points)

def g(rv_E, rv_m1, rv_m2, rv_F, mode):
    if mode == 'adapted':
        with multiprocessing.Pool() as pool:
            # Parallelize the loop
            u = pool.starmap(solve_sample, [(i, T, rv_E, rv_m1, b1, rv_m2, b2, k_coupling, rv_F, dt, N_time, time_points, period, N_space, space_points, L, I, mode) for i in range(N_initial_MC_samples)])

    else: # mode == 'single'
            u = solve_sample(0, T, rv_E, m1, b1, rv_m2, b2, k_coupling, rv_F, dt, N_time, time_points, period, N_space, space_points, L, I, mode)

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




# Subset Sampling part ------------------------------------------- #

# Parameters of the subset simulation
target_Pf = 0.10 # Target failure probability
N_markow_chains = math.floor(N_initial_MC_samples * target_Pf) #
N_chain_length = math.ceil(N_initial_MC_samples / N_markow_chains) # N: samples per chain
delta_Z = 0.5 # Shape parameter of the proposal distribution


# Counters
N_model_calls = 0
N_rejections = 0

# Direct Markow chain sampling
ilevel = 0
MZ = np.random.randn(N_initial_MC_samples, 4)

rv_E_sns, rv_m1_sns, rv_m2_sns, rv_F_sns = x_E(MZ[:, 0]), x_m1(MZ[:, 1]), x_m2(MZ[:, 2]), x_F(MZ[:, 3])
N_model_calls = N_model_calls + N_initial_MC_samples

if __name__ == '__main__':
    g_MCS = g(rv_E_sns, rv_m1_sns, rv_m2_sns, rv_F_sns, mode='adapted')

    idx_closest_to_failure = np.argsort(g_MCS)[:N_markow_chains]

    z_closest = []
    g_closest = []
    g_intermediate = []
    Pf_i = []
    cov_i = []

    z_closest.append(MZ[idx_closest_to_failure, :])
    g_closest.append(g_MCS[idx_closest_to_failure])

    g_intermediate.append(g_MCS[idx_closest_to_failure[-1]])
    Pf_i.append(np.sum(g_MCS <= g_intermediate[0]) / N_initial_MC_samples)
    cov_i.append(np.sqrt(Pf_i[0] - Pf_i[0]**2) / N_initial_MC_samples / Pf_i[0])

    # Markow chains 
    t1 = time.time()
    while g_intermediate[ilevel] > 0:
        t1_1 = time.time()
        ilevel += 1
        z_chains = np.zeros((N_chain_length, 4, N_markow_chains))
        g_chains = np.zeros((N_chain_length, N_markow_chains))

        for i_chain in range(N_markow_chains):
            z_chains[0, :, i_chain] = z_closest[ilevel-1][i_chain, :]
            g_chains[0, i_chain] = g_closest[ilevel-1][i_chain]

            for i_sample in range(1, N_chain_length):
                z = z_chains[i_sample-1, :, i_chain]
                z_proposal = z + delta_Z * np.random.randn(1, 4)[0]



                x_E_prop = x_E(z_proposal[0])
                x_m1_prop = x_m1(z_proposal[1])
                x_m2_prop = x_m2(z_proposal[2])
                x_F_prop = x_F(z_proposal[3])


                alpha_sns = norm.pdf(z_proposal[0])/norm.pdf(z[0]) * norm.pdf(z_proposal[1])/norm.pdf(z[1]) * norm.pdf(z_proposal[2])/norm.pdf(z[2]) * norm.pdf(z_proposal[3])/norm.pdf(z[3])

                alpha = alpha_sns

                u = np.random.rand()
                
                if alpha < 1 and alpha < u:
                    z_chains[i_sample, :, i_chain] = z_chains[i_sample-1, :, i_chain]
                    g_chains[i_sample, i_chain] = g_chains[i_sample-1, i_chain]

                    N_rejections += 1

                else:
                    g_proposal = g(x_E_prop, x_m1_prop, x_m2_prop, x_F_prop, 'single')
                    N_model_calls += 1
                    
                    if g_proposal <= g_intermediate[ilevel-1]:
                        z_chains[i_sample, :, i_chain] = z_proposal
                        g_chains[i_sample, i_chain] = g_proposal

                    else:
                        z_chains[i_sample, :, i_chain] = z_chains[i_sample-1, :, i_chain]
                        g_chains[i_sample, i_chain] = g_chains[i_sample-1, i_chain]
                        N_rejections += 1

        idx_closest_to_failure = np.argsort(g_chains.ravel())[:N_markow_chains]
        g_intermediate.append(g_chains.ravel()[idx_closest_to_failure[-1]])
        g_closest.append(g_chains.ravel()[idx_closest_to_failure])

        shape = g_chains.shape
        i_sample, i_chain = np.unravel_index(idx_closest_to_failure, shape)

        z_closest.append(np.zeros((N_markow_chains, 4)))

        for i in range(N_markow_chains):
            z_closest[ilevel][i, :] = z_chains[i_sample[i], :, i_chain[i]]

        if g_intermediate[ilevel] > 0:
            Pf_i.append(np.sum(g_chains.ravel() <= g_intermediate[ilevel]) / (N_markow_chains * N_chain_length))

        else:
            Pf_i.append(np.sum(g_chains.ravel() <= 0) / (N_markow_chains * N_chain_length))





        t2_1 = time.time()
        print(f"At level: {ilevel} -> {Pf_i[ilevel]} | {(t2_1-t1_1):.3f} [s]")


        # Computing coefficient of variation
        if g_intermediate[ilevel] > 0:
            Mindicator_g = g_chains < g_intermediate[ilevel]

        else:
            Mindicator_g = g_chains <= 0

        # Compute the correlation of the indicator functions across different chains
        # Eq. 29
        Ri = np.zeros(N_chain_length)

        for k in range(1, N_chain_length+1): # k is from 0 to Nchanins - 1, nut MATLAB starts counting from 1. So you always do (k-1)
            for j in range(1, N_markow_chains+1):
                for l in range(1, N_chain_length+1 - (k-1)):
                    Ri[k-1] = Ri[k-1] + Mindicator_g[l-1, j-1] * Mindicator_g[l-1+k-1 -1, j-1]
                
            Ri[k-1] = Ri[k-1]/(N_initial_MC_samples - k*N_chain_length) - Pf_i[ilevel]**2

        
        # Eq. 25
        rho = Ri / Ri[0] # again, Ri(0) in the paper is Ri(1) here because MATLAB counts from 1
        
        # Eq. 27
        gamma_i = 0
        for k in range(1, N_chain_length):
            gamma_i = gamma_i + (1 - k*N_chain_length/N_initial_MC_samples)*rho[k-1]

        gamma_i = 2*gamma_i
        
        # Eq. 28
        cov_i.append( np.sqrt( (1-Pf_i[ilevel]) / (Pf_i[ilevel]*N_initial_MC_samples) *(1+gamma_i)) )


    t2 = time.time()
    Pf_hat = np.prod(Pf_i)
    cov_i = np.array(cov_i)

    CoV_Pf = np.sqrt( np.sum(cov_i**2 ))
    print(f"Probability of failure using SS: {Pf_hat:.3e}")
    print(f"Coefficient of variation using SS: {CoV_Pf:.3e}")
    print("Model calls: ", N_model_calls)
    print(f"Solved the model in: {(t2-t1):.3f} [s]")
    print(f"Required MC samples: {int(1/CoV_Pf**2 / Pf_hat)}")


