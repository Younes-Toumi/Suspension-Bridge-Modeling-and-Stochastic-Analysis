from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import pickle
import numpy as np

with open("Results\\stored_result_3m.pickle", 'rb') as file:
    content = pickle.load(file)

    N_samples = content['N_samples']
    rv_matrix = content['rv_matrix']
    u_array = content['u_array']
    Pf = content['Pf']



rv_plot_Matrix = np.array([
    "Tension T [N]",
    "Young's modulus E [Pa]",
    "m1 cable  [kg/m]",
    "b1 cable",
    "m2 deck  [kg/m]",
    "b2 deck",
    "k [N.m]",
    "F [N/m]"
    ]) 



pearson_array = []
spearman_array = []

for i in range(8):
    figure, axis = plt.subplots(figsize=(8, 6))
    # Compute Pearson correlation coefficient
    correlation_p, _ = pearsonr(rv_matrix[:, i], u_array)
    correlation_s, _ = spearmanr(rv_matrix[:, i], u_array)

    pearson_array.append(correlation_p)
    spearman_array.append(correlation_s)

    accepted_sample = (u_array <= -3.3)
    rejected_sample = (u_array > -3.3)


    # Split X using the masks
    rv_accepted = rv_matrix[:, i][accepted_sample]
    rv_rejected = rv_matrix[:, i][rejected_sample]

    u_array_accepted = u_array[accepted_sample]
    u_array_rejected = u_array[rejected_sample]

    plt.scatter(rv_accepted, u_array_accepted, color='red', s=4, label='Accepted Samples')
    plt.scatter(rv_rejected, u_array_rejected, color='blue', s=1, label='Rejected Samples')

    axis.set_xlabel(f"{rv_plot_Matrix[i]}")
    axis.set_ylabel("Maximum downward deflection u [m]")
    axis.set_title(f"COR of {rv_plot_Matrix[i]} -> Pearsonr: {correlation_p:.3f} | Spearmanr: {correlation_s:.3f} ")

    plt.legend()



import pickle
with open('Results\\correlation_result.pickle', 'wb') as file:
    
    stored_data = {
        'pearson_array': pearson_array,
        'spearman_array': spearman_array,

    }
    
    pickle.dump(stored_data, file)


plt.show()
