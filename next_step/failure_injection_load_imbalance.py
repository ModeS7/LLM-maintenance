import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

dataset = joblib.load('variable_of_interest.joblib')
dataset_health = joblib.load('variable_of_interest.joblib')

keys = ['Bus1_load', 'Bus1_Avail_load', 'Bus2_Avail_load', 'Bus2_load', 'BowThr1_Power',
    'BowThr2_Power', 'BowThr3_Power', 'SternThr1_Power', 'SternThr2_Power', 'Main_Prop_PS_Drive_Power',
    'Main_Prop_SB_Drive_Power', 'Main_Prop_PS_ME1_Power', 'Main_Prop_PS_ME2_Power','Draft_Aft', 'Draft_Fwd',
    'Latitude', 'Longitude', 'Speed']

#keys_of_interest = ['BowThr1_Power', 'BowThr2_Power', 'BowThr3_Power']

keys_of_interest = ['Bus1_load', 'Bus1_Avail_load', 'Bus2_Avail_load', 'Bus2_load']


#failure_coefficient = [-0.001, -0.1, -0.01] # for bow thruster

failure_coefficient = [-50, -25, +10, +15] # for bus load

time_of_injection = 20000#5000

#passo = -0.001 # for bow thruster

passo = -0.05 # for load

for i, key in enumerate(keys_of_interest):
    for j in range(len(dataset[key])):
        if j > time_of_injection:
            dataset[key] = dataset[key].astype(float)
            dataset[key][j] += failure_coefficient[i]
            if key == 'Bus1_load' or key == 'Bus1_Avail_load':
                failure_coefficient[i] = failure_coefficient[i]+passo
            else:
                failure_coefficient[i] = failure_coefficient[i] - passo


# plot
for i, key in enumerate(keys_of_interest):
    l = np.linspace(0, len(dataset[key]), len(dataset[key]))

    plt.figure()
    plt.plot(l, dataset[key], label = f'{keys_of_interest[i]} modified', color = 'r')
    plt.plot(l, dataset_health[key], label=f'{keys_of_interest[i]} original', color = 'b')
    plt.xlabel('Sample')
    plt.ylabel(f'{keys_of_interest[i]}')
    plt.title(f'{keys_of_interest[i]}')
    plt.legend()
    plt.show()

joblib.dump(dataset, 'dataset_imbalanced_load.joblib')