# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 23:41:14 2022

@author: Renzo
"""
import common
import numpy as np
import scipy 

# %% Load the joined trajectories from the csv file and split them up
joined_trajectories = np.genfromtxt('storage/data/trajectories.csv', delimiter=',')


# %% Split up the joined trajectories 
current_trajectory = []
reference_data = []

last_time = -1
for i in range(joined_trajectories.shape[0]):
    point = joined_trajectories[i, :]
    time = point[0]
    
    if(time < last_time):
        reference_data.append(np.array(current_trajectory))
        current_trajectory = []
  
    current_trajectory.append(point) 
    last_time = time
    
reference_data.append(np.array(current_trajectory))
current_trajectory = []


reference_data = np.array(reference_data)
(num_traj, points, columns) = reference_data.shape

# shape: (number of trajectories, 3, points per trajectory, number of columns)
#
# The dimension of size 3 diffentiates between the three datasets: 
# reference, nn model, SINDy model.
data = np.zeros([num_traj, 3, points, columns]);
data[:, 0, :, :] = reference_data

#Copy the time column to the nn and SINDy datasets
data[:, 1, :, 0] = reference_data[:, :, 0]
data[:, 2, :, 0] = reference_data[:, :, 0]


# %% Make predictions using the neural network prediction model
print("Making predictions using Neural Network model...")

NN_INPUT_SIZE = 10
NN_OUTPUT_SIZE = 1

model = common.RNN_model.load()

normalization = common.Normalization.from_model()
ground_truth_norm = normalization.normalize(reference_data[:, :, 1:])

predicted_traj_norm = np.zeros(ground_truth_norm.shape)
predicted_traj_norm[:, 0:NN_INPUT_SIZE, :] = ground_truth_norm[:, 0:NN_INPUT_SIZE, :]

i = NN_INPUT_SIZE

while (i+1) < points:
    
    copy_length = NN_OUTPUT_SIZE
    
    if(i+1+copy_length >= points):
        copy_length = points - (i+1)
        
        
    model_input = predicted_traj_norm[:, i-NN_INPUT_SIZE:i, :]
    model_output = model(model_input)
    
    
    predicted_traj_norm[:, i : i+copy_length, :] = model_output[:, 0:copy_length, :]
    
    i += copy_length


data[:, 1, :, 1:] = normalization.unnormalize(predicted_traj_norm[:, :, :])


# %% Make predictions using the SINDy prediction model
print("Making predictions using SINDy model...")


data[:, 2, 0:NN_INPUT_SIZE, :] = reference_data[:, 0:NN_INPUT_SIZE, :]

# Note that the function below is the matlab function 'optimal_ODE_func.m',
# translated to python code. 
def sindy_ODE_func(t1, o1, t2, o2):
    t4 = t1*2.0;
    t5 = t2*2.0;
    t6 = -t2;
    t7 = -t5;
    t8 = t1+t6;
    t9 = t1+t7;
    t11 = np.sin(t8);
    t13 = t4+t7;
    t10 = np.cos(t9);
    t12 = np.sin(t9);
    t14 = np.cos(t13);
    t15 = np.sin(t13);
    et1 = t11*(-7.72897286819941e-1)-t12*3.665491809199351-t15*3.763379226239405-np.sin(t1)*3.774169134051179-o2*t10*5.242986945888626e-1;
    et2 = o1*t14*6.659024321098571e-1-o2*t14*7.342839244213353e-1-t14*t15*6.497357600540721e-1-(o2**2)*t11*6.754697825155339e-1;
    et3 = t15*2.871332054138329+np.sin(t4+t6)*1.22619782360144-np.cos(t8)*6.478462671834546e-1-np.sin(t2)*9.688397635721511-o1*t14*6.459559549165819e-1;
    et4 = o2*t14*6.986929633126754e-1+t10*t12*5.074519443601693e-1+t14*t15*4.429576730217896e-1+(o1**2)*t11*5.28880682963013e-1;
    return [o1,et1+et2,o2,et3+et4];

def ODE_func(t, y):
    return sindy_ODE_func(y[0], y[1], y[2], y[3])

for i in range(num_traj):
    y0 = reference_data[i, NN_INPUT_SIZE-1, 1:]
    t_range = reference_data[i, NN_INPUT_SIZE-1:, 0]
    result = scipy.integrate.solve_ivp(ODE_func, [t_range[0], t_range[-1]], y0, t_eval=t_range)
    
    data[i, 2, NN_INPUT_SIZE:, 1:] = np.transpose(result.y)[1:, :]

# %% Save results
print("Saving results...")

np.save("performance_analysis_data.npy", data)