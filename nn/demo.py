# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 23:41:14 2022

@author: Renzo
"""
import common
import numpy as np
import time
import matplotlib.animation as animation
import matplotlib.pyplot as plt


# Which trajectories to use for the demos. Should be either:
#   - a list of indices
#   - "all"
TRAJECTORY_INDICES = "all"


# Which data source to get the indices from
# Should be either "train" or "test"
DATA_SOURCE = "test"


##############################################################################
# %% Set demo flags

# This demo script contains multiple visualizations and other demos.
# Use the flags below to turn demos on/off

# Create a plot of the distance between the endpoints of the reference and 
# the predicted trajectory. (Creates one figure with a line per trajectory)
PLOT_ENDPOINT_DISTANCES = True

# Create a plot of the average distance between the endpoints of 
# the reference and the predicted trajectory. Averaged over all trajectories.
PLOT_AVERAGE_ENDPOINT_DISTANCE = True

# Show a visual, animated representation of the reference and the 
# predicted trajectory. You can pause the animation with the space bar or skip
# through the frames using the arrow keys. 
# 
# WARNING: creates an animated plot per trajectory. Enabling while
# TRAJECTORY_INDICES = "all" will cause a hundred animated popups, which might 
# cause a lot of computational strain.
VISUALIZE_TRAJECTORIES = False

##############################################################################

INPUT_SIZE = 10
OUTPUT_SIZE = 1

model = common.RNN_model.load()
model.summary()

normalization = common.Normalization.from_model()

normalization.save_to_model()

if DATA_SOURCE == "train":
    data = np.load('storage/data/train.npy')
elif DATA_SOURCE == "test":
    data = np.load('storage/data/test.npy')
else:
    raise Exception("DATA_SOURCE should be either \"train\" or \"test\"");
    
data_norm = normalization.normalize(data)

if TRAJECTORY_INDICES == "all":
    TRAJECTORY_INDICES = range(data_norm.shape[0])


ground_truth_norm = data_norm[TRAJECTORY_INDICES, :, :]
    


tot_points = ground_truth_norm.shape[1]



predicted_traj_norm = np.zeros(ground_truth_norm.shape)
predicted_traj_norm[:, 0:INPUT_SIZE, :] = ground_truth_norm[:, 0:INPUT_SIZE, :]

i = INPUT_SIZE

while (i+1) < tot_points:
    
    copy_length = OUTPUT_SIZE
    
    if(i+1+copy_length >= tot_points):
        copy_length = tot_points - (i+1)
        
        
    model_input = predicted_traj_norm[:, i-INPUT_SIZE:i, :]
    model_output = model(model_input)
    
    
    predicted_traj_norm[:, i : i+copy_length, :] = model_output[:, 0:copy_length, :]
    
    i += copy_length


ground_truth = normalization.unnormalize(ground_truth_norm[:, :, :])
predicted_traj = normalization.unnormalize(predicted_traj_norm[:, :, :])



if PLOT_AVERAGE_ENDPOINT_DISTANCE:
    common.plot_average_endpoint_distance(ground_truth, predicted_traj)
    
if PLOT_ENDPOINT_DISTANCES:
    labels = [];
    for i in TRAJECTORY_INDICES:
        labels.append("Trajectory " + str(i+1))
    common.plot_endpoint_distance(ground_truth, predicted_traj, labels = labels)

if VISUALIZE_TRAJECTORIES:
    for i in range(ground_truth.shape[0]):
        both = np.concatenate((predicted_traj[i:i+1, :, :], ground_truth[i:i+1, :, :]), axis=0)
        ani1 = common.visualize_trajectories(both, labels=("Predicted trajectory", "Ground truth"), speed=0.25, save_gif=False)