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

# %% Load the analysis data.

data = np.load("performance_analysis_data.npy");

[num_trajs, _, num_points, num_columns] = data.shape


# %% Compute endpoint distance metrics

dist = np.zeros([2, num_trajs, num_points])
avg_dist = np.zeros([2, num_points])
    
for j in range(num_points):
    for i in range(num_trajs):
  
        ref = data[i, 0, j, :]
        nn_pred = data[i, 1, j, :]
        sindy_pred = data[i, 2, j, :]
        
        nn_dist = common.calc_endpoint_distance(ref, nn_pred)
        sindy_dist = common.calc_endpoint_distance(ref, sindy_pred)
        
        dist[0, i, j] = nn_dist
        dist[1, i, j] = sindy_dist
        
        avg_dist[0, j] += nn_dist
        avg_dist[1, j] += sindy_dist
        
    avg_dist[:, j] /= num_trajs


# %% Plot average endpoint distance (full & cropped)

fig = plt.figure()
ax = plt.axes()
    
ax.set_title("Average distance between double pendulum endpoints")
ax.set_xlabel("Frame (dt=0.05)")
ax.set_ylabel("Avg. endpoint distance")
ax.set_xlim([-5, 205])
ax.set_ylim([-0.2, 3.2])
  
ax.plot(avg_dist[0, :], label="Neural Network prediction model")
ax.plot(avg_dist[1, :], label="SINDy prediction model")
    
ax.legend()
fig.show()



fig = plt.figure()
ax = plt.axes()
  
ax.set_title("Average distance between double pendulum endpoints")
ax.set_xlabel("Frame (dt=0.05)")
ax.set_ylabel("Avg. endpoint distance")
ax.set_xlim([-5, 55])
ax.set_ylim([-0.2, 3.2])
  
ax.plot(avg_dist[0, :], label="Neural Network prediction model")
ax.plot(avg_dist[1, :], label="SINDy prediction model")
    
ax.legend()
fig.show()


# %% Calculate and plot amount of non-diverged trajectories (full & cropped)

cutoff = 0.5;

nn_diverged_trajs = [];
sindy_diverged_trajs = [];

nn_converged_count = np.zeros(num_points)
sindy_converged_count = np.zeros(num_points)

for j in range(num_points):
    for i in range(num_trajs):
    
        if i not in nn_diverged_trajs:
            nn_dist = dist[0, i, j]
            nn_diverged = nn_dist > cutoff
            
            if nn_diverged:
                nn_diverged_trajs.append(i)
            else:
                nn_converged_count[j] = nn_converged_count[j] + 1
                
        if i not in sindy_diverged_trajs:
            sindy_dist = dist[1, i, j]
            sindy_diverged = sindy_dist > cutoff
            
            if sindy_diverged:
                sindy_diverged_trajs.append(i)
            else:
                sindy_converged_count[j] = sindy_converged_count[j] + 1



fig = plt.figure()
ax = plt.axes()
  
ax.set_title("Number of trajectories that have not yet diverged\nA trajectory counts as diverged once the endpoint distance has exceeded 0.5.")
ax.set_xlabel("Frame (dt=0.05)")
ax.set_ylabel("Number of trajectories that have not diverged")
ax.set_xlim([-5, 205])

  
ax.plot(nn_converged_count, label="Neural Network prediction model")
ax.plot(sindy_converged_count, label="SINDy prediction model")
    
ax.legend()
fig.show()


fig = plt.figure()
ax = plt.axes()
  
ax.set_title("Number of trajectories that have not yet diverged\nA trajectory counts as diverged once the endpoint distance has exceeded 0.5.")
ax.set_xlabel("Frame (dt=0.05)")
ax.set_ylabel("Number of trajectories that have not diverged")
ax.set_xlim([0, 30])
#ax.set_ylim([-0.2, 3.2])

  
ax.plot(nn_converged_count, label="Neural Network prediction model")
ax.plot(sindy_converged_count, label="SINDy prediction model")
    
ax.legend()
fig.show()


# %% Calculate and plot amount of non-exploded trajectories

cutoff = 100;

nn_exploded_trajs = [];
sindy_exploded_trajs = [];

nn_non_exploded_count = np.zeros(num_points)
sindy_non_exploded_count = np.zeros(num_points)

for j in range(num_points):
    for i in range(num_trajs):
        data_sindy = data[i, 2, j, 1:]
    
        if i not in nn_exploded_trajs:
            nn_data = data[i, 1, j, 1:]
            nn_exploded = np.any(np.abs(nn_data) > cutoff)
            
            if nn_exploded:
                nn_exploded_trajs.append(i)
            else:
                nn_non_exploded_count[j] = nn_non_exploded_count[j] + 1
                
        if i not in sindy_exploded_trajs:
            sindy_data = data[i, 2, j, 1:]
            sindy_exploded = np.any(np.abs(sindy_data) > cutoff)
            
            if sindy_exploded:
                sindy_exploded_trajs.append(i)
            else:
                sindy_non_exploded_count[j] = sindy_non_exploded_count[j] + 1

fig = plt.figure()
ax = plt.axes()
  
ax.set_title("Number of trajectories that have not yet exploded\nA trajectory counts as exploded once any absolute \nstate variable (angle or angular velocity) exceeds 100.")
ax.set_xlabel("Frame (dt=0.05)")
ax.set_ylabel("Number of trajectories that have not exploded")
ax.set_xlim([-5, 205])

  
ax.plot(nn_non_exploded_count, label="Neural Network prediction model")
ax.plot(sindy_non_exploded_count, label="SINDy prediction model")
    
ax.legend()
fig.show()


# %% Generate animations of every trajectory (turned off right now, takes a long time to run)
if(False):
    for i in range(num_trajs):
        trajectory_data = data[i, :, :, :];
        common.visualize_trajectories(trajectory_data, labels=("Ground truth", "Neural Network model", "SINDy model"), speed=0.25, save_gif=True, gif_filename="figures/trajectory-animations/trajectory-" + str(i) + ".gif")


