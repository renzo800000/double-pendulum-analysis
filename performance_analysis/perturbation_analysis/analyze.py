# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 23:41:14 2022

@author: Renzo
"""
import common
import numpy as np

# %% Load the analysis data.

data = np.load("perturbation_analysis_data.npy");

# %% Show and export three animations: 1 with all trajectories of the 
# reference data, one from the SINDy model, and one from the nn model

common.visualize_trajectories(data[:, 0, :, :], save_gif=True, gif_filename="figures/perturbation_ref.gif")
common.visualize_trajectories(data[:, 1, :, :], save_gif=True, gif_filename="figures/perturbation_SINDy.gif")
common.visualize_trajectories(data[:, 2, :, :], save_gif=True, gif_filename="figures/perturbation_nn.gif")