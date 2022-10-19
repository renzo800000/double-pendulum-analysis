This folder contains the code of the neural network model used for predicting the behaviour of the double-pendulum system. All scripts use the model saved in the 'storage/models/current' folder. To learn how to save or load different models, please refer to 'storage/models/readme.md'.

# File meanings

## generate_data.py
Running this file will load the trajectories from 'storage/data/trajectories.csv', separate the trajectories, and split them into training and testing data (as defined by parameters in the generate_data.py file). The training and testing data will be saved as .npy files in 'storage/data'.

## train.py
Running this file will actually train the model, using the hyperparameters set in the file.

## demo.py
Running this file will generate plots and print out data about the current model and it's performance.

## common.py
Common.py contains common functions and definitions all of the other scripts use. Running it will have no effect.
