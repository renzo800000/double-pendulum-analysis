# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 20:46:20 2022

@author: Renzo
"""

import tensorflow as tf
import numpy as np
import common
import matplotlib.pyplot as plt
import csv
import os

INPUT_LENGTH = 10;

BUFFER_SIZE = 10000
BATCH_SIZE = 128
EPOCHS = 2500
HIDDEN_LAYER_SIZE = 32
MODEL_OUTPUT_SIZE = 1
    
def preprocess_data(data, input_length, test_points):
    
    # Amount of trajectories, rows and columns in the data
    (trajs, rows, cols) = data.shape
    
    # Amount of training points we can generate per trajectory
    traj_points = rows - input_length - test_points;
    
    #Total amount of training points
    tot_points = traj_points * trajs
    
    
    x = np.ndarray((tot_points, input_length, cols),float)
    y = np.ndarray((tot_points, test_points, cols),float)
     
    for traj in range(0,trajs):    
        for row in range(traj_points):
            point = traj*traj_points + row
            
            x[point, :, :] = data[traj, row:row+input_length, :]
            y[point, :, :] = data[traj, row+input_length:row+input_length+test_points, :]
            
    #y = y[:, 0, :]
     
    return x, y
    
if __name__ == '__main__':
    #Load data, but without the time column
    data_train = np.load('storage/data/train.npy')
    data_test = np.load('storage/data/test.npy')
    
    normalization = common.Normalization(data_train)
    
    data_train_norm = normalization.normalize(data_train)
    data_test_norm = normalization.normalize(data_test)
     
    x_train, y_train = preprocess_data(data_train_norm, INPUT_LENGTH, MODEL_OUTPUT_SIZE)
    x_test, y_test = preprocess_data(data_test_norm, INPUT_LENGTH, MODEL_OUTPUT_SIZE)
    
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    
    train_ds = train_ds.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    test_ds = test_ds.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    
    model = common.RNN_model(
        input_shape = (INPUT_LENGTH, 4),
        hidden_layer_size = HIDDEN_LAYER_SIZE,
        output_shape = (MODEL_OUTPUT_SIZE, 4),
        batch_size = BATCH_SIZE
        )
    
    model.compile(
        loss = tf.keras.losses.MeanSquaredError(),
        optimizer = "Adam",#tf.keras.optimizers.Adam(learning_rate = LEARNING_RATE),
        metrics=[tf.keras.metrics.Accuracy()]
        )
    
    fit_result = model.fit(
        train_ds,
        validation_data = test_ds,
        #batchesPerEpoch = len(train_ds),
        epochs = EPOCHS
        )
    
    
    train_loss_hist = fit_result.history['loss']
    test_loss_hist = fit_result.history['val_loss']
        
        
    print("Saving model...")
    model.save(path="storage/model/current")
    
    if not os.path.exists("storage/model/current/training_info"):
        os.makedirs("storage/model/current/training_info")
        
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(train_loss_hist, label='Train loss')
    ax.plot(test_loss_hist, label='Test loss')
    ax.legend()
    ax.set(xlabel='$Epoch$', ylabel='$Loss$',yscale='log')
    fig.savefig("storage/model/current/training_info/loss_history.png")
    
    
    with open("storage/model/current/training_info/loss_history.csv", "w+", newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Training loss', 'Testing loss'])
        for i in range(len(train_loss_hist)):
            writer.writerow([train_loss_hist[i], test_loss_hist[i]])