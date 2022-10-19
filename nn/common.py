import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tensorflow as tf
import tensorflow.keras as ks
import os

class RNN_model(ks.Model):
    
    def __init__(self, input_shape, batch_size, hidden_layer_size, output_shape):
        super(RNN_model, self).__init__()
        
        self.hidden_layer_size = hidden_layer_size
        self.m_input_shape = input_shape
        self.batch_size = batch_size
        
        self.m_layers = [
            
            ks.layers.LSTM(
                self.hidden_layer_size, 
                activation='relu',
                kernel_initializer='glorot_normal',
                input_shape = self.m_input_shape,
                return_sequences = True),
            
            ks.layers.LSTM(
                self.hidden_layer_size, 
                activation='relu',
                kernel_initializer='glorot_normal',
                input_shape = self.m_input_shape,
                return_sequences = True),
            
            ks.layers.LSTM(
                self.hidden_layer_size, 
                activation='relu',
                kernel_initializer='glorot_normal'),
            
            ks.layers.Dense(
                output_shape[0]*output_shape[1],
                kernel_initializer=tf.initializers.zeros() ,
                activation='linear'),
            
            ks.layers.Reshape(output_shape)
            ]
        
    def call(self, x):
        for i in range(len(self.m_layers)):
            x = self.m_layers[i](x)
        
        return x
    
    def save(self, path = "storage/model/current"):
        ks.Model.save(self, path)
    
    @staticmethod
    def load(path = "storage/model/current"):
        if os.path.exists(path):
          return ks.models.load_model(path, compile=False)
      
        raise Exception("There is currently no neural network model selected.\nPlease see the readme.md file in the folder 'storage/model' on how to select a model.");
        

#create a normalization class
class Normalization:
    def __init__(self,data):
        
        
        self.means = []
        self.stddevs = []
        
        
        # find mean and stddev of each column and store in self
        for i in range(data.shape[-1]):
            self.means.append(np.mean(data[:, :,i]))
            self.stddevs.append(np.std(data[:, :,i]))
        
    def normalize(self, data):
        
        # Create empty array for normalized data
        normalized_data = np.empty(data.shape)
        
        #iterate over all columns
        for i in range(0, data.shape[-1]):
            
            # Normalize each column using the mean and stddev values
            normalized_data[:, :,i] = (data[:, :,i]-self.means[i])/self.stddevs[i]
        
        # return
        return normalized_data
    
    def unnormalize(self, data):
        
        # Create empty array for unnormalized data
        unnormalized_data = np.empty(data.shape)
        
        #iterate over all columns
        for i in range(0, data.shape[-1]):            
            
            # Un-normalize each column using the mean and stddev values
            unnormalized_data[:, :,i] = self.means[i] + self.stddevs[i]*data[:, :,i]          
            
        # return
        return unnormalized_data
    
    
def calc_endpoint_distance(a, b, L1=1, L2=1.5):
    #Calculate the distance between the end of the pendulum for two data points
    
    xa = L2*np.sin(a[2]) + L1*np.sin(a[0])
    ya = -L2*np.cos(a[2]) -L1*np.cos(a[0])
    
    xb = L2*np.sin(b[2]) + L1*np.sin(b[0])
    yb = -L2*np.cos(b[2]) -L1*np.cos(b[0])
    
    return np.sqrt((xa-xb)**2+(ya-yb)**2)

def plot_endpoint_distance(trajectory_a, trajectory_b, L1=1, L2=1.5, dt=0.05, labels=None):
    
    if(trajectory_a.ndim == 2):
        trajectory_a = trajectory_a[None, ...]
        
    if(trajectory_b.ndim == 2):
        trajectory_b = trajectory_b[None, ...]
        
    fig = plt.figure()
    ax = plt.axes()
    ax.set_title("Distance between endpoint labels")
    ax.set_xlabel("Frame (dt=" + str(dt) + ")")
    ax.set_ylabel("Avg. endpoint distance")
        
    for i in range(trajectory_a.shape[0]):
        dist = np.zeros(trajectory_a.shape[1])
        
        for j in range(trajectory_a.shape[1]):
            a = trajectory_a[i, j, :]
            b = trajectory_b[i, j, :]
            this_dist = calc_endpoint_distance(a, b, L1=L1, L2=L2)
            dist[j] = this_dist
        
        label = None
        if labels is not None:
            label = labels[i]
            
        ax.plot(dist, label=label)
    
    if labels is not None:
        ax.legend()   
        
    fig.show() 
    
    
def plot_average_endpoint_distance(trajectory_a, trajectory_b, L1=1, L2=1.5, dt=0.05):
    
    if(trajectory_a.ndim == 2):
        trajectory_a = trajectory_a[None, ...]
        
    if(trajectory_b.ndim == 2):
        trajectory_b = trajectory_b[None, ...]
        
    dist = np.zeros(trajectory_a.shape[1])
    
    for j in range(trajectory_a.shape[1]):
        for i in range(trajectory_a.shape[0]):
        
            a = trajectory_a[i, j, :]
            b = trajectory_b[i, j, :]
            this_dist = calc_endpoint_distance(a, b, L1=L1, L2=L2)
            dist[j] += this_dist
        dist[j] /= trajectory_a.shape[0]
    
    
    fig = plt.figure()
    ax = plt.axes()
    
    
    ax.set_title("Average distance between double pendulum endpoints")
    ax.set_xlabel("Frame (dt=" + str(dt) + ")")
    ax.set_ylabel("Avg. endpoint distance")
    
    ax.plot(dist)
    
    fig.show()
    
    
def visualize_trajectories(trajectories, labels=None, L1=1, L2=1.5, dt=0.05, speed=1):
    
    if(trajectories.ndim == 2):
        trajectories = trajectories[None, ...]
    
    (trajs, rows, cols) = trajectories.shape
    
    if cols == 5:
        #Remove the time column if it is present
        trajectories = trajectories[:, :, 1:]
        cols = 4
    
    L = L1+L2
    
    x1 = L1*np.sin(trajectories[:, :, 0])
    y1 = -L1*np.cos(trajectories[:, :, 0])

    x2 = L2*np.sin(trajectories[:, :, 2]) + x1
    y2 = -L2*np.cos(trajectories[:, :, 2]) + y1
    
    x0 = np.zeros((trajs, rows))
    y0 = np.zeros((trajs, rows))
    
    x = np.empty((trajs, rows, 3))
    y = np.empty((trajs, rows, 3))
    
    for i in range(trajs):
        x[i, :, :] = np.vstack([x0[i, :], x1[i, :], x2[i, :]]).transpose()
        y[i, :, :] = np.vstack((y0[i, :], y1[i, :], y2[i, :])).transpose()
        
    
    
    fig = plt.figure(figsize=(5, 6.5))
    ax = fig.add_subplot(autoscale_on=False, xlim=(-L, L), ylim=(-L, 1.3*L))
    ax.set_aspect('equal')
    ax.grid()
    
    lines = []
    
    for i in range(trajs):
        label = None
        if labels is not None:
            label = labels[i]
            
        line, = ax.plot([], [], 'o-', lw=2, label=label) 
        lines.append(line)
        
    time_template = 'Frame %.0f/%.0f\ntime = %.2fs'
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
    
    if labels is not None:
        ax.legend()
    
    history_x = np.empty((trajs, rows))
    history_y = np.empty((trajs, rows))
    
    num_frames = x.shape[1]
    last_frame = None
    
    def animate(i):
        global history_x, history_y, last_frame
        thisx = x[:, i]
        thisy = y[:, i]
    
        if i == 0:
            history_x = np.empty((trajs, rows))
            history_y = np.empty((trajs, rows))
    
        history_x[:, i] = thisx[:, 2]
        history_y[:, i] = thisy[:, 2]
        
        for j in range(trajs):
            lines[j].set_data(thisx[j, :], thisy[j, :])
            #traces[j].set_data(history_x[j, :], history_y[j, :])
            
        time_text.set_text(time_template % (i+1, num_frames, i*dt))
        
        artists = []
        artists += lines
        artists.append(time_text)
        
        last_frame = i
        
        return artists
    
    def on_press(event):
        global last_frame;
      
        if event.key.isspace():
            if ani1.running:
                ani1.pause()
            else:
                ani1.resume()
            ani1.running ^= True
        
        
        if event.key == 'left':
            ani1.direction = -1
        elif event.key == 'right':
            ani1.direction = +1
        

        if event.key in ['left','right']:
            ani1.pause()
            ani1.running = False
            
            if last_frame is not None:
                if ani1.direction > 0:
                    i = last_frame + 1
                else:
                    i = last_frame - 1
          
                while i >= num_frames:
                    i = i - num_frames
              
                while i < 0:
                    i = i + num_frames
        
                animate(i)
                plt.draw()

    
    fig.canvas.mpl_connect('key_press_event', on_press)
    
    ani1 = None #Needed so we don't get reference errors in animate() function
    ani1 = animation.FuncAnimation(fig, animate, rows, interval=1/(dt*speed), blit=True)
    ani1.running = True
    ani1.direction = +1
    ani1.last_frame = None
    
    plt.show()
    
    
    return ani1