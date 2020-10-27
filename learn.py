import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from NN_learn import *
import os, shutil

from tensorflow.keras.layers import Activation
from tensorflow.keras import backend as K
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras import layers
from tensorflow import keras
def swish(x):
    return x * K.sigmoid(x)

get_custom_objects().update({'swish': Activation(swish)})


samples =  np.loadtxt( "samples.csv" ).T

samples_test = np.loadtxt( "samples_test.csv" ).T
N=samples.shape[1]
N_samples = 10**6






schedule =  tf.keras.optimizers.schedules.InverseTimeDecay(
0.01,   decay_steps= 10000,
decay_rate=0.2, staircase=True)
width=15

model = tf.keras.Sequential()
model.add(layers.Dense(width,activation= swish, input_shape=(N-1,)))
model.add(layers.Dense(width, activation=swish))
#model.add(layers.Dense(width, activation=swish))
model.add(layers.Dense(1))


#model.add(layers.Dense(1 , input_shape=(N-1,)))
model.summary()
H = train_NN_ising(samples, model, epochs = 100, batch_size=2000, eta = schedule)


print("Training done! Sampling from the learned model")
NN_samples = NN_MCMC(H,N, N_samples)

ns_list = [4*10**4, 10**5, 4*10**5, 10**6]
base_line = []
TV_err = []

print("Sampling done! Calculating TV")

for ns in ns_list:
    base_line.append( TVD(samples[:ns,:], samples_test[:ns, :],  ns) )
    TV_err.append( TVD(samples_test[:ns,:], NN_samples[:ns, :], ns) )



plt.rc('font', family='serif')
plt.plot(ns_list, TV_err , '-+', label =  "$n=10^6$,NN_GRISE")

plt.plot(ns_list, base_line, '--', label =  "Sampling error")


plt.xlabel("Number of testing samples")
plt.ylabel("TVD")
plt.xscale('log')
plt.yscale('log')
plt.yticks([0.1,1])

plt.legend(bbox_to_anchor=(0, 0), loc=3, prop={'size': 7})
plt.show()
