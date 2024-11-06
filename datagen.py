from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import pdb
import pickle
import numpy as np
import scipy.sparse as sp
import networkx as nx
import matplotlib.pyplot as plt
from utils import consensus_matrix

np.random.seed(0)

# Eperiment
dataID = sys.argv[1]
dataID = 'set'+str(dataID)+'_dynamic'

# Number of nodes
nNodes = 25

# Path gain exponent
pl = 2.2

# Channel
channel = 'nlos'

# Rayleigh (NLOS) or Rician(LOS) distribution scale
alpha = 1

# Batch size
batch_size = 64

# Training iterations
tr_iter = 100

# Testing iterations
te_iter = 100


# Build random geometric graph
def build_adhoc_network( nNodes, r=1.0, pl=2.2 ):
    transmitters = np.random.uniform(low=-nNodes/r, high=nNodes/r, size=(nNodes,2))
    receivers = transmitters + np.random.uniform(low=-nNodes/4,high=nNodes/4, size=(nNodes,2))

    L = np.zeros((nNodes,nNodes))
    dist = np.zeros((nNodes, nNodes))

    for i in np.arange(nNodes):
        for j in np.arange(nNodes):
            d = np.linalg.norm(transmitters[i,:]-receivers[j,:])
            L[i,j] = np.power(d,-pl)
            dist[i, j] = d

    return( dict(zip(['tx', 'rx'],[transmitters, receivers] )), L, dist )


def rician_distribution(batch_size, alpha):
    # Zero-mean Rician distribution to simulate fading for LOS channel
    x = np.random.normal(0, alpha, (batch_size, nNodes, nNodes))
    y = np.random.normal(0, alpha, (batch_size, nNodes, nNodes))
    samples = np.sqrt(x**2 + y**2)
    return samples


# Simuate Fading
def sample_graph(batch_size, A, alpha=1):
    if channel == 'nlos':
        samples = np.random.rayleigh(alpha, (batch_size, nNodes, nNodes))
    elif channel == 'los':
        samples = rician_distribution(batch_size, alpha)
    #samples = (samples + np.transpose(samples,(0,2,1)))/2
    PP = samples[None,:,:] * A
    return PP[0]

# Training Data
def generate_data(batch_size, alpha, A, nNodes):
    tr_H = []
    te_H = []
    
    for indx in range(tr_iter):
        # sample training data 
        H = sample_graph(batch_size, A, alpha )
        tr_H.append( H )

    for indx in range(te_iter):
        # sample test data
        A_dynamic = build_adhoc_network( nNodes )[1]
        H = sample_graph(batch_size, A_dynamic, alpha )
        te_H.append( H )

    return( dict(zip(['train_H', 'test_H'],[tr_H, te_H] ) ) )


def main():
    coord, A, dist = build_adhoc_network( nNodes )
    
    # Create data path
    if not os.path.exists('data/'+dataID):
        os.makedirs('data/'+dataID)
    
    # Training data
    data_H = generate_data(batch_size, alpha, A, nNodes)
    f = open('data/'+dataID+'/H.pkl', 'wb')
    pickle.dump(data_H, f)
    f.close()


if __name__ == '__main__':
    main()
