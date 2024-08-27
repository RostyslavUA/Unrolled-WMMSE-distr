from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import pdb
import pickle
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from utils import consensus_matrix

np.random.seed(0)

# Eperiment
dataID = sys.argv[1]
dataID = 'set'+str(dataID)

# Number of nodes
nNodes = 15

# Path gain exponent
pl = 2.2

# Channel
channel = 'nlos'

# Rayleigh (NLOS) or Rician(LOS) distribution scale
alpha = 1

# Batch size
batch_size = 64

# Training iterations
tr_iter = 10000

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
        H = sample_graph(batch_size, A, alpha )
        te_H.append( H )

    return( dict(zip(['train_H', 'test_H'],[tr_H, te_H] ) ) )


def pl_uma_nlos_optional(batch_size, dist, fc=10):
    sigma_sf = 7.8  # dB
    path_loss = 32.4 + 20*np.log10(fc) + 30*np.log10(dist)
    shadow_fading = np.random.lognormal(0, sigma_sf, [batch_size, *path_loss.shape])
    path_loss = path_loss[np.newaxis] + shadow_fading
    return path_loss


def get_adj(sample, thr=0.001):
    adj_0 = np.ones_like(sample)
    adj_0[sample > thr] = 0.0
    adj_0 += adj_0.T
    adj_0 = np.divide(adj_0, adj_0, out=np.zeros_like(adj_0), where=adj_0 != 0)
    np.fill_diagonal(adj_0, 1.0)
    return adj_0


def generate_csi_and_operators(dataset_size, batch_size, dist, plot_save=False):
    tr_H, tr_adj, tr_cmat = [], [], []
    batch_H = np.zeros((batch_size, *dist.shape))
    batch_adj = np.zeros((batch_size, *dist.shape))
    batch_cmat = np.zeros((batch_size, *dist.shape))
    i = 0
    while True:
        # sample training data
        # H = pl_uma_nlos_optional(1, dist)
        H = sample_graph(1, dist)
        H = np.squeeze(H, 0)
        adj = get_adj(H)
        cmat = consensus_matrix(adj)[1].todense()
        if nx.is_connected(nx.Graph(adj)):
            batch_H[i] = H
            batch_adj[i] = adj
            batch_cmat[i] = cmat
            if plot_save:
                g = nx.Graph(adj)
                g.remove_edges_from(nx.selfloop_edges(g))
                if i == 0:
                    pos = nx.spring_layout(g)
                nx.draw_networkx(g, pos=pos)
                print(i)
                plt.savefig(f'output/fig/{i}.jpg')
                plt.clf()
                if i == 20:
                    import sys; sys.exit(0)
            i += 1
            if i == batch_size:
                i = 0
                tr_H.append(batch_H)
                tr_adj.append(batch_adj)
                tr_cmat.append(batch_cmat)
                batch_H = np.zeros((batch_size, *dist.shape))
                batch_adj = np.zeros((batch_size, *dist.shape))
                batch_cmat = np.zeros((batch_size, *dist.shape))
                print(len(tr_H))
        if len(tr_H) == dataset_size-1:
            break
    return tr_H, tr_adj, tr_cmat


def generate_training_data(batch_size, dist):
    tr_H, tr_adj, tr_cmat = generate_csi_and_operators(tr_iter, batch_size, dist)
    te_H, te_adj, te_cmat = generate_csi_and_operators(te_iter, batch_size, dist)
    data_H = dict(zip(['train_H', 'test_H'],[tr_H, te_H]))
    data_adj = dict(zip(['train_adj', 'test_adj'],[tr_adj, te_adj]))
    data_cmat = dict(zip(['train_cmat', 'test_cmat'],[tr_cmat, te_cmat]))
    return data_H, data_adj, data_cmat


def main():
    coord, A, dist = build_adhoc_network( nNodes )
    
    # Create data path
    if not os.path.exists('data/'+dataID):
        os.makedirs('data/'+dataID)

    # Coordinates of nodes
    f = open('data/'+dataID+'/coordinates.pkl', 'wb')  
    pickle.dump(coord, f)         
    f.close()

    # Geometric graph
    f = open('data/'+dataID+'/A.pkl', 'wb')  
    pickle.dump(A, f)          
    f.close()
    
    # Training data
    data_H, data_adj, data_cmat = generate_training_data(batch_size, A)
    f = open('data/'+dataID+'/H.pkl', 'wb')
    pickle.dump(data_H, f)
    f.close()
    f = open('data/'+dataID+'/adj.pkl', 'wb')
    pickle.dump(data_adj, f)
    f.close()
    f = open('data/'+dataID+'/cmat.pkl', 'wb')
    pickle.dump(data_cmat, f)
    f.close()


if __name__ == '__main__':
    main()
