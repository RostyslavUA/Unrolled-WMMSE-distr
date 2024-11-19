from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import pickle
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

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
def generate_data(batch_size, alpha, nNodes):
    tr_H = []
    te_H = []
    
    for indx in range(tr_iter):
        # sample training data
        A_dynamic = build_adhoc_network( nNodes )[1]
        H = sample_graph(batch_size, A_dynamic, alpha )
        tr_H.append( H )

    for indx in range(te_iter):
        # sample test data
        A_dynamic = build_adhoc_network( nNodes )[1]
        H = sample_graph(batch_size, A_dynamic, alpha )
        te_H.append( H )

    return( dict(zip(['train_H', 'test_H'],[tr_H, te_H] ) ) )


def gen_pl_nlos_umi(nLoc, nShadow, nNodes, area=1000, fc=5.8, c=3e8, hbs=1.7, hut=1.7):
    he = 1.0
    dbp = 4*(hbs-he)*(hut-he)*fc*1e9/c
    tx_loc = np.random.uniform(0, 1, (nLoc, nNodes, 2))*np.sqrt(area)
    rx_loc = np.random.uniform(0, 1, (nLoc, nNodes, 2))*np.sqrt(area)

    dist = np.sqrt(np.sum((tx_loc[:, :, np.newaxis, :] - rx_loc[:, np.newaxis, :, :])**2, 3))
    dist[dist < 10] = 10
    dist[dist > 5000] = 5000
    pl1 = 32.4 + 21*np.log10(dist) + 20*np.log10(fc)
    pl2 = 32.4 + 40*np.log10(dist) + 20*np.log10(fc) - 9.5*np.log10(dbp**2 + (hbs - hut)**2)
    pl_umi_los = np.zeros_like(dist)
    pl_umi_los[dist < dbp] = pl1[dist < dbp]
    pl_umi_los[dist >= dbp] = pl2[dist >= dbp]
    shadowing_los = np.random.normal(0, 4, (nShadow, nNodes, nNodes))
    pl_umi_los = pl_umi_los[:, np.newaxis] + shadowing_los[np.newaxis, :]

    pl_umi_nlos_prime = 35.3*np.log10(dist) + 22.4 + 21.3*np.log10(fc) - 0.3*(hut-1.5)
    pl_umi_nlos = np.maximum(pl_umi_los, pl_umi_nlos_prime[:, np.newaxis])
    shadowing_nlos = np.random.normal(0, 7.82, (nShadow, nNodes, nNodes))
    pl_umi_nlos += shadowing_nlos[np.newaxis]
    pl_umi_nlos_lin = 10**(pl_umi_nlos/10)

    # Check if the rx signal - thermal noise > required snr
    # pl_umi_nlos = 10*np.log10(pl_umi_nlos_lin)
    # k = 1.380649e-23  # Boltzmann constant
    # T = 290  # Kelvin
    # bw = 5e6
    # noise_power_lin = k*T*bw
    # noise_power = 10*np.log10(noise_power_lin)  # dB
    # snr_req = 30  # dB
    # rx_sig = 0 - pl_umi_nlos - noise_power  # dB
    # poor_channels = np.where(rx_sig < snr_req)  # typically a few values
    # plt.hist(rx_sig.flatten(), bins=100)
    # plt.xlabel('bins')
    # plt.ylabel('RSS, dB')
    # plt.grid()
    # plt.show()
    return dist, pl_umi_nlos_lin


def gen_adjacency(pl_umi_nlos_lin, dist, pl_thr=90.0, dist_thr=25.0):
    nLoc, nShadow, nFading, nNodes, _ = pl_umi_nlos_lin.shape
    dist = np.broadcast_to(dist, (nShadow, nFading, nLoc, nNodes, nNodes))
    dist = np.transpose(dist, (2, 0, 1, 3, 4))
    pl_umi_nlos = 10*np.log10(pl_umi_nlos_lin)
    adj = np.ones_like(pl_umi_nlos_lin)
    adj[pl_umi_nlos > pl_thr] = 0.0
    adj[dist > dist_thr] = 0.0
    adj = adj * np.transpose(adj, (0, 1, 2, 4, 3))  # Make symmetric
    return adj


def gen_fading(nFading, pl_umi_nlos_lin, alpha=1.0):
    nLoc, nShadow, nNodes, _ = pl_umi_nlos_lin.shape
    fading = np.random.rayleigh(alpha, (nLoc, nShadow, nFading, nNodes, nNodes))
    pl_umi_nlos_lin = pl_umi_nlos_lin[:, :, np.newaxis] * fading
    return pl_umi_nlos_lin


def main():
    # coord, A, dist = build_adhoc_network( nNodes )
    nLoc = 10
    nShadow = 64
    nFading = 5
    dist, pl_umi_nlos_lin = gen_pl_nlos_umi(nLoc, nShadow, nNodes, area=1000)
    pl_umi_nlos_lin = gen_fading(nFading, pl_umi_nlos_lin, alpha=alpha)
    adj = gen_adjacency(pl_umi_nlos_lin, dist)

    # con = 0
    # for a in np.reshape(adj, (-1, nNodes, nNodes)):
    #     g = nx.from_numpy_matrix(a)
    #     if nx.is_connected(g):
    #         con+=1
    # print(con)  # all the graphs are connected
    # TODO: double-check, reshape and save adj with pl_umi_nlos_lin if everything is correct

    # # Create data path
    # if not os.path.exists('data/'+dataID):
    #     os.makedirs('data/'+dataID)


if __name__ == '__main__':
    main()
