from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import pickle
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix

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


def build_adhoc_network( nNodes, xy_lim=500, pl0=40, d0=10, gamma=3.0, std=7.0):
    # Define square of 1km x 1km
    # Generate coordinates for transmitters
    transmitters = np.random.uniform(low=-xy_lim, high=xy_lim, size=(nNodes,2))
    # Generate random radius for intended receivers
    r_vec = np.random.uniform(low=10, high=200, size=nNodes)
    a_vec = np.random.uniform(low=0, high=360, size=nNodes)
    # Calculate random delta coordinates for intended receivers
    xy_delta = np.zeros_like(transmitters)
    xy_delta[:, 0] = r_vec * np.sin(a_vec*np.pi/180)
    xy_delta[:, 1] = r_vec * np.cos(a_vec*np.pi/180)
    receivers = transmitters + xy_delta

    # Calculate the distance matrix between all pairs of transmitters and receivers
    d_mtx = distance_matrix(transmitters, receivers)
    pl0 = gen_pl_umi_nlos(d_mtx, fc=2.4, c=3e8, hbs=1.7, hut=1.7, gamma=gamma)
    h_mtx = lognormal_pathloss(d_mtx, pl0=pl0, d0=d0, gamma=gamma, std=std)
    return( dict(zip(['tx', 'rx'],[transmitters, receivers] )), h_mtx, d_mtx )


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
    tr_H, tr_adj = [], []
    te_H, te_adj = [], []
    indx_tr, indx_te = 0, 0
    while indx_tr < tr_iter:
        # sample training data
        coord, h_mtx, d_mtx = build_adhoc_network( nNodes, xy_lim=500, pl0=40, d0=10, gamma=2.0, std=7.82)
        H = sample_graph(batch_size, h_mtx, alpha )
        tr_H.append( H )
        indx_tr += 1
        # hist_pl(1/h_mtx)

    while indx_te < te_iter:
        # sample test data
        coord, h_mtx, d_mtx = build_adhoc_network( nNodes, xy_lim=500, pl0=40, d0=10, gamma=2.0, std=7.82)
        H = sample_graph(batch_size, h_mtx, alpha )
        te_H.append( H )
        indx_te += 1

    return( dict(zip(['train_H', 'test_H'],[tr_H, te_H] ) ), )


def gen_pl_umi_nlos(dist, fc=5.8, c=3e8, hbs=1.7, hut=1.7, gamma=3.0):
    he = 1.0
    dbp = 4*(hbs-he)*(hut-he)*fc*1e9/c
    pl1 = 32.4 + 21*np.log10(dist) + 20*np.log10(fc)
    pl2 = 32.4 + 40*np.log10(dist) + 20*np.log10(fc) - 9.5*np.log10(dbp**2 + (hbs - hut)**2)
    pl_umi_los = np.zeros_like(dist)
    pl_umi_los[dist < dbp] = pl1[dist < dbp]
    pl_umi_los[dist >= dbp] = pl2[dist >= dbp]
    h_umi_los_lin = lognormal_pathloss(dist, pl0=pl_umi_los, d0=10, gamma=gamma, std=4.0)
    pl_umi_los = -10*np.log10(h_umi_los_lin)

    pl_umi_nlos_prime = 35.3*np.log10(dist) + 22.4 + 21.3*np.log10(fc) - 0.3*(hut-1.5)
    pl_umi_nlos = np.maximum(pl_umi_los, pl_umi_nlos_prime)
    return pl_umi_nlos


def weighted_poisson_graph(N, area, radius=1.0):
    """
    Create a Poisson point process 2D graph
    """
    # N = np.random.poisson(lam=area*density)
    lenth_a = np.sqrt(area)
    xys = np.random.uniform(0, lenth_a, (N, 2))
    d_mtx = distance_matrix(xys, xys)
    adj_mtx = np.zeros([N, N], dtype=int)
    adj_mtx[d_mtx <= radius] = 1
    np.fill_diagonal(adj_mtx, 0)
    # graph = nx.from_numpy_matrix(adj_mtx)
    return adj_mtx, d_mtx, xys


def lognormal_pathloss(d_mtx, pl0=40, d0=10, gamma=3.0, std=7.0):
    '''
    PL = PL_0 + 10 \gamma \log_{10}\frac{d}{d_0} + X_g
    https://en.wikipedia.org/wiki/Log-distance_path_loss_model
    Args:
        d_mtx: distance matrix

    Returns:
        pl_mtx: path loss matrix
    '''
    # pl_mtx = np.ones_like(d_mtx)
    x_g = np.random.normal(0, std, size=d_mtx.shape)
    x_g = np.clip(x_g, 0-2*std, 0+2*std)
    pl_db_mtx = pl0 + 10.0 * gamma * np.log10(d_mtx/d0) + x_g
    h_mtx = 10.0 ** (-pl_db_mtx/10.0)
    return h_mtx


def hist_pl(pl_umi_nlos_lin):
    # Check if the rx signal - thermal noise > required snr
    pl_umi_nlos = 10*np.log10(pl_umi_nlos_lin)
    k = 1.380649e-23  # Boltzmann constant
    T = 290  # Kelvin
    bw = 5e6
    noise_power_lin = k*T*bw
    noise_power = 10*np.log10(noise_power_lin)  # dB
    snr_req = 30  # dB
    rx_sig = 0 - pl_umi_nlos - noise_power  # dB
    poor_channels = np.where(rx_sig < snr_req)  # path loss is very high due to large distance. Many poor channels
    plt.hist(rx_sig.flatten(), bins=100)
    plt.xlabel('RSS, dB')
    plt.ylabel('hits')
    plt.grid()
    plt.show()



def main():
    # Create data path
    if not os.path.exists('data/'+dataID):
        os.makedirs('data/'+dataID)

    # Training data
    data_H = generate_data(batch_size, alpha, nNodes)
    f = open('data/'+dataID+'/H.pkl', 'wb')
    pickle.dump(data_H, f)
    f.close()
    # f = open('data/'+dataID+'/adj.pkl', 'wb')
    # pickle.dump(data_adj, f)
    # f.close()


if __name__ == '__main__':
    main()
