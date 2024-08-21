import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def threshold_csi(samp, thr=0.01):
    """
    Threshold CSI, such that resulting adjacency is connected. Select the threshold through bisection:
    threshold(t+1) = threshold(t) + c threshold(t)/2, where c \in (-1, 1).
    :param samp: a CSI sample of the shape [M, M]
    :param thr: convergence threshold
    :return: thresholded, connected adjacency
    """
    h_frac = samp.max()
    h_frac_prev = h_frac
    attempt = 0
    thr_init = thr
    while True:
        adj_0 = np.ones_like(samp)
        adj_0[samp <= h_frac] = 0.0
        c = 1 if nx.is_connected(nx.Graph(adj_0)) else -1
        h_frac = h_frac + c * h_frac/2
        if abs(h_frac - h_frac_prev) < thr:
            adj = adj_0
            break
        else:
            h_frac_prev = h_frac
            attempt += 1
            if attempt == 100:  # Cannot converge. Increasing threshold
                thr += thr_init
                attempt = 0
    return adj


def hist_csi(train_H, batch_idx, hist_num=5):
    for i in range(hist_num):
        csi = np.array(np.array(train_H[batch_idx][i]))
        plt.figure()
        plt.hist(csi.flatten())
        plt.grid()
        plt.title(f"batch_idx: {batch_idx}, sample_idx: {i}")
        plt.show(block=False)
