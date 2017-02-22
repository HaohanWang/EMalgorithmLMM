__author__ = 'Haohan Wang'

import numpy as np
import scipy

def generateSyntheticData(seed, n, p, d, g, sig, sigC):
    np.random.seed(seed)

    dense = d

    n = n
    p = p
    g = g
    sig = sig
    sigC = sigC
    wz = 1e-3

    center = np.random.normal(0, 1, [g,p])
    sample = n/g
    X = []

    for i in range(g):
        x = np.random.multivariate_normal(center[i,:], sig*np.diag(np.ones([p,])), size=sample)
        X.extend(x)
    X = np.array(X)
    print X.shape

    Z = []
    for i in range(g):
        for j in range(sample):
            Z.append(center[i,:]+np.random.normal(0, 0.1, size=[p]))
    Z = np.array(Z)

    featureNum = int(p * dense)
    idx = scipy.random.randint(0,p,featureNum).astype(int)
    idx = sorted(idx)
    w = np.random.normal(0, 1, size=featureNum)
    yp = scipy.dot(X[:,idx],w)
    yp = yp.reshape(yp.shape[0])

    Cy = np.dot(Z, Z.T)

    causal = np.array(zip(idx, w))

    yK = np.random.multivariate_normal(yp, sigC * Cy, size=1)
    yK = yK.reshape(yK.shape[1])

    np.savetxt('../data/X.csv', X, delimiter=',')
    np.savetxt('../data/y.csv', yK, delimiter=',')
    np.savetxt('../data/Z.csv', Z, delimiter=',')

    beta = np.zeros([p])
    beta[causal[:,0].astype(int)] = causal[:,1]

    np.savetxt('../data/beta.csv', beta, delimiter=',')

if __name__ == '__main__':
    generateSyntheticData(seed=0, n=100, p=2, d=0.5, g=5, sig=1, sigC=0.1)
