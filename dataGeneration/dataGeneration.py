__author__ = 'Haohan Wang'

import numpy as np
import scipy

def generateSyntheticData(seed, n, p, d, g, sig, sigC, visualFlag=False):
    plt = None
    if visualFlag:
        from matplotlib import pyplot as plt

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

    if visualFlag:
        plt.imshow(X)
        plt.show()

    Z = []
    for i in range(g):
        for j in range(sample):
            Z.append(center[i,:]+np.random.normal(0, 0.1, size=[p]))
    Z = np.array(Z)

    # Z = []
    # for i in range(g):
    #     m = np.zeros([g])
    #     m[i] = 1
    #     for j in range(sample):
    #         Z.append(m)
    # Z = np.array(Z)


    featureNum = int(p * dense)
    idx = scipy.random.randint(0,p,featureNum).astype(int)
    idx = sorted(idx)
    w = np.random.normal(0, 1, size=featureNum)
    yp = scipy.dot(X[:,idx],w)
    yp = yp.reshape(yp.shape[0])

    Cy = np.dot(Z, Z.T)

    if visualFlag:
        plt.imshow(Cy)
        plt.show()

    causal = np.array(zip(idx, w))

    yK = np.random.multivariate_normal(yp, sigC * Cy, size=1)
    yK = yK.reshape(yK.shape[1])
    yK = yp

    # v = np.random.normal(size=[g, 1])
    # yK = yp + np.dot(Z, v).reshape(yp.shape[0])*wz

    if visualFlag:
        index = np.arange(yK.shape[0])
        plt.scatter(index, yK)
        plt.show()

    np.savetxt('../data/X.csv', X, delimiter=',')
    np.savetxt('../data/y.csv', yK, delimiter=',')
    np.savetxt('../data/Z.csv', Z, delimiter=',')

    if visualFlag:
        from mpl_toolkits.mplot3d import Axes3D
        from sklearn.decomposition import PCA
        clf = PCA(n_components=2)
        X = clf.fit_transform(X)
        fig = plt.figure(dpi=100)
        ax = fig.add_subplot(111, projection='3d')
        for i in range(g):
            ax.scatter(X[i*sample:(i+1)*sample,0],X[i*sample:(i+1)*sample,1],yK[i*sample:(i+1)*sample],marker="o")
        ax.set_xlabel('1st PC of X')
        ax.set_ylabel('2nd PC of X')
        ax.set_zlabel('y')
        plt.show()

    beta = np.zeros([p])
    beta[causal[:,0].astype(int)] = causal[:,1]

    np.savetxt('../data/beta.csv', beta, delimiter=',')

if __name__ == '__main__':
    generateSyntheticData(seed=0, n=100, p=2, d=0.5, g=5, sig=1, sigC=0.1, visualFlag=True)
