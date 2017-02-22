__author__ = 'Haohan Wang'

import numpy as np

def p(A, B):
    return np.dot(A, B)

def mse(a, b):
    return np.mean(np.square(a - b))


class LMM:
    def __init__(self, X, y, Z, eps=1e-5):
        self.X = X
        self.y = y
        self.Z = Z
        self.eps = eps

        [self.n, self.p] = self.X.shape
        [self.n, self.q] = self.Z.shape
        self.y = self.y.reshape([self.n, 1])

        self.beta = np.zeros([self.p, 1])
        self.u = np.zeros([self.q, 1])
        #
        # tmpM = np.append(self.X, self.Z, 1)
        # tmp = p(p(np.linalg.inv(p(tmpM.T, tmpM)), tmpM.T), self.y)
        # self.beta = tmp[:self.p]
        # self.u = tmp[self.p:]

        self.su = 1  # sigma_u^2
        self.s0 = 1  # sigma_epsilon^2

    def setBeta(self, b):
        self.b = b

    def loglikelihood(self):
        r = self.y - p(self.X, self.beta) - p(self.Z, self.u)

        return -0.5 * self.n * np.log(self.s0) - 0.5 * self.q * np.log(self.su) - 0.5 * p(r.T, r) / self.s0 - 0.5 * p(
            self.u.T, self.u) / self.su

    def em(self):
        lls = []
        bs = []
        pll = -np.inf
        ll = self.loglikelihood()
        print ll
        ZZt = np.dot(self.Z, self.Z.T)
        while ll - pll > self.eps:
            lls.append(ll[0])
            bs.append(mse(self.beta, self.b))
            pll = ll
            # E step
            R = self.y - np.dot(self.X, self.beta)
            V = ZZt * self.su + self.s0
            iV = np.linalg.pinv(V)
            utu = self.su * self.su * (p(p(p(p(R.T, iV), ZZt), iV), R)) + np.trace(
                self.su - self.su * self.su * (p(p(self.Z.T, iV), self.Z)))
            w = p(self.X, self.beta) + self.s0 * (p(iV, R))
            ete = self.s0 * self.s0 * (p(p(p(R.T, iV), iV), R)) + np.trace(self.s0 - self.s0 * self.s0 * (iV))

            # M step
            self.s0 = ete / self.n
            self.su = utu / self.q
            self.beta = p(p(np.linalg.pinv(p(self.X.T, self.X)), self.X.T), w)

            ll = self.loglikelihood()

        return lls, bs

if __name__ == '__main__':
    X = np.loadtxt('data/X.csv', delimiter=',')
    y = np.loadtxt('data/y.csv', delimiter=',')
    Z = np.loadtxt('data/Z.csv', delimiter=',')
    b = np.loadtxt('data/beta.csv', delimiter=',')

    lmm = LMM(X=X, y=y, Z=X)
    lmm.setBeta(b)
    lls, bs = lmm.em()

    from matplotlib import pyplot as plt
    x = xrange(len(bs))
    plt.plot(x, lls)
    plt.xlabel('iteration')
    plt.ylabel('log likelihood')
    plt.savefig('figs/ll.png')
    plt.clf()
    plt.plot(x, bs)
    plt.xlabel('iteration')
    plt.ylabel('mse')
    plt.savefig('figs/mse.png')
    plt.clf()