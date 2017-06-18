import numpy as np
import matplotlib.pyplot as plt


def gen(K=4,T=5000,dt_max=100):
    xx = np.arange(dt_max,dtype='double')
    impulse = ((dt_max-xx)/dt_max)
    plt.plot(impulse)
    plt.show()
    W = np.random.random((K,K))
    o = W < 0.05
    W[o] = 1
    W[~o] = 0
    W = W * 0.3 * np.random.random((K,K))
    W = np.maximum(W,0)
    for i in range(K):
        W[i,i] = 0
    out = 0.005*np.ones((K,T))
    result = np.zeros((K,T))
    a = np.ones((K,1))
    a[1] = 0
    for i in range(T):
        dice = np.random.random((K,1))
        events = dice < out[0:K,i:i+1]
        trigger = W.dot(events)
        b = min(dt_max,T-i)
        out[0:K,i:i+b] = out[0:K,i:i+b]+ trigger * impulse[0:b]
        result[:,i:i+1] = events
    return W,result
        


if __name__ == '__main__':
    print gen()

