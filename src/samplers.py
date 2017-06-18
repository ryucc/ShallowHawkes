import numpy as np
import scipy.ndimage.filters as filters
import scipy.signal
import matplotlib.pyplot as plt

class rateSampler:
    def __init__(self):
        pass
    def sample(self,raw_data):
        raise NotImplementedError

class GaussianMeanSampler(rateSampler):
    def __init__(self,sigma=20,mean=0.5):
        self.sigma = sigma
        self.mean = mean
    def sample(self,raw_data):
        mean = self.mean
        inducted_rate = filters.gaussian_filter1d(raw_data,self.sigma,axis=0)
        total_rate = np.sum(raw_data,axis=0)/raw_data.shape[0]
        return inducted_rate*(1-mean) + mean * total_rate
    def sampleWithBG(self,raw_data,bg_rate):
        total_rate = np.sum(raw_data,axis=0)/raw_data.shape[0]
        print total_rate/bg_rate
        exit(0)

class ConvolveKernelSampler(rateSampler):
    def __init__(self,kernel=None):
        if kernel == None:
            kernel = scipy.signal.gaussian( M = 200,std = 100)
            kernel = kernel/sum(kernel)
        self.kernel = kernel
    def sample(self,raw_data):
        ret = np.zeros(raw_data.shape)
        for i in range(raw_data.shape[1]):
            ret[:,i] = np.convolve(raw_data[:,i],self.kernel,'same')
        return ret

class GaussianParentSampler(rateSampler):
    def __init__(self):
        raise NotImplementedError
    def sample(self,raw_data):
        raise NotImplementedError


if __name__=='__main__':
    a = np.random.random(1000);
    a.shape= 1000,1
    t = a>0.05
    a[t] = 0
    a[~t] = 1
    sam = GaussianMeanSampler(50,0.2)
    b = sam.sample(a)
    print b.shape

    plt.plot(a[:,0])
    plt.plot(b[:,0])
    plt.show()
