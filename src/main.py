import numpy as np
import scipy.ndimage.filters as filters
import scipy.signal
import matplotlib.pyplot as plt
from samplers import *
from predictors import *
from hawkes_generator import * 
class Model:
    def __init__(self,data,T,K,dt_max):
        self.data = data
        self.T = T
        self.K = K
        self.dt_max = dt_max
class ConvolutionalTrainer(Model):
    def __init__(self,data,T,K,dt_max):
        self.data = data
        self.T = T
        self.K = K
        self.dt_max = dt_max
        self.Predictor = BackgroundLogisticPositiveEdgePredictor(K,dt_max)
        self.rateSampler = GaussianMeanSampler(sigma = 50, mean = 0.5)
        self.resample_rates()
    def train(self):
        self.resample_rates()
        for i in range(500):
            print i
            self.train_epoch()
    def train_epoch(self):
        # data is K x dt_max, rate is 
        dt_max = self.dt_max
        data = self.data
        for i in range(dt_max,self.T):
            self.Predictor.train(data[0:self.K,i-dt_max:i],self.rate[:,i])
    def predict(self):
        dt_max = self.dt_max
        pad = np.zeros((self.K,self.dt_max))
        data = np.hstack((pad,self.data))
        predicted_rate = np.zeros(self.data.shape)
        for i in range(dt_max,dt_max+self.T):
            cpr = self.Predictor.predict(data[0:self.K,i-dt_max:i])
            predicted_rate[0:self.K,i-dt_max] = cpr.reshape(self.K)
        return predicted_rate
    def resample_rates(self):
        self.rate = self.rateSampler.sample(self.data.T).T
    def resample_rates1(self):
        B = self.Predictor.B
        self.rate = self.rateSampler.sampleWithBG(self.data.T,B).T

 
class DiscreteLogisticModel(Model):
    def __init__(self,data,T,K,dt_max):
        self.data = data
        self.T = T
        self.K = K
        self.dt_max = dt_max
        self.Predictor = BackgroundLogisticPositiveEdgePredictor(K,dt_max)
        self.rateSampler = GaussianMeanSampler(sigma = 50, mean = 0.5)
        self.resample_rates()
    def train(self):
        self.resample_rates()
        for i in range(500):
            print i
            self.train_epoch()
    def train_epoch(self):
        dt_max = self.dt_max
        data = self.data
        for i in range(dt_max,self.T):
            self.Predictor.train(data[0:self.K,i-dt_max:i],self.rate[:,i])
    def predict(self):
        dt_max = self.dt_max
        pad = np.zeros((self.K,self.dt_max))
        data = np.hstack((pad,self.data))
        predicted_rate = np.zeros(self.data.shape)
        for i in range(dt_max,dt_max+self.T):
            cpr = self.Predictor.predict(data[0:self.K,i-dt_max:i])
            predicted_rate[0:self.K,i-dt_max] = cpr.reshape(self.K)
        return predicted_rate
    def resample_rates(self):
        self.rate = self.rateSampler.sample(self.data.T).T
    def resample_rates1(self):
        B = self.Predictor.B
        self.rate = self.rateSampler.sampleWithBG(self.data.T,B).T

            

T = 20000
K = 20
p = 0.05
dt_max = 40

np.random.seed(1240)
data = np.random.random(K*T).reshape(K,T)
W, data = gen(K,T)

M = DiscreteLogisticModel(data,T,K,dt_max)
M.train()
pdr = M.predict()
rt = M.rate
pulse,WW = M.Predictor.graph()

ind = np.diag_indices(K)
WW[ind] = 0
www = np.sort(WW.reshape(K*K))
A = W>0
TPR = []
FPR = []
for i in range(K*K):
    TP = A & (WW >= www[i])
    TN = ~A & (WW < www[i])
    FP = ~A & (WW >= www[i])
    FN = A & (WW < www[i])
    tp,fp = sum(sum(TP)),sum(sum(FP))
    tn,fn = sum(sum(TN)),sum(sum(FN))
    #print tp,fp,tn,fn
    tpr = float(tp)/float(tp+fn)
    TPR = TPR + [tpr]
    fpr = float(fp)/float(tn+fp) 
    FPR = FPR + [fpr]
plt.scatter(FPR,TPR)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()

for i in range(K):
    for j in range(K):
        if W[i,j] > 0:
            plt.plot(pulse[i,j,0:dt_max][::-1])
            plt.show()

f, (ax0,ax1, ax2,ax3) = plt.subplots(4, 1, sharex=True)
ax0.plot(data[0,:])
ax0.plot(pdr[0,:])
ax0.plot(rt[0,:])
ax1.plot(data[1,:])
ax1.plot(pdr[1,:])
ax1.plot(rt[1,:])
ax2.plot(data[2,:])
ax2.plot(pdr[2,:])
ax2.plot(rt[2,:])
ax3.plot(data[3,:])
ax3.plot(pdr[3,:])
ax3.plot(rt[3,:])
plt.show()
#a = LogisticPredictor(K,T)
'''
'''
'''
'''
