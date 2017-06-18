import numpy as np
import scipy.ndimage.filters as filters
import scipy.signal
import matplotlib.pyplot as plt
from samplers import *

class ratePredictor:
    def __init__(self,K,dt_max):
        # K processes, dt_max in number of time bins
        self.K = K
        self.dt_max = dt_max
    def predict(self,data,k):
        # data: K x dt_max 
        # k : output node
        raise NotImplementedError
def logist(x):
    return 1/(1+np.exp(-x))
class LogisticPredictor(ratePredictor):
    def __init__(self,K,dt_max,neta = 0.01):
        self.K = K
        self.dt_max = dt_max
        self.neta = neta
        self.W = 0.0001*(np.random.random((K,K*dt_max))-0.5)
        self.C = 0.0001*(np.random.random((K,1))-0.5)
    def predict(self,data):
        K = self.K
        dt_max = self.dt_max
        data = data.reshape(K * dt_max,1)
        xx = self.W.dot(data)
        xx = xx +self.C
        return logist(xx)
    def train(self,data,outcome):
        # constants
        K = self.K
        dt_max = self.dt_max
        neta = self.neta
        # predict
        data = data.reshape(K * dt_max,1)
        outcome.shape = K,1
        xx = self.W.dot(data)
        xx = xx +self.C
        yy = logist(xx)
        # SGD
        delt = outcome - yy
        delt = (1-yy) * yy * delt
        dw = np.outer(delt,data)
        self.W = self.W + neta*dw
        self.C = self.C + neta*delt
        # self.W = np.maximum(self.W,0)
class LogisticPositiveEdgePredictor(ratePredictor):
    def __init__(self,K,dt_max,neta = 1):
        self.K = K
        self.dt_max = dt_max
        self.neta = neta
        self.W = 0.0001*(np.random.random((K,K*dt_max))-0.5)
        self.C = -5*np.ones((K,1))
    def predict(self,data):
        K = self.K
        dt_max = self.dt_max
        data = data.reshape(K * dt_max,1)
        xx = self.W.dot(data)
        xx = xx +self.C
        return logist(xx)
    def train(self,data,outcome):
        # constants
        K = self.K
        dt_max = self.dt_max
        neta = self.neta
        # predict
        data = data.reshape(K * dt_max,1)
        outcome.shape = K,1
        xx = self.W.dot(data)
        xx = xx +self.C
        yy = logist(xx)
        # SGD
        delt = outcome - yy
        delt = (1-yy) * yy * delt
        dw = np.outer(delt,data)
        self.W = self.W + neta*dw
        self.W = np.maximum(self.W,0)
    def graph(self):
        K = self.K
        dt_max = self.dt_max
        out = np.zeros((K,K))
        temp = self.W.reshape(K,K,dt_max)
        out += np.sum(temp,axis=2)
        return out

class BackgroundLogisticPositiveEdgePredictor(ratePredictor):
    def __init__(self,K,dt_max,neta = 1,neta2 = 0.01):
        self.K = K
        self.dt_max = dt_max
        # learning rate for network
        self.neta = neta
        # learning rate for B
        self.neta2 = neta2
        self.W = 0.0001*(np.random.random((K,K*dt_max))-0.5)
        self.C = -5*np.ones((K,1))
        self.B = np.zeros((K,1))
    def predict(self,data):
        K = self.K
        dt_max = self.dt_max
        data = data.reshape(K * dt_max,1)
        xx = self.W.dot(data)
        xx = xx +self.C
        return logist(xx)+self.B
    def train(self,data,outcome):
        # constants
        K = self.K
        dt_max = self.dt_max
        neta = self.neta
        neta2= self.neta2
        # predict
        data = data.reshape(K * dt_max,1)
        outcome.shape = K,1
        xx = self.W.dot(data)
        xx = xx +self.C
        yy = logist(xx) + self.B
        # SGD
        delt = outcome - yy
        delt = (1-yy) * yy * delt
        dw = np.outer(delt,data)
        self.W = self.W + neta*dw
        self.W = np.maximum(self.W,0)
        self.B = self.B + neta2*(outcome - yy)
        self.B = np.maximum(self.B,0)
    def graph(self):
        K = self.K
        dt_max = self.dt_max
        out = np.zeros((K,K))
        temp = self.W.reshape(K,K,dt_max)
        out += np.sum(temp,axis=2)
        #print out
        #print self.B
        return temp,out
class doubleLogisticConvolutional(ratePredictor):
    def __init__(self,K,dt_max,num_basis = None,neta = 0.1,neta2 = 0.01):
        self.K = K
        self.dt_max = dt_max
        self.neta = neta
        self.neta2 = neta2
        if num_basis == None:
            num_basis = K*dt_max
        self.num_basis = num_basis
        # W1 K*dt_max -> num_basis
        self.W = 0.001 * (np.random.random((num_basis,K*dt_max))-0.5)
        # W2 num_basis -> K
        self.W2 = 0.001 * (np.random.random((K,num_basis))-0.5)
        self.C = 0.001*(np.random.random((num_basis,1))-0.5)
        self.C2 = 0.001*(np.random.random((K,1))-0.5)
    def predict(self,data):
        K = self.K
        dt_max = self.dt_max
        neta = self.neta
        neta2= self.neta2
        data = data.reshape(K * dt_max,1)
        # first layer, take time window

class BackgroundShallowLearningPositiveEdgePredictor(ratePredictor):
    def __init__(self,K,dt_max,num_basis = None,neta = 0.1,neta2 = 0.01):
        self.K = K
        self.dt_max = dt_max
        self.neta = neta
        self.neta2 = neta2
        if num_basis == None:
            num_basis = K*dt_max
        self.num_basis = num_basis
        # W1 K*dt_max -> num_basis
        self.W = 0.001 * (np.random.random((num_basis,K*dt_max))-0.5)
        # W2 num_basis -> K
        self.W2 = 0.001 * (np.random.random((K,num_basis))-0.5)
        self.C = 0.001*(np.random.random((num_basis,1))-0.5)
        self.C2 = 0.001*(np.random.random((K,1))-0.5)
    def predict(self,data):
        K = self.K
        dt_max = self.dt_max
        data = data.reshape(K * dt_max,1)
        xx = self.W.dot(data)
        xx = xx +self.C
        yy1 = logist(xx)
        xx2 = np.matmul(self.W2,yy1)
        xx2 = xx2 + self.C2
        yy2 = logist(xx2)
        return yy2

    def train(self,data,outcome):
        # constants
        K = self.K
        dt_max = self.dt_max
        neta = self.neta
        neta2= self.neta2
        data = data.reshape(K * dt_max,1)
        # predict    
        xx = self.W.dot(data)
        print 'C'
        print self.C
        xx = xx +self.C
        yy1 = logist(xx)
        print 'yy1'
        print yy1
        xx22 = self.W2.dot(yy1)
        xx2 = np.matmul(self.W2,yy1)
        print 'xx2'
        print xx2
        print 'C2'
        print self.C2
        xx2 = xx2 + self.C2
        yy2 = logist(xx2)
        # train
        print ""
        print yy2
        print outcome.reshape(K,1)
        delta2 = yy2-outcome.reshape(K,1)
        print "delta2",delta2.shape
        oo = (1-yy2) * yy2
        print "yy1",yy1.shape
        change2 = np.outer(oo*delta2,yy1)
        print "change2",change2.shape
        cchange2 = oo*delta2
        print "cchange2",cchange2.shape
        print "delta2",delta2.shape
        delta1 = self.W2.T.dot(delta2)
        print "W2", self.W2.shape
        print "delta1",delta1.shape
        oo1 = (1-yy1) * yy1
        print data.shape
        change1 = np.outer(oo1*delta1,data)
        print "change1",change1.shape
        cchange1 = oo1 * delta1.shape
        print "cchange1",cchange1.shape
        print "W", self.W.shape

        self.W = self.W - neta*change1
        self.C = self.C - neta*cchange1
        self.W2 = self.W2 - neta*change2
        self.C2 = self.C2 - neta*cchange2

    def graph(self):
        K = self.K
        dt_max = self.dt_max
        out = np.zeros((K,K))
        temp = self.W.reshape(K,K,dt_max)
        out += np.sum(temp,axis=2)
        print out
        print self.B
        return out



if __name__ == '__main__':
    a = BackgroundShallowLearningPositiveEdgePredictor(1,2,num_basis=4)
    data1 = np.array([[0, 0]])
    data2 = np.array([[1, 0]])
    data3 = np.array([[0, 1]])
    data4 = np.array([[1, 1]])
    out0 = np.array([0])
    out1 = np.array([1])
    np.random.seed(10)
    for i in range(0,1000):
        a.train(data1,out1)
        a.train(data2,out0)
        a.train(data3,out0)
        a.train(data4,out1)
    print a.predict(data1)
    print a.predict(data2)
    print a.predict(data3)
    print a.predict(data4)
