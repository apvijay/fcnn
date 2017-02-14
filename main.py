import numpy as np
import cPickle, gzip



class fcnn():
    def __init__(self):
        # num_layers = input + hidden + output
        self.nhidden = 1
        self.nlayers = self.nhidden + 2
        self.ninputs = 28*28
        self.noutputs = 10
        self.nweights = self.nlayers - 1
        self.nnodes = self.ninputs

        self.W = [None] * self.nweights # empty list
        self.b = [None] * self.nweights # empty list
        for k in range(self.nweights-1): # 0,1,2,...,num_weights-2
            self.W[k] = np.random.randn(self.nnodes, self.nnodes)/np.sqrt(self.nnodes*self.noutputs/2)
            self.b[k] = np.random.randn(1,self.nnodes)/np.sqrt(self.nnodes/2)
        self.W[self.nweights-1] = np.random.randn(self.nnodes,self.noutputs)/np.sqrt(self.nnodes*self.noutputs/2)
        self.b[self.nweights-1] = np.random.randn(1,self.noutputs)/np.sqrt(self.nnodes/2)

        self.sgd_mom_upd_W = [None] * self.nweights
        self.sgd_mom_upd_b = [None] * self.nweights
        for k in range(self.nweights-1): # 0,1,2,...,num_weights-2
            self.sgd_mom_upd_W[k] = np.zeros((self.nnodes, self.nnodes))
            self.sgd_mom_upd_b[k] = np.zeros((1,self.nnodes))
        self.sgd_mom_upd_W[self.nweights-1] = np.zeros((self.nnodes,self.noutputs))
        self.sgd_mom_upd_b[self.nweights-1] = np.zeros((1,self.noutputs))

        self.batch_size = 1
        self.max_epochs = 50
        self.max_batches = self.ninputs / self.batch_size
        self.optim_method = 'sgd'
        self.eta = 1*1e-5
        self.gamma = 0.99

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def sigmoid_diff(self, x):
        return np.multiply(self.sigmoid(x), (1 - self.sigmoid(x)))

    def softmax(self,x):
        xd = x - np.max(np.abs(x))
        return np.exp(xd) / np.sum(np.exp(xd))

    def forward(self, x):
        a = [None] * self.nweights
        h = [None] * (self.nweights-1)
        for k in range(self.nweights-1): 
            a[k] = np.zeros((1,self.nnodes))
            h[k] = np.zeros((1,self.nnodes))
        a[self.nweights-1] = np.zeros((1,self.noutputs))
        for k in range(self.nweights-1):
            if k == 0:
                a[k] = self.b[k] + np.dot(x,self.W[k])
            else:
                a[k] = self.b[k] + np.dot(h[k-1],self.W[k])
            h[k] = self.sigmoid(a[k])
        a[self.nweights-1] = self.b[self.nweights-1] + np.dot(h[self.nweights-2],self.W[self.nweights-1])
        ycap = self.softmax(a[self.nweights-1])
        return (h,a,ycap)

    def optim(self, set_x, set_y):
        for i in range(self.max_epochs):
            dWsum = [None] * self.nweights
            dbsum = [None] * self.nweights
            for k in range(self.nweights-1): # 0,1,2,...,num_weights-2
                dWsum[k] = np.zeros((self.nnodes, self.nnodes))
                dbsum[k] = np.zeros((1,self.nnodes))
            dWsum[self.nweights-1] = np.zeros((self.nnodes,self.noutputs))
            dbsum[self.nweights-1] = np.zeros((1,self.noutputs))
            err = 0
            acc = 0
            for j in range(self.max_batches):
                
                x,y = read_batch(set_x, set_y, j, self.batch_size)
                (h,a,ycap) = self.forward(x)
                (dW,db) = self.backward(h,a,ycap,y)
                err += -np.log(ycap[0,np.int32(y)])
                acc += np.double(np.argmax(ycap) == y)
                for k in range(self.nweights):
                    dWsum[k] += dW[k]
                    dbsum[k] += db[k]
                #if j%100 == 0:
                #    print('Epoch {}, Minibatch {} err: {}'.format(i,j,err))

            for k in range(self.nweights):
                if self.optim_method == 'sgd':
                    self.W[k] -= self.eta * dWsum[k]/self.batch_size
                    self.b[k] -= self.eta * dbsum[k]/self.batch_size
                elif self.optim_method =='sgdm':
                    self.sgd_mom_upd_W[k] = self.gamma * self.sgd_mom_upd_W[k] + self.eta * dWsum[k]/self.batch_size
                    self.sgd_mom_upd_b[k] = self.gamma * self.sgd_mom_upd_b[k] + self.eta * dbsum[k]/self.batch_size                    
                    self.W[k] -= self.sgd_mom_upd_W[k]
                    self.b[k] -= self.sgd_mom_upd_b[k]
                    

            print('Epoch {} err: {}, acc: {}'.format(i,err/self.max_batches,acc*100/self.ninputs))


    def backward(self,h,a,ycap,y): 
        da = [None] * self.nweights
        dh = [None] * (self.nweights-1)
        for k in range(self.nweights-1): 
            da[k] = np.zeros((1,self.nnodes))
            dh[k] = np.zeros((1,self.nnodes))
        da[self.nweights-1] = np.zeros((1,self.noutputs))

        dW = [None] * self.nweights
        db = [None] * self.nweights
        for k in range(self.nweights-1):
            dW[k] = np.zeros((self.nnodes, self.nnodes))
            db[k] = np.zeros((1,self.nnodes))
        dW[self.nweights-1] = np.zeros((self.nnodes,self.noutputs))
        db[self.nweights-1] = np.zeros((1,self.noutputs))

        ey = np.zeros((1,self.noutputs))
        ey[0,np.int32(y)] = 1.0 
        da[self.nweights-1] = -(ey[0,np.int32(y)] - ycap)
        for k in range(self.nweights-1,-1,-1):
            dW[k] = np.dot(h[k-1].T,da[k])
            db[k] = da[k]
            if k-1 >= 0:
                dh[k-1] = np.dot(da[k],self.W[k].T)
                da[k-1] = np.multiply(dh[k-1], self.sigmoid_diff(a[k-1]))
        return (dW,db)

def read_batch(set_x, set_y, index, batch_size):
    x = np.asarray(set_x[(index) * batch_size: (index+1) * batch_size])
    y = np.asarray(set_y[(index) * batch_size: (index+1) * batch_size],dtype='float32')
    return (x,y)

def load_mnist(data_path):
    f = gzip.open(data_path, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    train_set_x, train_set_y = train_set
    valid_set_x, valid_set_y = valid_set
    test_set_x, test_set_y = test_set

    #train_set_x /= np.std(train_set_x,axis=0)
    #valid_set_x /= np.std(train_set_x,axis=0)
    #test_set_x /= np.std(train_set_x,axis=0)
    
    train_set_x -= np.mean(train_set_x,axis=0)
    valid_set_x -= np.mean(train_set_x,axis=0)
    test_set_x -= np.mean(train_set_x,axis=0)

    return (train_set_x, train_set_y, valid_set_x, valid_set_y, test_set_x, test_set_y)

def train(model,train_set_x,train_set_y):
  model.optim(train_set_x,train_set_y)

def test():
    pass


train_set_x, train_set_y, valid_set_x, valid_set_y, test_set_x, test_set_y = load_mnist('mnist.pkl.gz')
model = fcnn()
#model.forward(train_set_x[0:1])
train(model,train_set_x,train_set_y)
#test(model)
