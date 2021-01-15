import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from keras.utils.np_utils import to_categorical

df = pd.read_csv("fashion-mnist_train.csv")

df.head()

labels = df.label
X = df.drop(["label"], axis = 1)
labels = labels.to_numpy()

labels = to_categorical(labels)


from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
sc.fit(X)
X = sc.transform(X)

print(f"Shape of train set is {X.shape}")
print(f"Shape of train label is {labels.shape}")



class NeuralNet():
    def __init__(self, layers=[784, 32, 10], alpha = 0.01, num_iters = 100):
        self.params = self.init_weights()
        self.layers = layers
        self.alpha= alpha
        self.num_iters = num_iters
        self.loss = []
        self.sample_size = None
        self.X = None
        self.y = None
        
    def init_weights(self):
        input_layer=784
        hidden_1=32
        output_layer=10

        params = {
            'W1':np.random.randn(hidden_1, input_layer) * np.sqrt(1. / hidden_1),
            'W2':np.random.randn(output_layer, hidden_1) * np.sqrt(1. / output_layer),
            'b1':np.random.randn(32,),
            'b2':np.random.randn(10,)
        }

        return params

        
    def relu(self,Z):
        return np.maximum(0,Z)
    
    def sigmoid(self, x, derivative=False):
        if derivative:
            return (np.exp(-x))/((np.exp(-x)+1)**2)
        return 1/(1 + np.exp(-x))
    
    def softmax(self, x, derivative=False):
        exps = np.exp(x - x.max())
        if derivative:
            return exps / np.sum(exps, axis=0) * (1 - exps / np.sum(exps, axis=0))
        return exps / np.sum(exps, axis=0)

    def entropy_loss(self,y, h):
        nsample = len(y)
        loss = -1/nsample * (np.sum(np.multiply(np.log(h), y) + np.multiply((1 - y), np.log(1 - h))))
        return loss

    
    
    
    
    def forward_propagation(self):
        
        Z1 = np.dot(self.params["W1"], self.X) + self.params['b1']
        A1 = self.sigmoid(Z1)
        Z2 = A1.dot(self.params['W2']) + self.params['b2']
        A2 = self.softmax(Z2)
        # save calculated parameters     
        self.params['Z1'] = Z1
        self.params['Z2'] = Z2
        self.params['A1'] = A1
        self.params['A2'] = A2
        return A2
    
    def back_propagation(self, output):
        def dRelu(x):
            x[x<=0] = 0
            x[x>0] = 1
            return x
        weights_change = {}
        params = self.params

        error = 2 * (output - self.y) / output.shape[0] * self.softmax(params['Z2'], derivative=True)      
        weights_change['W2'] = np.outer(error, params['A1'])
        
        error = np.dot(params['W2'].T, error) * self.sigmoid(params['Z1'], derivative=True)
        weights_change['W1'] = np.outer(error, self.X)
        
        for key, value in weights_change.items():
            self.params[key] -= self.l_rate * value
        
        return weights_change

    
   
    '''    
        dl_wrt_h = -(np.divide(self.y,h) - np.divide((1 - self.y),(1-h)))
        dl_wrt_sig = h * (1-h)
        dl_wrt_z2 = dl_wrt_h * dl_wrt_sig

        dl_wrt_A1 = dl_wrt_z2.dot(self.params['W2'].T)
        dl_wrt_w2 = self.params['A1'].T.dot(dl_wrt_z2)
        dl_wrt_b2 = np.sum(dl_wrt_z2, axis=0)

        dl_wrt_z1 = dl_wrt_A1 * dRelu(self.params['Z1'])
        dl_wrt_w1 = self.X.T.dot(dl_wrt_z1)
        dl_wrt_b1 = np.sum(dl_wrt_z1, axis=0)

        #update the weights and bias
        self.params['W1'] = self.params['W1'] - self.alpha * dl_wrt_w1
        self.params['W2'] = self.params['W2'] - self.alpha * dl_wrt_w2
        self.params['b1'] = self.params['b1'] - self.alpha * dl_wrt_b1
        self.params['b2'] = self.params['b2'] - self.alpha * dl_wrt_b2
    '''
    

    
    def fit(self, X, y):
        self.X = X
        self.y = y
        self.init_weights() #initialize weights and bias


        for i in range(self.num_iters):
            h = self.forward_propagation()
            self.back_propagation(h)
#            self.loss.append(loss)
            
            
    def predict(self, X):
        Z1 = X.dot(self.params['W1']) + self.params['b1']
        A1 = self.relu(Z1)
        Z2 = A1.dot(self.params['W2']) + self.params['b2']
        pred = self.softmax(Z2)
        return np.round(pred) 
    
    def acc(self, y, h):
        acc = int(sum(y == h) / len(y) * 100)
        return acc


    def plot_loss(self):
        '''
        Plots the loss curve
        '''
        plt.plot(self.loss)
        plt.xlabel("Iteration")
        plt.ylabel("logloss")
        plt.title("Loss curve for training")
        plt.show()
        
    

nn = NeuralNet(X, y_label)
nn.fit(X, y_label)
nn.plot_loss()