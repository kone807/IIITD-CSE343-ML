#!/usr/bin/env python
# coding: utf-8

# # CSE343 Assignment-2 | Q2 | 2019040

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder


def train_test_val_split(x,y,test_ratio,val_ratio):
    
    np.random.seed(0)
    random1 = np.random.rand(x.shape[0])
    split1 = random1 < np.percentile(random1,(1-test_ratio-val_ratio)*100)
    
    x_train = x[split1]
    y_train = y[split1]
    
    x_test_val = x[~split1]
    y_test_val = y[~split1]
    
    random2 = np.random.rand(x_test_val.shape[0])
    
    val_ratio = val_ratio/(val_ratio+test_ratio)
    split2 = random2 < np.percentile(random2,(1-val_ratio)*100)
    
    x_test = x_test_val[split2]
    y_test = y_test_val[split2]
    
    x_val = x_test_val[~split2]
    y_val = y_test_val[~split2]
    
    return x_train,x_test,x_val,y_train,y_test,y_val


# In[ ]:


# Layers of the NN
class Layer:
    
    def __init__(self):
       
        self.input_data = None
        self.output_data = None

    
    def forward_propagation(self, input_data):
        pass

    # computes derivative
    def backward_propagation(self, output_error, learning_rate):
        pass
        
## fully connected layer
class FCLayer(Layer):
    
    def __init__(self, input_size, output_size, weight_init):
        
        self.weight_init = weight_init
        self.weights, self.bias = self.init_weight(weight_init, input_size, output_size)
        
    def init_weight(self, weight_init, input_size, output_size):
        
        if weight_init == "zero":
            
            weights = np.zeros(input_size, output_size)
            bias = np.zeros(1, output_size)
            return weights, bias
        
        if weight_init == "random":
            
            weights = np.random.rand(input_size,output_size)*0.01
            bias = np.random.rand(1,output_size)*0.01
            return weights, bias
        
        if weight_init == "normal":
            
            weights = np.random.randn(input_size,output_size)*0.01
            bias = np.random.randn(1,output_size)*0.01
            return weights, bias
        
    # override function from parent class
    def forward_propagation(self, input_data):
        
        self.output_data = np.dot(input_data, self.weights) + self.bias
        self.input_data = input_data
        return self.output_data

    # calculate derivatives and update params (dW and dB)
    def backward_propagation(self, output_error, learning_rate):
        
        weights_error = np.dot(self.input_data.T, output_error)
        input_error = np.dot(output_error, self.weights.T)
        
        # update parameters
        self.bias -= learning_rate * output_error
        self.weights -= learning_rate * weights_error
        return input_error
    

## Activation layer
class ActivationLayer(Layer):
    
    def __init__(self, activation_name):
        
        self.activation_name = activation_name
      
    def activation_function(self,z,name):

        if name == "relu":
            return np.maximum(0,z)

        if name == "leaky_relu":

            ## choosing threshold 0.01 for leaky relu
            return np.where(z>0,z,z*0.01)

        if name == "sigmoid":
            return 1/(1+np.exp(-z))

        if name == "linear":
            return np.dot(z,self.weights)+self.bias
           
        if name == "tanh":
            return np.tanh(z)

        if name == "softmax":
            return np.exp(z)/np.sum(np.exp(z))
        
        if name == "identity":
            return z
        
    def gradient(self,z,name):    

        if name == "relu":
            return (z>0)*1

        if name == "leaky_relu":

            ## choosing threshold 0.01 for leaky relu
            derivative = np.ones_like(z)
            derivative[z<0]=0.01
            return derivative

        if name == "sigmoid":
            return self.activation_function(z,name)*(1-self.activation_function(z,name))

        if name == "linear": 
            return self.weights.T
           
        if name == "tanh":
            return 1-np.tanh(z)**2

        if name == "softmax":
            return self.activation_function(z,name)*(1-self.activation_function(z,name))
        
        if name == "identity":
            return 1
        
    # apply activation function on input
    def forward_propagation(self, input_data):
        
        self.output_data = self.activation_function(input_data,self.activation_name)
        self.input_data = input_data
        return self.output_data

    # take derivative of activation function to get dZ 
    def backward_propagation(self, output_error, learning_rate):
        return self.gradient(self.input_data,self.activation_name) * output_error


# In[ ]:


class MyNeuralNetwork:
    
    def __init__(self, n_layers, learning_rate, batch_size, num_epochs):
        
        self.layers = []
        
        self.n_layers = n_layers
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs

    ## calculate accuracy (both are 1D arrays)
    def score(self, y_pred, y_true):
        
        return np.mean(y_pred==y_true)
    
    # add new layer (like pytorch)
    def add(self, layer):
        self.layers.append(layer)

    ## cross entropy loss
    def loss(self,y_true, y_pred):
        
        return(np.mean(np.power(y_true-y_pred, 2)))

    ## derivative of cross entropy loss
    def loss_derivative(self,y_true, y_pred):
        
        return(2*(y_pred-y_true)/y_true.size)
      
    # predict output probabilities for given input
    def predict_proba(self, x):
        
        y_proba = []
        samples = len(x)
        
        for i in range(samples):
            
            # forward propagation
            output = x[i]
            
            for layer in self.layers:
                output = layer.forward_propagation(output)
                
            y_proba.append(output)

        return np.array(y_proba)

    ## predict the class from the probability
    def predict(self, x):
        
        y_pred = []
        y_proba = self.predict_proba(x)
        
        for i in range(len(y_proba)):
            y_pred.append(np.argmax(y_proba[i]))
            
        return np.array(y_pred)
        
    # fit function
    def fit(self, x_train, y_train, x_val, y_val):
        
        y_train_loss = []
        y_val_loss = []
        x_iter = []
        
        samples = len(x_train)
        
        for i in range(self.num_epochs):
            
            # loss is just used for printing and plotting
            loss = 0
            val_loss = 0
            # o and y store values for a batch and update it collectively
            o = []
            y = []
            
            for j in range(samples):
                
                if j<len(x_val):
                    output = x_val[j]
                    for layer in self.layers:
                        output = layer.forward_propagation(output)
                    val_loss += self.loss(y_val[j],output)
                    
                # forward propagation
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward_propagation(output)
                   
                # calculate loss
                loss += self.loss(y_train[j], output)
                o.append(output)
                y.append(y_train[j])
                    
                # update in batches
                if j%self.batch_size == 0:
                    # backward propagation
                    
                    # error contains the value dA
                    error = 0
                    for k in range(len(o)):
                        error += self.loss_derivative(y[k], o[k])
                        
                    error /= self.batch_size
                    for layer in reversed(self.layers):
                        error = layer.backward_propagation(error, self.learning_rate)
                        
                    o.clear()
                    y.clear()

            # calculate average error
            loss /= samples
            y_train_loss.append(loss)
            val_loss /= samples
            y_val_loss.append(val_loss)
            x_iter.append(i+1)
            print("epoch:",i+1,"/",self.num_epochs," error:",loss)
            
        return x_iter, y_train_loss, y_val_loss


# In[ ]:


import idx2numpy

def pre_process(x,y):
    
    x = x/255
    x = x.reshape(x.shape[0],1,-1)
    encoder = OneHotEncoder(sparse=False,categories="auto")
    y = encoder.fit_transform(y.reshape(len(y),-1))
    return x,y
    
## loading the dataset
x_train = idx2numpy.convert_from_file("train-images.idx3-ubyte")
x_test = idx2numpy.convert_from_file("t10k-images.idx3-ubyte")
y_train = idx2numpy.convert_from_file("train-labels.idx1-ubyte")
y_test = idx2numpy.convert_from_file("t10k-labels.idx1-ubyte")

x_train = x_train.reshape(60000,-1)
x_test = x_test.reshape(10000,-1)

x = np.concatenate((x_train,x_test),axis=0)
y = np.concatenate((y_train,y_test),axis=0)

x,y = pre_process(x,y)

x_train, x_test, x_val, y_train, y_test, y_val = train_test_val_split(x,y,0.2,0.1)


# In[ ]:


def run_model(activation_function):
    
    model = MyNeuralNetwork(n_layers=4, learning_rate=0.08, batch_size=1, num_epochs=150)

    model.add(FCLayer(784,256,"normal"))
    model.add(ActivationLayer(activation_function))
    
    model.add(FCLayer(256,128,"normal"))
    model.add(ActivationLayer(activation_function))

    model.add(FCLayer(128,64,"normal"))
    model.add(ActivationLayer(activation_function))

    model.add(FCLayer(64,32,"normal"))
    model.add(ActivationLayer(activation_function))

    model.add(FCLayer(32,10,"normal"))
    model.add(ActivationLayer("softmax"))

    x_iter, y_train_loss, y_val_loss = model.fit(x_train,y_train,x_val,y_val)
    
    ## loss plot
    plt.plot(x_iter, y_train_loss, label="train loss")
    plt.plot(x_iter, y_val_loss, label="validation loss")
    plt.legend()
    title = "loss plot using " + activation_function + " activation function"
    plt.title(title)
    plt.xlabel("iterations")
    plt.ylabel("loss")
    
    ## accuracy
    test_acc = model.score(model.predict(x_test),np.argmax(y_test,axis=1))
    val_acc = model.score(model.predict(x_val),np.argmax(y_val,axis=1))
    
    print("test acc using",activation_function,"is:",test_acc)
    print("val acc using",activation_function,"is:",val_acc)
    
    ## write weights and bias to a file
    weights = []
    bias = []

    for layer in model.layers:

        if hasattr(layer,"weights"):
            weights.append(layer.weights)
            bias.append(layer.bias)

    name = activation_function+".txt"
    textfile = open(name, "w")
    for element in weights:
        textfile.write(str(element) + "\n")
    for element in bias:
        textfile.write(str(element)+"\n")
    textfile.close()
    
## run model using different activation functions


# In[ ]:


run_model("tanh")


# In[ ]:


run_model("relu")


# In[ ]:


run_model("sigmoid")


# In[ ]:


run_model("leaky_relu")


# In[ ]:


run_model("identity")


# In[ ]:


## using inbuilt
from sklearn.neural_network import MLPClassifier

## loading the dataset
x_train = idx2numpy.convert_from_file("train-images.idx3-ubyte")/255
x_test = idx2numpy.convert_from_file("t10k-images.idx3-ubyte")/255
y_train = idx2numpy.convert_from_file("train-labels.idx1-ubyte")
y_test = idx2numpy.convert_from_file("t10k-labels.idx1-ubyte")

x_train = x_train.reshape(60000,-1)
x_test = x_test.reshape(10000,-1)

x = np.concatenate((x_train,x_test),axis=0)
y = np.concatenate((y_train,y_test),axis=0)

x_train, x_test, x_val, y_train, y_test, y_val = train_test_val_split(x,y,0.2,0.1)


# In[ ]:


def inbuilt_model(activation_function):
    
    model = MLPClassifier(hidden_layer_sizes=(256,128,64,32), activation=activation_function, batch_size="auto", learning_rate_init=0.01, max_iter=150, verbose=True)
    model.fit(x_train,y_train)
    
    print("test acc:",model.score(x_test,y_test))
    print("val acc:",model.score(x_val,y_val))


# In[ ]:


inbuilt_model("tanh")


# In[ ]:


inbuilt_model("relu")


# In[ ]:


inbuilt_model("logistic")


# In[ ]:


inbuilt_model("identity")


# In[ ]:


## try different learning rates on self implemented

def run_model_lr(lr):
    
    activation_function = "leaky_relu"
    
    model = MyNeuralNetwork(n_layers=4, learning_rate=lr, batch_size=1, num_epochs=50)

    model.add(FCLayer(784,256,"normal"))
    model.add(ActivationLayer(activation_function))

    model.add(FCLayer(256,128,"normal"))
    model.add(ActivationLayer(activation_function))

    model.add(FCLayer(128,64,"normal"))
    model.add(ActivationLayer(activation_function))

    model.add(FCLayer(64,32,"normal"))
    model.add(ActivationLayer(activation_function))

    model.add(FCLayer(32,10,"normal"))
    model.add(ActivationLayer("softmax"))

    x_iter, y_train_loss, y_val_loss = model.fit(x_train,y_train,x_val,y_val)
    
    ## loss plot
    plt.plot(x_iter, y_train_loss, label="train loss")
    plt.plot(x_iter, y_val_loss, label="validation loss")
    plt.legend()
    title = "loss plot using " + activation_function + " activation function"
    plt.title(title)
    plt.xlabel("iterations")
    plt.ylabel("loss")
    
    ## accuracy
    test_acc = model.score(model.predict(x_test),np.argmax(y_test,axis=1))
    val_acc = model.score(model.predict(x_val),np.argmax(y_val,axis=1))
    
    print("test acc using",activation_function,"is:",test_acc)
    print("val acc using",activation_function,"is:",val_acc)
    
    weights = []
    bias = []

    for layer in model.layers:

        if hasattr(layer,"weights"):
            weights.append(layer.weights)
            bias.append(layer.bias)

    name = activation_function+".txt"
    textfile = open(name, "w")
    for element in weights:
        textfile.write(str(element) + "\n")
    for element in bias:
        textfile.write(str(element)+"\n")
    textfile.close()


# In[ ]:


run_model_lr(0.001)


# In[ ]:


run_model_lr(0.01)


# In[ ]:


run_model_lr(0.1)


# In[ ]:


run_model_lr(1)

