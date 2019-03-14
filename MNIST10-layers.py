import math

import numpy as np

import random

import os

def w_list_b_list(n,X):#input a list that contains row's value of each W

    c = X.T.shape[0]

    w_list=[]

    b_list=[]

    for i in n:

        w_list.append(np.random.normal(0,1/np.sqrt(c),(i,c)))

        b_list.append(np.random.normal(0,1/np.sqrt(c),(i,1)))

        c = i

    return w_list,b_list

def forward_propagation(X,w_list,b_list,z_list,h_list):#z_list and h_list =[].

    z = np.copy(X.T)

    for i in range(len(w_list)):

        if i!=len(w_list)-1:

            z = w_list[i].dot(z)+b_list[i]

            z_list.append(z)

            z[z<0]=0

            h_list.append(z)

        else:

            z = w_list[i].dot(z)+b_list[i]

            y_hat = np.exp(z)/np.sum(np.exp(z),axis=0)

    return z_list,h_list,y_hat
            
'''
we already have a list of matrix h;a list of matrix w; a list of matrix z,z_list.
'''
def jacobian_df_dz(y,y_hat):#y and y_hat are 10*1 vector

    dy_dz = (-1/(y.shape[0]))*(y.T-y_hat)

    return dy_dz

def jacobian_dh_dz(z):# input: vector z

    dh_dz = np.copy(z)

    dh_dz[dh_dz <= 0] = 0

    dh_dz[dh_dz > 0] = 1

    return dh_dz
    
def back_propagation(y,y_hat,z_list,h_list,w_list,b_list,X,learning_rate,alpha):#y,y_hat as vectors, z_list,w,b,h is a list contains all w matrix.

    p = jacobian_df_dz(y, y_hat)

    for i in range(len(w_list)-1,-1,-1):

        h_j= i-1

        if h_j == -1:

            w_list[i] -= learning_rate*(p.dot(X))+alpha*w_list[i]

        else:

            p_1 = (w_list[i].T.dot(p))*jacobian_dh_dz(z_list[h_j])

            w_list[i] -= learning_rate*(p.dot(h_list[h_j].T))+alpha*w_list[i]

        b_list[i] = (b_list[i].T- learning_rate*(np.sum(p,axis=1))).T

        p = p_1

    return w_list, b_list


def network(n,X,y,learning_rate,K,decayrate,epoch,batch,alpha):

    list_1 = [i for i in range(batch)]

    n_1 = int(X.shape[0]/batch)

    a = 0

    for _ in range(epoch):

        random.shuffle(list_1)

        for j in list_1:

            z_list=[]

            h_list=[]

            if a%K == 0 and a>=K:

                 learning_rate=decayrate*learning_rate

            e=(j*n_1)

            f= ((j+1)*n_1)

            X_1=X[e:f,:] 

            Y_1=y[e:f,:]

            if a == 0:

                w_list,b_list = w_list_b_list(n,X_1)

            z_list,h_list,y_hat = forward_propagation(X_1,w_list,b_list,z_list,h_list)

            w_list, b_list = back_propagation(Y_1,y_hat,z_list,h_list,w_list,b_list,X_1,learning_rate,alpha)

            a +=1

        acc = accuracy(y,X,w_list,b_list)[0]

        cost = cost_function(y,w_list,b_list,X)

        print(int(a/1000),'acc:',acc,'cost:',cost)

    return w_list, b_list

   
def cost_function(y,w_list,b_list,X):

     z_list=[]

     h_list=[]

     y = np.asmatrix(y)

     prediction = np.asmatrix(forward_propagation(X,w_list,b_list,z_list,h_list)[2]).T

     np.putmask(prediction, prediction<1E-323, 1E-323)

     n = y.shape[0]

     cost =((-1/(2*n))*np.sum(np.multiply(y,np.log(prediction))))

     return cost
     
def accuracy(y,X,w_list,b_list):

    z_list=[]

    h_list=[]

    prediction = np.asmatrix(forward_propagation(X,w_list,b_list,z_list,h_list)[2].T)

    prediction_tag = np.asmatrix((np.isin((prediction - np.amax(prediction,axis=1)),0)).astype(int))

    accuracy = 100-(np.sum(np.amax((y-prediction_tag),axis=1))/(y.shape[0]))*100

    return accuracy, prediction_tag

def main():

    trainimages = np.load('/Users/york/Desktop/MNIST/mnist_train_images.npy')

    trainlabels = np.load('/Users/york/Desktop/MNIST/mnist_train_labels.npy')

    validationimages = np.load('/Users/york/Desktop/MNIST/mnist_validation_images.npy')

    validationlabels = np.load('/Users/york/Desktop/MNIST/mnist_validation_labels.npy')

    testimages = np.load('/Users/york/Desktop/MNIST/mnist_test_images.npy')

    testlabels = np.load('/Users/york/Desktop/MNIST/mnist_test_labels.npy')

    X= trainimages

    y = trainlabels

    learning_rate= 0.06

    n=[40,40,40,10]

    K = 500

    decayrate=1

    epoch = 50

    batch= 1000

    alpha = 1e-100000

    w_list, b_list = network(n,X,y,learning_rate,K,decayrate,epoch,batch,alpha)

    acc,y_hat = accuracy(testlabels,testimages,w_list,b_list)

    print('accuracy on test dataset:',acc)

if __name__== "__main__":
    main()
