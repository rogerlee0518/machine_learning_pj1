from __future__ import division
import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt


def sigmoid(z):

        # Notice that z can be a scalar, a vector or a matrix
        # return the sigmoid of input z

	return  1.0 / (1.0 + np.exp(-1.0 * z))#your code here -- taken from gradience?


def nnObjFunction():
	"""% nnObjFunction computes the value of objective function (negative log 
	%   likelihood error function with regularization) given the parameters 
	%   of Neural Networks, thetraining data, their corresponding training 
	%   labels and lambda - regularization hyper-parameter.

	% Input:
	% params: vector of weights of 2 matrices w1 (weights of connections from
	%     input layer to hidden layer) and w2 (weights of connections from
	%     hidden layer to output layer) where all of the weights are contained
	%     in a single vector.
	% n_input: number of node in input layer (not include the bias node)
	% n_hidden: number of node in hidden layer (not include the bias node)
	% n_class: number of node in output layer (number of classes in
	%     classification problem
	% training_data: matrix of training data. Each row of this matrix
	%     represents the feature vector of a particular image
	% training_label: the vector of truth label of training images. Each entry
	%     in the vector represents the truth label of its corresponding image.
	% lambda: regularization hyper-parameter. This value is used for fixing the
	%     overfitting problem.

	% Output: 
	% obj_val: a scalar value representing value of error function
	% obj_grad: a SINGLE vector of gradient value of error function
	% NOTE: how to compute obj_grad
	% Use backpropagation algorithm to compute the gradient of error function
	% for each weights in weight matrices.

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% reshape 'params' vector into 2 matrices of weight w1 and w2
	% w1: matrix of weights of connections from input layer to hidden layers.
	%     w1(i, j) represents the weight of connection from unit j in input 
	%     layer to unit i in hidden layer.
	% w2: matrix of weights of connections from hidden layer to output layers.
	%     w2(i, j) represents the weight of connection from unit j in hidden 
     layer to unit i in output layer."""
	n_input = 5 
	n_hidden = 3
	n_class = 2
	training_data = np.array([np.linspace(0,1,num=5),np.linspace(1,0,num=5)])
	training_label = np.array(([1,0,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0,0]))
	lambdaval = 0
	params = np.linspace(-5,5, num=26)
	#n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

	w1 = params[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
	w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
	obj_val = 0  

	#Your code here

	#Feedforward Propagation
        
	hidden_value = np.zeros((np.shape(training_data)[0],n_hidden+1))
	
	for j in range(np.shape(training_data)[0]):
	   for i in range(n_hidden):
	       for k in range(n_input):
	           hidden_value[j][i] += w1[i][k]*training_data[j][k]
				
	#add bias node
	for i in range (n_hidden+1):
	    for j in range (np.shape(training_data)[0]):
		hidden_value[j][i] += 1

	for i in range(n_hidden):
	   for j in range (np.shape(training_data)[0]):
		hidden_value[j][i] = sigmoid(hidden_value[j][i])



	output_value = np.zeros((np.shape(training_data)[0],n_class))

        for k in range(np.shape(training_data)[0]):
	   for i in range(n_class):
	       for j in range(n_hidden+1):
	           output_value[k][i] += w2[i][j]*hidden_value[k][j]

	for i in range(n_class):
	    for k in range(np.shape(training_data)[0]):
		output_value[k][i] = sigmoid(output_value[k][i])

        # Now we do Error Function to find value of J
	J = np.zeros(np.shape(training_data)[0])
	J_average = 0
	print(np.shape(training_label))
        print(np.shape(output_value))

	
	for i in range(np.shape(training_data)[0]):
	    for l in range(n_class):
	        J[i] += (-1.0)*(training_label[i][l]*np.log(output_value[i][l])+(1-training_label[i][l])*np.log(1-output_value[i][l]))
	for i in range(np.shape(training_data)[0]):
	    J_average += J[i]
        J_average = J_average/np.shape(training_data)[0]
        print J_average
	# Now compute the derivative of err function from hidden unit to output
	#computing gradient for each weight and push them into hidden_err

	output_err = np.zeros((n_class,n_hidden))

	# WE SET training_label[0][1] for simplicity. Maybe because input doesn't affect our weight??
	################################NEED TO CHECK####################################
	for l in range(0,n_class):
		for j in range(0,n_hidden):
			output_err[l][j] = (output_value[l] - training_label[0][l])*hidden_value[j]

	# Now compute the derivative of err function from input unit to output
	hidden_err = np.zeros((n_hidden,n_input))
	sigma_output = np.zeros(n_hidden)
	for l in range(0,n_class):
		for j in range(0,n_hidden):
			sigma_output[j] += w2(l,j) * (output_value[l] - training_label[0][l])

	input_value = np.zeros(n_input)
	for i in range(n_input):
		for j in range(0,np.shape(training_data)[1]):
			input_value[i] += training_data[i][j]

	for j in range(0,n_hidden):
		for i in range(0,n_input):
			hidden_err[j][i] = (1 - hidden_value[j]) * hidden_value[j] * sigma_output[j] * input_value[i]
	           
	grad_temp = 0

	for l in range(0,n_class):
		for j in range(0,n_hidden):
			grad_temp += output_err[l][j]
                
	for j in range(0,n_hidden):
		for i in range(0,n_input):
			grad_temp += hidden_err[j][i]
        
	obj_val = (grad_temp) / ((n_class * n_hidden) + (n_hidden * n_input))


	#Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
	#you would use code similar to the one below to create a flat array
	#obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
	obj_grad = np.concatenate((output_err.flatten(),hidden_err.flatten()),0)


	return (obj_val,obj_grad)
nnObjFunction()
