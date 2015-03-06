from __future__ import division
import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt



def initializeWeights(n_in,n_out):
	"""
	# initializeWeights return the random weights for Neural Network given the
	# number of node in the input layer and output layer
	# Input:
	# n_in: number of nodes of the input layer
	# n_out: number of nodes of the output layer
	   
	# Output: 
	# W: matrix of random initial weights with size (n_out x (n_in + 1))"""

	epsilon = sqrt(6) / sqrt(n_in + n_out + 1);
	W = (np.random.rand(n_out, n_in + 1)*2* epsilon) - epsilon;
	return W
    
    
    
def sigmoid(z):

	# Notice that z can be a scalar, a vector or a matrix
	# return the sigmoid of input z

	return  1.0 / (1.0 + np.exp(-1.0 * z))#your code here -- taken from gradience?
    
    

def preprocess():
	""" Input:
	Although this function doesn't have any input, you are required to load
	the MNIST data set from file 'mnist_all.mat'.
	Output:
	train_data: matrix of training set. Each row of train_data contains 
	feature vector of a image
	train_label: vector of label corresponding to each image in the training
	set
	validation_data: matrix of training set. Each row of validation_data 
	contains feature vector of a image
	validation_label: vector of label corresponding to each image in the 
	training set
	test_data: matrix of training set. Each row of test_data contains 
	feature vector of a image
	test_label: vector of label corresponding to each image in the testing
	set
	Some suggestions for preprocessing step:
	- divide the original data set to training, validation and testing set
	with corresponding labels
	- convert original data set from integer to double by using double()
	function
	- normalize the data to [0, 1]
	- feature selection"""
	mat = loadmat('mnist_all.mat') #loads the MAT object as a Dictionary
	#Pick a reasonable size for validation data

	training = np.vstack((mat['train0'],mat['train1'],mat['train2'],mat['train3'],mat['train4'],mat['train5'],mat['train6'],mat['train7'],mat['train8'],mat['train9']))
	testing = np.vstack((mat['test0'],mat['test1'],mat['test2'],mat['test3'],mat['test4'],mat['test5'],mat['test6'],mat['test7'],mat['test8'],mat['test9']))



	train_vector=[]		#These will be for the labels
	for i in range(0,10):
		temp='train'+str(i)
		for j in range(0,np.shape(mat[temp])[0]):
			train_vector.append(labelmaker(i))

	test_vector=[]
	for i in range(0,10):
		temp='test'+str(i)
		for j in range(0,np.shape(mat[temp])[0]):
			test_vector.append(labelmaker(i))

	#normalizing and repalces the matrices as doubles using the normalizer definition
	testing=normalizer(testing)	
	training=normalizer(training)

	rand=np.random.permutation(60000)			#make a list of random numbers
	randmatrix=list(training)				#makes new matrix for randomized data
	randlabels=list(train_vector)				#makes new list of labels corresponding to randomness
	for i in range(0, np.shape(rand)[0]):
		randmatrix[rand[i]]=training[i]			#copy the values from our training matrix over to our new random matrix
		randlabels[rand[i]]=train_vector[i]		#copy our labels to corresponding random values to preserve order

	train_data = np.array(randmatrix[0:49999])
	train_label = np.array(randlabels[0:49999])
	validation_data = np.array(randmatrix[50000:59999])
	validation_label = np.array(randlabels[50000:59999])
	test_data = np.array(testing[0:9999])
	test_label = np.array(test_vector[0:9999])
	print('preprocessing is complete')
	return train_data, train_label, validation_data, validation_label, test_data, test_label

def normalizer (matrice):
	matrice=matrice.astype(np.double, copy=False)  #converts to double
	for x in range(0,np.shape(matrice)[0]):
		matrice[x]=np.double(matrice[x])/np.double(255.0) #normalizes by dividing by 255
	return matrice

def labelmaker (x):
	ret=[]
	for i in range (0,10):
		if i==x:
			ret.append(1)
		else:
			ret.append(0)
	return ret
	

def nnObjFunction(params, *args):
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
	%     layer to unit i in output layer."""

	n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

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
            
	#hidden_value = np.dot(training_data, w1)
				
	#add bias node
	for i in range (n_hidden):
		for j in range (np.shape(training_data)[0]):
			hidden_value[j][i] += 1 * w1[i][n_input]

	for j in range (np.shape(training_data)[0]):
			hidden_value[j][n_hidden] = 1

	for i in range(n_hidden):
		for j in range (np.shape(training_data)[0]):
			hidden_value[j][i] = sigmoid(hidden_value[j][i])



	#output_value = np.zeros((np.shape(training_data)[0],n_class))
	#for k in range(np.shape(training_data)[0]):
	#   for i in range(n_class):
	#       for j in range(n_hidden+1):
	#           output_value[k][i] += w2[i][j]*hidden_value[k][j]
	
	#output_value = np.ones((np.shape(training_data)[0], n_class))
	output_value = np.dot(hidden_value, w2.T)
	#output_value = np.dot(temp,output_value)  
 
	for i in range(n_class):
		for k in range(np.shape(training_data)[0]):
			output_value[k][i] = sigmoid(output_value[k][i])

        # Now we do Error Function to find value of J
	J = np.zeros(np.shape(training_data)[0])
	J_average = 0

	
	for i in range(np.shape(training_data)[0]):
		for l in range(n_class):
			J[i] += (-1.0)*(training_label[i][l]*np.log(output_value[i][l])+(1-training_label[i][l])*np.log(1-output_value[i][l]))
	for i in range(np.shape(training_data)[0]):
		J_average += J[i]
	J_average = J_average/np.shape(training_data)[0]
	print (J_average)
	# Now compute the derivative of err function from hidden unit to output
	#computing gradient for each weight and push them into hidden_err

        ########################################################################


	output_err = np.zeros((np.shape(training_data)[0],n_class,n_hidden+1))


	for k in range(np.shape(training_label)[0]):
		for l in range(n_class):
			for j in range(n_hidden+1):
				output_err[k][l][j] = (output_value[k][l] - training_label[k][l])*hidden_value[k][j]

	# Now compute the derivative of err function from input unit to output
	hidden_err = np.zeros((np.shape(training_data)[0],n_hidden,n_input+1))
	#sigma_output = np.zeros((np.shape(training_label)[0],n_hidden))
	#for k in range(np.shape(training_label)[0]):
	#    for l in range(0,n_class):
	#        for j in range(0,n_hidden):
	#            sigma_output[k][j] += w2[l][j] * (output_value[k][l] - training_label[k][l])
	train_temp = training_label[:,:n_class]
	sigma_output = np.dot((output_value-train_temp),w2)
 	#sigma_output = np.dot(w2,(output_value-training_label))
	for k in range(np.shape(training_data)[0]):
		for j in range(n_hidden):
			for i in range(n_input):
				hidden_err[k][j][i] = (1 - hidden_value[k][j]) * hidden_value[k][j] * sigma_output[k][j] * training_data[k][i]

   	#for i in range(np.shape(training_data)[0]):
   	#   	for j in range(n_hidden):
    	#  	   	temp1 = np.dot((1-hidden_value[j]),hidden_value[j].T)
  	#	   	temp2 = np.dot(temp1, sigma_output[j])
        #   	   	hidden_err[i] = np.dot(temp2, training_data[j])

	for k in range(np.shape(training_data)[0]):
		for j in range(n_hidden):
			hidden_err[k][j][n_input] = (1 - hidden_value[k][j]) * hidden_value[k][j] * sigma_output[k][j] * 1
    
	grad_output = np.zeros((n_class,n_hidden+1))
    
   	#for k in range(np.shape(training_label)[0]):
   	#   	for l in range(n_class):
   	#   	   	for j in range(n_hidden+1):
   	#   	   	   	grad_output[l][j] += output_err[k][l][j] / np.shape(training_label)[0]

	for i in range(np.shape(training_label)[0]):
		grad_output += np.dot(output_err[i], 1/np.shape(training_label)[0])
    
   	
   	#grad_output = output_err * (1/np.shape(training_label)[0]) 	
   	      
	grad_hidden = np.zeros((n_hidden,n_input+1))
	#for k in range(np.shape(training_data)[0]):
   	#   	for j in range(n_hidden):
   	#   	   	for i in range(n_input+1):
   	#   	   	   	grad_hidden[j][i] += hidden_err[k][j][i] / np.shape(training_data)[0]
   	
	for i in range(np.shape(training_data)[0]):
		grad_hidden += np.dot(hidden_err[i], 1/np.shape(training_data)[0])
	obj_val = J_average
        

	#Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
	#you would use code similar to the one below to create a flat array
	#obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
	obj_grad = np.concatenate((grad_hidden.flatten(),grad_output.flatten()),0)

	return (obj_val,obj_grad)





def nnPredict(w1,w2,data):

	"""% nnPredict predicts the label of data given the parameter w1, w2 of Neural
	% Network.
	% Input:
	% w1: matrix of weights of connections from input layer to hidden layers.
	%     w1(i, j) represents the weight of connection from unit i in input 
	%     layer to unit j in hidden layer.
	% w2: matrix of weights of connections from hidden layer to output layers.
	%     w2(i, j) represents the weight of connection from unit i in input 
	%     layer to unit j in hidden layer.
	% data: matrix of data. Each row of this matrix represents the feature 
	%       vector of a particular image
	% Output: 
	% label: a column vector of predicted labels""" 

	labels = np.array([[]]).reshape(0, 10)
	data_ones_column = np.copy(data[:,-1])
	for i in range(0, data_ones_column.size):
		data_ones_column[i] = 1
	data_ones_column_reshape = data_ones_column.reshape(-1, 1)
	data_stack_1 = np.hstack((data, data_ones_column_reshape))
	data_sigmoid_w1 = sigmoid(np.dot(data_stack_1, w1.T)) 
	data_stack_2 = np.hstack((data_sigmoid_w1, data_ones_column_reshape))
	data_sigmoid_w2 = sigmoid(np.dot(data_stack_2, w2.T))
	data_max = np.argmax(data_sigmoid_w2, 1);
	for i in range(0, data_max.size):
		temp = [0,0,0,0,0,0,0,0,0,0]
		temp[data_max[i]] = 1;
		labels = np.vstack((labels, temp))
	# return labels
	return labels



"""**************Neural Network Script Starts here********************************"""

train_data, train_label, validation_data,validation_label, test_data, test_label = preprocess();


#  Train Neural Network

# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]; 

# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 20;

# set the number of nodes in output unit
n_class = 10;				   

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden);
initial_w2 = initializeWeights(n_hidden, n_class);

# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()),0)

# set the regularization hyper-parameter
lambdaval = 0.3;


args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

#Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

opts = {'maxiter' : 50}    # Preferred value.

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args,method='CG', options=opts)

#In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
#and nnObjGradient. Check documentation for this function before you proceed.
#nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


#Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))


#Test the computed parameters

predicted_label = nnPredict(w1,w2,train_data)

#find the accuracy on Training Dataset

print('\n Training set Accuracy:' + str(100*np.mean((predicted_label == train_label).astype(float))) + '%')

predicted_label = nnPredict(w1,w2,validation_data)

#find the accuracy on Validation Dataset

print('\n Validation set Accuracy:' + str(100*np.mean((predicted_label == validation_label).astype(float))) + '%')


predicted_label = nnPredict(w1,w2,test_data)

#find the accuracy on Validation Dataset

print('\n Test set Accuracy:' + str(100*np.mean((predicted_label == test_label).astype(float))) + '%')
