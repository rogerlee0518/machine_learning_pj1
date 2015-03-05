from scipy.io import loadmat
import numpy as np


mat = loadmat('/Users/macuser/Documents/machine_learning_pj1/basecode/mnist_all.mat')


train0 = mat.get('train0')    
train1 = mat.get('train1')
train2 = mat.get('train2')
train3 = mat.get('train3')
train4 = mat.get('train4')
train5 = mat.get('train5')
train6 = mat.get('train6')
train7 = mat.get('train7')
train8 = mat.get('train8')
train9 = mat.get('train9')
train_all = np.vstack((train0,train1,train2,train3,train4,train5,train6,train7,train8,train9))
print train_all[0]

for x in train_all:
        x = x/255.0
        print x
r_num = np.random.permutation(59999)
train_vali_data = np.zeros(784)
for x in r_num:
    train_vali_data = np.vstack((train_vali_data,train_all[x]))
train_data = train_vali_data[0:49999]
vali_data = train_vali_data[50000:]

train_label_temp = np.zeros(784)
'''
for i in range(train0.shape[0]):
    train_label_temp = np.vstack((train_label_temp,0))
    print 0
for i in range(train1.shape[0]):
    train_label_temp = np.vstack((train_label_temp,1))
    print 1
for i in range(train2.shape[0]):
    train_label_temp = np.vstack((train_label_temp,2))
    print 2
for i in range(train3.shape[0]):
    train_label_temp = np.vstack((train_label_temp,3))
for i in range(train4.shape[0]):
    train_label_temp = np.vstack((train_label_temp,4))
for i in range(train5.shape[0]):
    train_label_temp = np.vstack((train_label_temp,5))
for i in range(train6.shape[0]):
    train_label_temp = np.vstack((train_label_temp,6))
for i in range(train7.shape[0]):
    train_label_temp = np.vstack((train_label_temp,7))
for i in range(train8.shape[0]):
    train_label_temp = np.vstack((train_label_temp,8))
for i in range(train9.shape[0]):
    train_label_temp = np.vstack((train_label_temp,9))
    
train_label = np.zeros(784)
for x in r_num:
    train_label = np.vstack(train_label_temp[x])
    print train_label_temp[x]
'''

r_num = np.random.permutation(599)

train_vali_data = np.zeros(784)
for x in r_num:
    train_vali_data = np.vstack((train_vali_data,train_all[x]))
    '''print train_all[x]'''
train_data = train_vali_data[0:499]
vali_data = train_vali_data[500:]
for x in train_all:
        x = x/255.0
for x in vali_data:
    '''print x'''

'''
r_num = np.random.permutation(50000)
train_data = np.array([])
for x in r_num:
    train_data = np.append(train_data,train_all[x])
print (train_data)'''
'''
rand_numlist = np.random.permutation(600)

print(rand_numlist)
'''
