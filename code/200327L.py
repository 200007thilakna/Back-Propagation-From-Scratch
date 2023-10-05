#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


# ## Neural Network

# #### This NN contains 4 layers inclusing the input layer.
# 
#     * input layer : 14 nodes
#     * layer 1 : 100 nodes
#     * layer 2 : 40 nodes
#     * output layer : 4 nodes
# 
# 
#     * Activation Function : ReLU
#     * Cost function : cross entropy cost

# In[2]:


number_of_layers = 4 

layer_set = [14,100,40,4]


# ### Forward Propagation

# #### Forward Propagation helper functions
#     activation_orward : 
#     activation_ReLU : ReLU (forward)
#     softmax : for the output layer
#     

# In[3]:


def activation_ReLU(z):
    return np.maximum(0,z), z


# In[4]:


def linear_forward(A_prev, W, b):
    # Compute the linear output Z for the current layer
    Z = np.dot(A_prev, W) + b

    # Create a cache tuple to store values needed for backward propagation
    cache = (A_prev, W, b)

    # Each row of Z corresponds to an example, and each column corresponds to a node in the layer

    return Z, cache


# In[5]:


def activation_forward(A_prev, W, b, activation):
    # Perform linear forward propagation and get Z (linear output) and cache for this layer
    Z, linear_forward_cache = linear_forward(A_prev, W, b)

    # Initialize activation_cache
    activation_cache = None

    # Check if the layer should have activation (ReLU) or not
    if activation:
        # Apply ReLU activation to Z and get the activated output A
        A, activation_cache = activation_ReLU(Z)
    else:
        # If no activation is required, A is set to Z
        A = Z

    # Combine linear and activation caches into a single cache for this layer
    cache = (linear_forward_cache, activation_cache)

    return A, cache


# In[6]:


def forward_propagation(X, parameters, L):
    # Initialize a list to store intermediate caches
    caches = []

    # Initialize A as the input X
    A = X
    activation = True

    # Loop through layers (1 to L-1)
    for l in range(1, L):
        A_prev = A

        # Determine if this is the last layer (no activation)
        if l == L - 1:
            activation = False

        # Perform forward pass through the layer
        A, cache = activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation)

        # Append the cache to the caches list
        caches.append(cache)

    return A, caches


# In[7]:


def softmax(inputs):
    exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
    probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
    return probabilities   
    


# ### Backward Propagation

# In[8]:


# calculates the gradients during the backwardpass: linear layer

def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[0]
    
    # Compute the gradient of the loss with respect to the weights (dW)
    dW = np.dot(A_prev.T, dZ) / m
    
    # Compute the gradient of the loss with respect to the bias (db)
    db = np.sum(dZ, axis=0, keepdims=True) / m
    
    # Compute the gradient of the loss with respect to the previous layer's activations (dA_prev)
    dA_prev = np.dot(dZ, W.T)
    
    return dA_prev, dW, db



# In[9]:


# calculates the gradient of the ReLU function during the backward pass 

def relu_backward(dA, activation_cache):
    condition = activation_cache > 0
    activation_cache[condition] = 1
    condition = activation_cache <= 0
    activation_cache[condition] = 0
    dZ = np.multiply(dA, activation_cache)
    return dZ


# In[10]:


# calculates the gradients for an activation layer during the backward pass
def activation_backward(dA, cache):
    linear_cache, activation_cache = cache
    dZ = relu_backward(dA, activation_cache)
    dA_prev, dW, db = linear_backward(dZ, linear_cache)
    return dA_prev, dW, db


# In[11]:


# calculates the gradients for a softmax output layer during the backward pass

def softmax_backward(dAL, linear_cache):
    dA_prev, dW, db = linear_backward(dAL, linear_cache)
    return dA_prev, dW, db


# In[12]:


# calculates the gradients of all layers

def backward_propagation(AL, Y, caches):
    # Initialize a dictionary to store gradients
    grads = {}

    m = AL.shape[0]
    L = len(caches)

    # Compute softmax probabilities and initial gradient dAL
    softmax_probabilities = softmax(AL)
    dAL = softmax_probabilities - Y

    current_cache = caches[L - 1]
    linear_cache, activation_cache = current_cache
    dA_prev_temp, dW_temp, db_temp = linear_backward(dAL, linear_cache)
    grads['dA' + str(L - 1)] = dA_prev_temp
    grads['dW' + str(L)] = dW_temp
    grads['db' + str(L)] = db_temp

    for l in reversed(range(L - 1)):  # Loop from l = L-2 to 0
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = activation_backward(grads['dA' + str(l + 1)], current_cache)
        grads['dA' + str(l)] = dA_prev_temp
        grads['dW' + str(l + 1)] = dW_temp
        grads['db' + str(l + 1)] = db_temp

    return grads


# #### Gradient Descent Algorithm

# In[13]:


def gradient_descent(params, grads, alpha):

  parameters = params.copy()

  for l in range(1,len(layer_set)):

    parameters['W' + str(l)] = params['W' + str(l)] - alpha*grads['dW' + str(l)]
    parameters['b' + str(l)] = params['b' + str(l)] - alpha*grads['db' + str(l)]

  return parameters


# In[14]:


def cost_function(Y, softmax_probabilities):
  m = Y.shape[0] # num of samples
  cost = - np.sum(np.multiply(Y,np.log(softmax_probabilities)), keepdims=True)/m
  cost = np.squeeze(cost)
  return cost


# ## Task 01

# #### Task 01 helper functions
#     set_parameters: set the parameters given in the csv files as the initial parameter values in the NN

# In[15]:


def set_parameters(layer_set,W,b):
    parameters ={}
    start_row = 0
    n = len(layer_set)
    for l in range(1,n):
        parameters['W' +str(l)] = (w_task_1_a[start_row:layer_set[l-1]+start_row,:layer_set[l]])
        parameters['b' +str(l)] = b_task_1_a[l-1:l,:layer_set[l]]


        start_row += layer_set[l-1]
        
        # check if the shapes are matched
        assert(parameters['W' +str(l)].shape == (layer_set[l-1],layer_set[l]))
        assert(parameters['b' +str(l)].shape == (1,layer_set[l]))
    return parameters


# In[16]:


w_task_1_a = pd.read_csv('Task_1/a/w.csv',header=None)

b_task_1_a = pd.read_csv('Task_1/a/b.csv',header=None)

true_db_taks_1_a = pd.read_csv('Task_1/a/true-db.csv',header=None)

true_dw_task_1_a = pd.read_csv('Task_1/a/true-dw.csv',header=None)

w_task_1_b = pd.read_csv('Task_1/b/w-100-40-4.csv',header=None)

b_task_1_b = pd.read_csv('Task_1/b/b-100-40-4.csv',header=None)



# In[17]:


w_task_1_a = w_task_1_a.iloc[:,1:].to_numpy()
b_task_1_a = b_task_1_a .iloc[:,1:].to_numpy()
true_db_taks_1_a= true_db_taks_1_a.iloc[:,1:].to_numpy()
true_dw_task_1_a = true_dw_task_1_a.iloc[:,1:].to_numpy()

w_task_1_b = w_task_1_b.iloc[:,1:].to_numpy()

b_task_1_b = b_task_1_b.iloc[:,1:].to_numpy()



# In[18]:


Y =np.array( [[0,0,0,1]])
X = np.array([[-1, 1, 1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1]])


# #### Initialize weights and biases

# In[19]:


parameters = set_parameters(layer_set,w_task_1_a,b_task_1_a)


# #### Forward propagation

# In[20]:


output, store = forward_propagation(X, parameters, 4)


# #### Backpropagation

# In[21]:


gradients = backward_propagation(output, Y,store)


# #### Saving the answers

# In[22]:


np.savetxt('dw3.csv',(gradients['dW3'].astype(np.float32)), delimiter=",",fmt='%.16f')
np.savetxt('dw2.csv',gradients['dW2'].astype(np.float32), delimiter=",",fmt='%.16f')
np.savetxt('dw1.csv',gradients['dW1'].astype(np.float32), delimiter=",",fmt='%.16f')


# In[23]:


np.savetxt('db3.csv',gradients['db3'].astype(np.float32), delimiter=",",fmt='%.16f')
np.savetxt('db2.csv',gradients['db2'].astype(np.float32), delimiter=",",fmt='%.16f')
np.savetxt('db1.csv',gradients['db1'].astype(np.float32), delimiter=",",fmt='%.16f')


# In[24]:


import pandas as pd

# Load the three CSV files into dataframes
df1 = pd.read_csv('dw1.csv', header=None)
df2 = pd.read_csv('dw2.csv', header=None)
df3 = pd.read_csv('dw3.csv', header=None)

# Concatenate the dataframes row-wise
combined_df = pd.concat([df1, df2, df3], axis=0)

# Save the combined dataframe to a new CSV file
combined_df.to_csv('Task_1/b/true-dw.csv', index=False, header=False)


# In[25]:


import pandas as pd

# Load the three CSV files into dataframes
df1 = pd.read_csv('db1.csv', header=None)
df2 = pd.read_csv('db2.csv', header=None)
df3 = pd.read_csv('db3.csv', header=None)

# Concatenate the dataframes row-wise
combined_df = pd.concat([df1, df2, df3], axis=0)

# Save the combined dataframe to a new CSV file
combined_df.to_csv('Task_1/b/true-db.csv', index=False, header=False)


# ## Task 02

# In[26]:


def initialize_parameters(layer_set):

  np.random.seed(3)
  parameters = {}
  n = len(layer_set)

  for l in range(1,n):
    parameters['W' + str(l)] = np.random.randn(layer_set[l-1],layer_set[l])*0.01
    parameters['b' + str(l)] = np.zeros((1,layer_set[l]))

    assert(parameters['W' + str(l)].shape == (layer_set[l-1],layer_set[l]) )
    assert(parameters['b' + str(l)].shape == (1,layer_set[l]) )

  return parameters


# In[27]:


import matplotlib.pyplot as plt

def train_model(X_train, Y_train, learning_rate, Y_true, num_iterations= 10000, print_cost=False):
    np.random.seed(3)

    # Initialize parameters for the neural network
    parameters = initialize_parameters(layer_set)

    # Lists to store the cost, accuracy, and parameters over iterations
    cost_history = []
    accuracy_history = []
    parameters_history = []

    for iteration in range(num_iterations):
        # Forward propagation to compute predictions and cost
        AL, caches = forward_propagation(X_train, parameters, 4)
        softmax_probabilities = softmax(AL)
        cost = cost_function(Y_train, softmax_probabilities)

        # Backward propagation to compute gradients
        grads = backward_propagation(AL, Y_train, caches)

        # Update parameters using gradient descent
        parameters = gradient_descent(parameters, grads, learning_rate)

        if iteration % 100 == 0:
            # Calculate and store accuracy
            predictions = predict_labels(X_train, parameters)
            accuracyscore = accuracy_score(Y_true, predictions)
            accuracy_history.append(accuracyscore)

            # Store cost and parameters
            cost_history.append(cost)
            parameters_history.append(parameters)

        if print_cost and iteration % 1000 == 0:
            print('Cost after iteration %i: %f' % (iteration, cost))
                # Create graphs for cost and accuracy
    plt.figure(figsize=(12, 5))

    # Plot cost history
    plt.subplot(1, 2, 1)
    plt.plot(cost_history)
    plt.title('Cost Over Iterations')
    plt.xlabel('Iterations')
    plt.ylabel('Cost')

    # Plot accuracy history
    plt.subplot(1, 2, 2)
    plt.plot(accuracy_history)
    plt.title('Accuracy Over Iterations')
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    
    plt.show()
    
    return parameters, cost_history, accuracy_history, parameters_history


# In[28]:


def predict_labels(X, parameters):
    # Forward propagation to compute predicted probabilities
    AL, _ = forward_propagation(X, parameters, 4)
    predicted_probabilities = softmax(AL)

    # Find the index (label) with the highest probability for each example
    predicted_labels = np.argmax(predicted_probabilities, axis=1)

    return predicted_labels


# In[29]:


import matplotlib.pyplot as plt

def predict_and_plot(X, parameters_list, Y, Y_true):
    cost_list = []
    accuracy_list = []

    for parameters in parameters_list:
        AL, caches = forward_propagation(X, parameters, 4)
        predicted_probabilities = softmax(AL)
        cost = cost_function(Y, predicted_probabilities)
        prediction = np.argmax(predicted_probabilities, axis=1)
        accuracyscore = accuracy_score(Y_true, prediction)
        cost_list.append(cost)
        accuracy_list.append(accuracyscore)

    # Plot cost and accuracy over iterations
    plt.figure(figsize=(12, 5))

    # Plot cost
    plt.subplot(1, 2, 1)
    plt.plot(cost_list)
    plt.title('Cost Over Iterations')
    plt.xlabel('Iterations')
    plt.ylabel('Cost')

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(accuracy_list)
    plt.title('Accuracy Over Iterations')
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')

    plt.show()

    return cost_list, accuracy_list


# In[30]:


x_train = pd.read_csv('Task_2/x_train.csv',header=None)
x_test = pd.read_csv('Task_2/x_test.csv',header=None)
y_train = pd.read_csv('Task_2/y_train.csv',header=None)
y_test = pd.read_csv('Task_2/y_test.csv',header=None)


# In[31]:


y_train_true =np.squeeze( y_train.iloc[:,:].to_numpy())
y_test_true = np.squeeze(y_test.iloc[:,:].to_numpy())


# In[32]:


num_classes = 4

# Convert y_train_true to one-hot encoding
y_train = np.eye(num_classes)[y_train_true]

# Convert y_test_true to one-hot encoding
y_test = np.eye(num_classes)[y_test_true]


# ### training_1
#     learning_rate = 1

# In[ ]:


parameters_1 ,cost,accuracy, parameters_list= train_model(x_train, y_train,1,y_train_true,print_cost=True,)


# ### training_2
#     learning_rate = 0.1

# In[ ]:


# training_2
# learning_rate = 0.1
parameters_2,cost_2,accuracy_2,parameters_list_2 = train_model(x_train, y_train,0.1,y_train_true,print_cost=True)


# ### training_3
#     learning_rate = 0.001

# In[ ]:


parameters_3,cost_3,accuracy_3,parameters_list_3 = train_model(x_train, y_train,0.001,y_train_true,print_cost=True)


# #### Test: alpha = 1 
# 

# In[ ]:


test_cost_1, test_accuracy_1 = predict_and_plot(x_test,parameters_list,y_test, y_test_true)


# #### Test : alpha = 0.1

# In[ ]:


test_cost_2, test_accuracy_2 = predict_and_plot(x_test,parameters_list_2,y_test, y_test_true)


# #### Test alpha = 0.001

# In[ ]:


test_cost_3, test_accuracy_3 = predict(x_test,parameters_list_3,y_test, y_test_true)

