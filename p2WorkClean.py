# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 19:37:38 2017

@author: Paul
"""

'''
'features' is a 4D array containing raw pixel data of the traffic sign images, 
(num examples, width, height, channels).

'labels' is a 1D array containing the label/class id of the traffic sign. 
The file signnames.csv contains id -> name mappings for each id.

'sizes' is a list containing tuples, (width, height) representing the 
original width and height the image.

'coords' is a list containing tuples, (x1, y1, x2, y2) representing coordinates 
of a bounding box around the sign in the image. THESE COORDINATES ASSUME 
THE ORIGINAL IMAGE. THE PICKLED DATA CONTAINS RESIZED VERSIONS (32 by 32)
 OF THESE IMAGES

'''

import pickle
import numpy as np


def normalize(x):
    x = (x -128)/128
    return x

#normalize array
# assume array of float passed in
def normalizeArray(x):
    count = 0
    for i in range(len(x)):
        for j in range(len(x[0])):
            for k in range(len(x[0][0])):
                for l in range(len(x[0][0][0])):
                    x[i][j][k][l]=normalize(x[i][j][k][l]) 
                    count+=1 
    print("normalized count:", count)
    return x       

# Load pickled data
# TODO: Fill this in based on where you saved the training and testing data
# phc 7/19/2017
training_file = 'train.p'
validation_file='valid.p'
testing_file = 'test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']
print("data loaded\nlen of X_train is:",len(X_train))
print("len of y_train is:",len(y_train))
print("len of X_valid is:",len(X_valid))
print("len of y_valid is:",len(y_valid))
print("len of X_test is:",len(X_test))
print("len of y_test is:",len(y_test))

print("shape of X_train:", np.shape(X_train))
print("shape of y_train:", np.shape(y_train))
print("shape of X_test:", np.shape(X_test))
print("shape of y_test:", np.shape(y_test))
print("shape of X_valid:", np.shape(X_valid))
print("shape of y_valid:", np.shape(y_valid))

#try the shapes 
print("shape of an X_train image:", np.shape(X_train[0]))

#plot
import random
import matplotlib.pyplot as plt
#%matplotlib inline

index = random.randint(0, len(X_train))
image = X_train[index].squeeze()

plt.figure(figsize=(1,1))
#plt.imshow(image, cmap="gray")
plt.imshow(image)
print("index is", index,"y_train[index] is:",y_train[index])

#n Preprocess the data
from sklearn.utils import shuffle
X_train, y_train = shuffle(X_train, y_train)

#convert to float
#valid set
X_valid = X_valid.astype(np.float)
print("X_valid[0][0][0][0] to float, before norm" , X_valid[0][0][0][0])
print("normalizing X_valid....")
X_valid = normalizeArray(X_valid)
               
#training set 
X_train = X_train.astype(np.float)
print("X_train[0][0][0][0] to float, before norm" , X_train[0][0][0][0])
print("normalizing X_train ....")
X_train = normalizeArray(X_train)
        
 #test set 
X_test = X_test.astype(np.float)
print("X_test[0][0][0][0] to float, before norm" , X_test[0][0][0][0])
print("normalizing X_test ....")
X_test = normalizeArray(X_test)
                       
#Display an image after normalization            
print("Display image after normalization")
image = X_train[index].squeeze()
plt.imshow(image)
print("index is", index,"y_train[index] is:",y_train[index])

##########################
### MODEL ARCHITECTURE ###
##########################
# start with LeNet 5
import tensorflow as tf

EPOCHS = 30
BATCH_SIZE = 128
from tensorflow.contrib.layers import flatten

def LeNet(x):    
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1
    
    # SOLUTION: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    # phc change from 5,5,1,6 to 5,5,3,6 
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 3, 6), mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

    # SOLUTION: Activation.
    conv1 = tf.nn.relu(conv1)

    # SOLUTION: Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # SOLUTION: Layer 2: Convolutional. Output = 10x10x16.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
    
    # SOLUTION: Activation.
    conv2 = tf.nn.relu(conv2)

    # SOLUTION: Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # SOLUTION: Flatten. Input = 5x5x16. Output = 400.
    fc0   = flatten(conv2)
    
    # SOLUTION: Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(400, 120), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1   = tf.matmul(fc0, fc1_W) + fc1_b
    
    # SOLUTION: Activation.
    fc1    = tf.nn.relu(fc1)

    # SOLUTION: Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_W  = tf.Variable(tf.truncated_normal(shape=(120, 84), mean = mu, stddev = sigma))
    fc2_b  = tf.Variable(tf.zeros(84))
    fc2    = tf.matmul(fc1, fc2_W) + fc2_b
    
    # SOLUTION: Activation.
    fc2    = tf.nn.relu(fc2)

    # SOLUTION: Layer 5: Fully Connected. Input = 84. Output = 10.
    # phc 7/23/17 
    # changed output from 10 to 43
    fc3_W  = tf.Variable(tf.truncated_normal(shape=(84, 43), mean = mu, stddev = sigma))
    fc3_b  = tf.Variable(tf.zeros(43))
    logits = tf.matmul(fc2, fc3_W) + fc3_b
    
    return logits


###########################
### Features and Labels ###
########################### 
# x is the placeholder for the inputs
# y is the placeholder for the labels 
# x = tf.placeholder(tf.float32, (None, 32, 32, 1))
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None))
#phc change labels to 43
one_hot_y = tf.one_hot(y, 43)
print("Did Features and Labels")

#########################
### Training Pipeline ###
#########################
rate = 0.001
logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)
print("Did Training Pipleline")

########################
### Model Evaluation ###
########################
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples
print("Did Model Evaluation")

#######################
### Train the Model ###
#######################
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    
    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
            
        validation_accuracy = evaluate(X_valid, y_valid)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
        
    saver.save(sess, './lenet')
    print("Model saved")

print("Did training")

########################
### Step 3 Test on   ###
### New Images       ###
########################
import matplotlib.image as img
new_images = np.ndarray(shape=(5,32,32,3))
file1 = 'new_images\\10.jpg'
file2 = 'new_images\\14.jpg'
file3 = 'new_images\\28.jpg'
file4 = 'new_images\\40.jpg'
file5 = 'new_images\\8.jpg'

#new lables and images 
new_labels = ([10,14,28,40,8])
new_images[0]= img.imread(file1) 
new_images[1]= img.imread(file2) 
new_images[2]= img.imread(file3) 
new_images[3]= img.imread(file4) 
new_images[4]= img.imread(file5) 

   
print("Shape of new_images is:",np.shape(new_images)) 
print("length of new_images is:", len(new_images))


# display the images in a row
# they don't display properly as float
# must convert to uint8
# https://stackoverflow.com/questions/3584805/in-matplotlib-what-does-the-argument-mean-in-fig-add-subplot111

plt.figure(figsize=(1,1))
fig = plt.figure()
for i in range(len(new_images)):
    a=fig.add_subplot(1,5,i+1)
    title = "image "
    title +=str(i)
    a.set_title(title)
    image= new_images[i].squeeze()
    image = image.astype(np.uint8)
    plt.imshow(image)
    
 
#normalize the new images  
print("begin normalization of new images")
normalizeArray(new_images)
print("normalization of new images complete")


################
### 7/29/2017
################
def displayImages(x,y,length):
    plt.figure(figsize=(1,1))
    fig = plt.figure()
    for i in range(len(x)):
        a=fig.add_subplot(1,length,i+1)
        title = str(y[i])
        a.set_title(title)
        image= x[i].squeeze()
        image = image.astype(np.uint8)
        plt.imshow(image)

#create an array from the training set
training_images = np.ndarray(shape=(10,32,32,3))
training_labels = [0] * 10
for j in range (0,10):
    index = random.randint(0, len(X_train))
    training_images[j] = X_train[index]
    training_labels[j] = y_train[index]
    

#display the normalized training images
#they are already normalized at this point   
print("displaying normalized training images")
displayImages(training_images,training_labels,len(training_images))

#create an array for the test set 
test_images = np.ndarray(shape=(10,32,32,3))
test_labels = [0] * 10
for j in range (0,10):
    index = random.randint(0, len(X_test))
    test_images[j] = X_test[index]
    test_labels[j] = y_test[index]
  
    
 #display the test images 
print("displaying normalized test images")
displayImages(test_images,test_labels,len(test_images))
    

#Restore the model and evaluate accuracy on new images
softmax = tf.nn.softmax(logits)
with tf.Session() as sess: 
    new_saver = tf.train.import_meta_graph('.\\lenet.meta')
    new_saver.restore(sess, tf.train.latest_checkpoint('.'))
    #saver.restore(sess, tf.train.latest_checkpoint('.'))
    new_image_accuracy = evaluate(new_images, new_labels)
    print("Test Accuracy new images = {:.3f}".format(new_image_accuracy))
    new_image_accuracy = evaluate(training_images, training_labels)
    print("Test Accuracy training images = {:.3f}".format(new_image_accuracy))
    new_image_accuracy = evaluate(test_images, test_labels)
    print("Test Accuracy test images = {:.3f}".format(new_image_accuracy))
    ### added
    result = sess.run(softmax, feed_dict={x: new_images})
    values, indices = tf.nn.top_k(result, 5)
    predictions  = sess.run(values)
    predictionIndices  = sess.run(indices)
    print("predictions")
    print(predictions)
    print("predictionIndices")
    print(predictionIndices)