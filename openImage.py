# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 20:48:09 2017

@author: Paul
https://discussions.udacity.com/t/new-image-prediction-error-invalidargumenterror-see-above-for-traceback-you-must-feed-a-value-for-placeholder-tensor-placeholder-24-with-dtype-float/244007/3?u=subodh.malgonde

This for "Step 3 Test a Model on new images"
"""

import matplotlib.image as img
import matplotlib.pyplot as plt
import numpy as np
import pickle
import random


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




new_images = np.ndarray(shape=(5,32,32,3))
file1 = 'new_images\\1.jpg'
file2 = 'new_images\\2.jpg'
file3 = 'new_images\\3.jpg'
file4 = 'new_images\\4.jpg'
file5 = 'new_images\\5.jpg'

#new labels and images 
new_labels = ([21,27,28,14,17])
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
    title = str(new_labels[i])
    a.set_title(title)
    image= new_images[i].squeeze()
    image = image.astype(np.uint8)
    plt.imshow(image)
    
 
#normalize the new images  
print("begin normalization of new images")
normalizeArray(new_images)
print("normalization of new images complete")


###
#ugliness 
###
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

####
## Finding a test set ###
###
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

#create an array from the training set
training_images = np.ndarray(shape=(10,32,32,3))
training_labels = [0] * 10
for j in range (0,10):
    index = random.randint(0, len(X_train))
    training_images[j] = X_train[index]
    training_labels[j] = y_train[index]
    
#display the training images
displayImages(training_images,training_labels,len(training_images))
    
#normalize the new images  
print("begin normalization of new images")
normalizeArray(new_images)
print("normalization of new images complete") 

#display the normalized training images   
print("normalizing training images")
normalizeArray(training_images) 
print("displaying normalized training images")
displayImages(training_images,training_labels,len(training_images))