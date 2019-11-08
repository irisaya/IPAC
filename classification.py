# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 16:30:22 2018

@author: Iris Zhou
"""
# Import libraries
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES']='0'
from tensorflow.python.client import device_lib
gpus = [x.name for x in device_lib.list_local_devices() \
        if x.device_type == 'GPU']

import cv2
import numpy as np
import time
import glob
import pandas
import tensorflow as tf
import collections
from copy import deepcopy
# for Figures
import matplotlib.pyplot as plt

from PIL import Image
from sklearn.manifold import TSNE
from scipy.misc import imsave
import matplotlib.cm as cm
# Keras
import keras
import keras.backend as K
from keras import activations
from keras.utils import multi_gpu_model
from keras.models import Model, load_model
from keras.layers import Activation, BatchNormalization, Dropout, Lambda
from keras.layers import Conv2D, MaxPooling2D, Input, UpSampling2D,SpatialDropout2D
from keras.layers import Dense, Flatten, Reshape, GlobalMaxPooling2D
from keras.layers import GlobalAveragePooling2D
from keras.models import *
from keras.optimizers import SGD
from keras.callbacks import *
# Confusion Matrix
from sklearn.metrics import confusion_matrix

# Create a place holder for the images
image_data=[]

# How many classed are there?
class_nb =5

# Image size
width = 160
height = 160 

# Load the data and set the parameters
def load_data(image_data, label, paths, datanumber, label_name, width=190, height=190, datatype='tif'):
    print("Now loading...")
    blob_nparray=[]
    imarrays = []
    label_for_train = label
    for i,path in enumerate(paths):
        fnamelist = glob.glob(os.path.join(path, '*.{}'.format(datatype)))
        if len(fnamelist) is not 0:
            np.random.shuffle(fnamelist)  # shuffle the fnamelist
            for filename in fnamelist[0:datanumber]:
                im = Image.open(filename)
                if im.mode is 'L' :
                    im = im.resize((width, height),Image.ANTIALIAS)
                    imarray = np.array(im)
                    imarray = (imarray - np.min(imarray))/(np.max(imarray)-np.min(imarray))
                    imarrays.append(imarray)
                else:
                    print('Images are not gray images, please check or use load_data_RGB')
                    break
            print('Files in ' +paths[i]+' are loded.')
        else:
            print('No {} files in this path.'.format(datatype))
    blob_nparray = np.reshape(np.asarray(imarrays), (len(imarrays), imarrays[1].size))
    blob_nparray = np.hstack((blob_nparray, np.arange(len(imarrays)).T.reshape(len(imarrays),1), label_for_train * np.ones((len(imarrays), 1)))) 
    image_data.append(np.asarray(blob_nparray, dtype=np.float32))
    print('Label {} {} is loaded.'.format(label,label_name))
    
load_data(image_data,
                 label = ,
                 paths = [r''],
                 datanumber = ,
                 label_name = '',
                 width  = width,
                 height = height,
                 datatype = 'tif')

#%%
# Concatenate the loaded data
for j in range(len(image_data)):
    if j is 0:
        wholedata=image_data[j][:]
    else:
        wholedata=np.concatenate((wholedata,image_data[j][:]), axis=0)

# Shuffle the data
np.random.shuffle(wholedata)

# Split the data to training and testing
def train_test_split(whole_data,ratio=0.8):
    if ratio <= 1:
        wholedata_train_upto = round(whole_data.shape[0] * ratio)
        wholedata_test_upto = whole_data.shape[0]
        print("Do you miss me?")
        train_data = whole_data[:wholedata_train_upto]
        test_data = whole_data[wholedata_train_upto:wholedata_test_upto]
    else:
        print("Ratio must be between 0 to 1.")
        
    return train_data,test_data



train_data,test_data =train_test_split(wholedata,0.8)

# Reshape the 1D data to 2D data and take the label info. out

train_label = train_data[...,-1:] 
train_data = train_data[...,:-2]
train_data = np.reshape(train_data, (train_data.shape[0],width,height,1))

test_label = test_data[...,-1:] 
test_data = test_data[...,:-2]
test_data = np.reshape(test_data, (test_data.shape[0],width,height,1))

# Create one hot label for the model
def convert_to_one_hot(y):
    
    return np.eye(np.max(y)+1)[y.reshape(-2)]


train_label = convert_to_one_hot(train_label.astype(int))
test_label = convert_to_one_hot(test_label.astype(int))

# Define model 
#%%
def Bilinear(ratio, name=None):
    return Lambda(lambda inputs: tf.image.resize_bilinear(
            inputs, tf.shape(inputs)[1:3]*ratio), name=name)
#%%
def ACmodel():
    #Input Layer
    input_img = Input(shape=(width, height, 1), name='input_layer')  # adapt this if using `channels_first` image data format
    
    #Encoder
    x = Conv2D(64, (3, 3), padding='same', name='block1_conv2')(input_img)
    x = BatchNormalization(name='block1_BN')(x)
    x = Activation('relu', name='block1_act')(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
#    x = SpatialDropout2D(0.25)(x)
    
    x = Conv2D(64, (3, 3), padding='same', name='block2_conv2')(x)
    x = BatchNormalization(name='block2_BN')(x)
    x = Activation('relu', name='block2_act')(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same', name='block2_pool')(x)
#    x = SpatialDropout2D(0.25)(x)
    
    x = Conv2D(128, (3, 3), padding='same', name='block3_conv2')(x)
    x = BatchNormalization(name='block3_BN')(x)
    x = Activation('relu', name='block3_act')(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same', name='block3_pool')(x)
#    x = SpatialDropout2D(0.25)(x)
    
    x = Conv2D(128, (3, 3), padding='same', name='block4_conv2')(x)
    x = BatchNormalization(name='block4_BN')(x)
    x = Activation('relu', name='block4_act')(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same', name='block4_pool')(x)
#    x = SpatialDropout2D(0.25)(x)
    #Bottleneck
    x = Dense((128), activation='relu', name='bottleneck_ae')(x)
    #Classification Layer
    cx = Flatten()(x)
    cx = Dropout(0.5)(cx)
#    cx = Dense((1024), activation='softmax', name='bottleneck_cs')(cx)
    class_output = Dense((class_nb), activation='softmax', name='class_output' )(cx)

    #Decoder
    x = Bilinear(2, name='block5_upsample')(x)
#    x = UpSampling2D((2, 2), name='block5_upsample')(x)
    x = Conv2D(128, (3, 3), padding='same', name='block7_conv2')(x)
    x = BatchNormalization(name='block5_BN')(x)
    x = Activation('relu', name='block5_act')(x)
#    x = SpatialDropout2D(0.25)(x)
    
    x = Bilinear(2, name='block6_upsample')(x)
#    x = UpSampling2D((2, 2), name='block6_upsample')(x)
    x = Conv2D(128, (3, 3), padding='same', name='block8_conv2')(x)
    x = BatchNormalization(name='block6_BN')(x)
    x = Activation('relu', name='block6_act')(x)
#    x = SpatialDropout2D(0.25)(x)
    
    x = Bilinear(2, name='block7_upsample')(x)
#    x = UpSampling2D((2, 2), name='block7_upsample')(x)
    x = Conv2D(64, (3, 3), padding='same', name='block9_conv2')(x)
    x = BatchNormalization(name='block7_BN')(x)
    x = Activation('relu', name='block7_act')(x)
#    x = SpatialDropout2D(0.25)(x)
    
    x = Bilinear(2, name='block8_upsample')(x)
#    x = UpSampling2D((2, 2), name='block8_upsample')(x)
    x = Conv2D(64, (3, 3), padding='same', name='block10_conv2')(x)
    x = BatchNormalization(name='block8_BN')(x)
    x = Activation('relu', name='block8_act')(x)
#    x = SpatialDropout2D(0.25)(x)
    
    decoded = Conv2D(1,(3, 3), name='decoder_output',padding='same')(x)
    autoencoder = Model(inputs = input_img, outputs = [class_output, decoded])
    
    return autoencoder

#%%
#Strat to train the model 
    
# Set the optimizer
sgd = SGD(lr=0.0008, decay=1e-6, momentum=0.9, nesterov=True)
#opt = keras.optimizers.Adam(lr=0.001)

# Because there're two GPU so it's 32*2
batch_size = 32*2

# To make the model stop at a nice place in case of overfitting
EStop = EarlyStopping(monitor='val_class_output_loss', min_delta=0, 
                      patience=6, verbose=1, mode='auto')

# Automaticaly changing the learning rate
ReduceLR = ReduceLROnPlateau(monitor='val_class_output_loss', factor=0.1, 
                             verbose=1, patience=3, mode='auto', min_lr=1e-8)

#Choose to use multi gpu or not
multigpu = True
#%%
if len(gpus) > 1:
#    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    with tf.device("/cpu:0"):
        model = ACmodel()
    parallel_model = multi_gpu_model(model, gpus=2)
    # Here's the parameters of the training process
    parallel_model.compile(loss={'class_output':'categorical_crossentropy',
                                 'decoder_output':'mean_squared_error'},
                           loss_weights={'class_output':1, 'decoder_output':1}, 
                           optimizer='Adam', metrics=['accuracy'])  # Adam can be changed to sgd 
    # training
    history = parallel_model.fit(train_data, 
                             {'class_output':train_label, 'decoder_output':train_data}, 
                             batch_size=batch_size, verbose=1, class_weight=['auto','auto'],
                             epochs=200,validation_split=0.2, callbacks=[EStop,ReduceLR])
# Evaluation
    score = parallel_model.evaluate(test_data,
                                {'class_output':test_label, 
                                 'decoder_output':test_data}, batch_size=batch_size, verbose=1)


else:
#    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    model = ACmodel()
    model.compile(loss={'class_output':'categorical_crossentropy',
                        'decoder_output':'mean_squared_error'},
                loss_weights={'class_output':1, 'decoder_output':1}, 
                optimizer='Adam', metrics=['accuracy'])  # Adam can be changed to sgd 

# training
    history = model.fit(train_data, 
                             {'class_output':train_label, 'decoder_output':train_data}, 
                             batch_size=batch_size, verbose=1, class_weight=['auto','auto'],
                             epochs=200,validation_split=0.2, callbacks=[EStop,ReduceLR])
# Evaluation
    score = model.evaluate(test_data,
                                {'class_output':test_label, 
                                 'decoder_output':test_data}, batch_size=batch_size, verbose=1)



# Save the trained Model and History
modelname=r'AE_4agonists_{}_2_(64128).h5'.format(time.strftime("%Y_%m_%d_%H_%M", time.localtime()))
model.save(modelname)

# extract history
def extracthist(history, score):
    # Save training history
    hist = history.history
    # Count the number of epoch
    for key, val in hist.items():
        numepo=len(np.asarray(val))
        break
    hist.update({'epoch':range(1,numepo+1), 'test_class_output_loss': score[2], 
                 'test_decoder_output_loss': score[1],'test_class_output_acc': score[4]})
    hist = collections.OrderedDict(hist)
    hist.move_to_end('epoch', last=False)
    return hist, numepo

hist, numepo = extracthist(history, score)
pandas.DataFrame(hist).to_excel(modelname[:-3] + '_traininghistory.xlsx', index=False)

# Reload the model for tSNE plotting and confusion matrix (because of using multiple gpu)
model = keras.models.load_model(modelname, custom_objects={'tf': tf})
model.compile(loss={'class_output':'categorical_crossentropy',
                             'decoder_output':'mean_squared_error'},
                       loss_weights={'class_output':1, 'decoder_output':1}, 
                       optimizer='Adam', metrics=['accuracy'])
#%%
# Create Confusion Matrix
test_pred = model.predict(test_data) # get the predicted results by the model
cnf_matrix = confusion_matrix(np.argmax(test_label, axis=1).reshape(-1,1),np.argmax(test_pred[0], axis=1).reshape(-1,1))
np.save(modelname[:-3]+'_cmdatacontrol.npy', cnf_matrix)
    

