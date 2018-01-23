from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Activation,Dropout,Convolution2D,MaxPooling2D,Flatten,Lambda
from keras.optimizers import Adam
from keras.models import model_from_json
import json

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline
matplotlib.style.use('ggplot')

import os
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import cv2




input_file = 'driving_log.csv'
model_json = 'model.json'
model_weights = 'model.h5'

dataset_cols = ['center', 'left','right','steering','throttle','brake','speed']
input_dataset = pd.read_csv(input_file,names=None)
input_dataset.columns = dataset_cols

len(input_dataset)

input_dataset.hist(column='steering')

def process_image(m,l,X_train,X_left,X_right,Y_train):    
    offset=1.2 
    steering = Y_train[m]
    if l == 0:
        image = plt.imread(X_left[m].strip())
        steer_change = -offset/ 100.0 * 360/( 2*np.pi) / 25.0
        steering += steer_change
    elif l == 1:
        image = plt.imread(X_train[m].strip())
    elif l == 2:
        image = plt.imread(X_right[m].strip())
        steer_change = offset/100.0 * 360/( 2*np.pi)  / 25.0
        steering += steer_change
    
    return image,steering

def random_crop(image,tx_lower,tx_upper,ty_lower,ty_upper,steering=0.0):
    # crop out the horizon and hood subsections of the image and resize image
    shape = image.shape
    col_start,col_end =abs(tx_lower),shape[1]-tx_upper
    horizon=60
    hood=136
    
    tx= np.random.randint(tx_lower,tx_upper+1)
    ty= np.random.randint(ty_lower,ty_upper+1)
  
    random_crop = image[horizon+ty:hood+ty,col_start+tx:col_end+tx,:]
    image = cv2.resize(random_crop,(64,64),cv2.INTER_AREA)
    
    # update steering to adjust to shift    
    steer_change = -tx/(tx_upper-tx_lower)/20.0
    steering += steer_change
    
    return image,steering


def transform_image(image, steering, transfrom_range):
    #randomly transform images
    rows, cols, ch = image.shape
    dx = np.random.randint(-transfrom_range, transfrom_range + 1)    
    random_point = [cols/2+dx,rows/2]
    pts1 = np.float32([[0,rows],[cols,rows],[cols/2,rows/2]])
    pts2 = np.float32([[0,rows],[cols,rows],random_point])
    steer_change = dx/(rows/2) * 360 / (2*np.pi*25.0) / 10.0    
    trans_M = cv2.getAffineTransform(pts1,pts2)
    image = cv2.warpAffine(image,trans_M,(cols,rows),borderMode=1)
    steering +=steer_change
    
    return image,steering

def augment_brightness(image):
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_bright = 1.0 + 0.1*(2*np.random.uniform()-1.0)
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1

def flip_image(image,steering):
    toss=np.random.randint(0,2)
    if toss==0:
        image,steering=cv2.flip(image,1),-steering
    return image,steering
	

X_train = input_dataset['center']
X_left  = input_dataset['left']
X_right = input_dataset['right']
Y_train = input_dataset['steering']
Y_train = Y_train.astype(np.float32)

import random

def extract_image(name):
    image = cv2.imread(name)   
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image = image/255.-.5
    return image
    
index = random.randint(0, len(input_dataset))

center_img = extract_image(input_dataset['center'][index].strip())
right_img = extract_image(input_dataset['right'][index].strip())
left_img = extract_image(input_dataset['left'][index].strip())
plt.figure()
plt.subplot(1,3,1)
plt.imshow(left_img+.5)
plt.axis('off')
plt.subplot(1,3,2)
plt.imshow(center_img+.5)
plt.axis('off')
plt.title('Steering angle : '+ str(np.round(np.array(input_dataset.steering,dtype=np.float32)[index]*25,2) ));
plt.subplot(1,3,3)
plt.imshow(right_img+.5)
plt.axis('off')

def process_training_samples(X_train,X_left,X_right,Y_train):
    
    tx_lower,tx_upper=-20,20
    ty_lower,ty_upper=-2,2
    
    m = np.random.randint(0,len(Y_train))
    l = np.random.randint(0,3)

    image,steering = process_image(m,l,X_train,X_left,X_right,Y_train)

    image,steering = transform_image(image,steering,transfrom_range=40)

    image,steering = random_crop(image,tx_lower,tx_upper,ty_lower,ty_upper,steering)

    image,steering = flip_image(image,steering)
    
    image = augment_brightness(image)


    return image,steering
    
image,steering = process_training_samples(X_train,X_left,X_right,Y_train)
plt.axis('off')
plt.imshow(image)
plt.title('Image with brightness augmented')

def generate_train_batch(X_train,X_left,X_right,Y_train,batch_size = 32):
    
    batch_images = np.zeros((batch_size, 64, 64, 3))
    batch_steering = np.zeros(batch_size)
    while 1:
        for i_batch in range(batch_size):
            x,y = process_training_samples(X_train,X_left,X_right,Y_train)
            batch_images[i_batch] = x
            batch_steering[i_batch] = y
        yield batch_images, batch_steering
		
batch_size=200
train_generator = generate_train_batch(X_train,X_left,X_right,Y_train,batch_size)

model = Sequential()
model.add(Lambda(lambda x: x/127.5 - 1.0,input_shape=(64,64,3)))
model.add(Convolution2D(32, 3,3 ,border_mode='same', subsample=(2,2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=(1,1)))
model.add(Convolution2D(64, 3,3 ,border_mode='same',subsample=(2,2)))
model.add(Activation('relu',name='relu2'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Convolution2D(128, 3,3,border_mode='same',subsample=(1,1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(128))
model.add(Dense(1))
model.summary()


adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

restart=True
if os.path.isfile(model_json) and restart:
    try:
        with open(model_json) as jfile:
            model = model_from_json(json.load(jfile))
            model.load_weights(model_weights)    
        print('Trained model successfully loaded ')
    except Exception as e:
        print('Unable to load model', model_name, ':', e)
        raise    

model.compile(optimizer=adam, loss='mse')

nb_epoch=1
history = model.fit_generator(train_generator,
                    samples_per_epoch=20000, nb_epoch=nb_epoch,
                    verbose=1)

json_string = model.to_json()
with open(model_json, 'w') as outfile:
    json.dump(json_string, outfile)
model.save_weights(model_weights)