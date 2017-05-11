import cv2
import numpy as np
import pandas as pd
import sklearn
import os
import math
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint

import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import pickle
from optparse import OptionParser


# define functions to be used for augmentation  
def random_brightness(img):
    """ random the entire image's brightness by a random amount
    """
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    #img_hsv = color.convert_colorspace(img, 'RGB', 'HSV')
    img_hsv[:,:,-1] = img_hsv[:,:,-1]*(np.random.uniform(0.3, 1))    
    return cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)   

def random_shadow(img):
    """ add random shawdow to the image, i. change the brightness of a region in the image
        the region determined by a random binary split of the image
    """
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    rows,cols, _ = img_hsv.shape
    row, col = np.indices((rows, cols))
    s_index = np.random.choice([-1,1],1)* (col - cols/2) + math.tan(math.pi*np.random.uniform(-0.49, 0.49))*(row-rows/2) >= np.random.uniform(-cols/2, cols/2)
    img_hsv[:,:,2][s_index] = (img_hsv[:,:,2][s_index] * 0.3).astype(np.int32)
    return cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)


def random_shift(img, x_lb=-50, x_rb=50):
    """ hift image in both x and y axis by a random amount. 
        return shifted image and shift amount
    """
    translation_x = np.random.uniform(x_lb, x_rb)
    translation_y = np.random.uniform(-3,3)    
    rows,cols,_ = img.shape
    M = np.float32([[1,0,translation_x],[0,1,translation_y]])
    return cv2.warpAffine(img,M,(cols, rows)), translation_x

def random_rotate(img, angle_lb=-5, angle_ub=5):
    """ Rotate image by a random angle
    """
    angle = np.random.uniform(angle_lb, angle_ub)
    rows, cols, _ = img.shape
    M = cv2.getRotationMatrix2D((cols/2, rows), angle, 1.0)
   
    return cv2.warpAffine(img,M,(cols, rows)), angle

# define generator 
def generator(samples, camera_pos_adjustment_coef, n_augmentation, 
              shift_factor=0, shift_limit=None,
              rotate_factor=0, rotate_limit=None,
              shadow = False,
              batch_size=32, flipimage=True):
    """ generator that reads raw image file and augment the images
    """
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        samples = sklearn.utils.shuffle(samples)
        np.random.seed(1) # reseed so that each epoch is the same
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            batch_info = []
            for key, batch_sample in batch_samples.iterrows():
                image0 = cv2.cvtColor(cv2.imread(batch_sample['image']), cv2.COLOR_BGR2RGB)
                angle0 = (float(batch_sample['steering'])
                         + float(batch_sample['camera_pos']) *  camera_pos_adjustment_coef
                        )
                
                images.append(image0)
                angles.append(angle0)                                
                if flipimage:
                    images.append(cv2.flip(image0,1))
                    angles.append(-angle0)             
                    
                for jcolor in range(0, n_augmentation):
                    # shift the image
                    image = image0.copy()
                    angle = angle0
                    if shift_limit is not None:                        
                        image, shift_delta = random_shift(image, -shift_limit, shift_limit)
                        angle = angle + shift_delta * shift_factor
                    if rotate_limit is not None:                        
                        image, rotate_delta = random_rotate(image, -rotate_limit, rotate_limit)
                        angle = angle + rotate_delta/180.0*math.pi * rotate_factor   
                        
                    # change brightness of the image
                    image = random_brightness(image)
                    if shadow:
                        image = random_shadow(image)
                    images.append(image)
                    angles.append(angle)

                    if flipimage:
                        images.append(cv2.flip(image,1))
                        angles.append(-angle)           
                
            # trim image to only see section with road
            X_train = np.array(images)
            # limit the output angles to [-1,1]
            y_train = np.clip(np.array(angles), -1,1)
                        
            yield X_train, y_train



if __name__ == '__main__':
    # the default option of this code produces the submitted model    
    parser = OptionParser()
    parser.add_option("-d", "--directory", dest="data_dir",
                      default='../behavior_data/data/',
                      type="string", 
                      help="directory of training data")
    parser.add_option("-t", "--tag", dest="tag",                     
                      type="string",
                      default='d',
                      help="tag to be used in output filename")
    parser.add_option("-a", "--adjustment", dest="adjustment_factor",
                      type="float", 
                      default=0.1,
                      help="camera adjustment factor")    
    parser.add_option("-s", "--shiftfactor", dest="shift_factor",
                      type="float", 
                      default=0.003,
                      help="shift factor")
    parser.add_option("-l", "--shiftlimit", dest="shift_limit", 
                      default=50, 
                      type="float", 
                      help="shift limit")
    parser.add_option("-r", "--rotatefactor", dest="rotate_factor",
                      type="float",
                      default=0,
                      help="rotation factor")
    parser.add_option("-e", "--rotatelimit", dest="rotate_limit", 
                      type="float",
                      help="rotation limit")
    parser.add_option("-p", "--removeprob", dest="remove_prob", 
                      default=0.7,
                      type="float",
                      help="remove probability")    
    parser.add_option("-w", "--shadow", dest="shadow", action='store_true',
                      default=True,                      
                      help="add shadow")      
    (options, args) = parser.parse_args()

    print(options)
    # load filenames and steering angles, etc.
    data_dir = options.data_dir
    labels = pd.read_csv(data_dir + 'driving_log.csv')                     
    labels[['center', 'left','right']] = labels[['center', 'left','right']].applymap(
        lambda x : data_dir + x.replace(' ', ''))

    # remove certain percentage of samples of which the steering angle is near 0
    prob_remove = options.remove_prob
    key_list = []
    for key, sample in labels.iterrows():    
        if abs(sample['steering'])<=0.02:
            if np.random.binomial(1, prob_remove):
                continue            
        key_list.append(key)
    labels_new = labels.loc[key_list,:]

    # visualize the distribution of steering angles
    f, axes = plt.subplots(1,2, sharex=True, figsize=(15,7))
    _= axes[0].hist(labels.loc[:,'steering'], bins=50)
    axes[0].set_xlabel('Steering')
    axes[0].set_ylabel('Counts for each steering angle bin before')
    axes[0].set_title('Raw')
    _= axes[1].hist(labels_new.loc[:,'steering'], bins=50)
    axes[1].set_xlabel('Steering')
    axes[1].set_ylabel('Counts of image tuples for each steering angle bin')
    axes[1].set_title('After resampling')
    f.savefig('steering_angle_distribuiton.png')


    # stack center, left and right images. Add a column 'camera_pos' to designated left, right and center images
    samples = pd.concat([labels_new.loc[:, ['center', 'steering']].rename(columns={'center':'image'}).assign(camera_pos = 0), 
                         labels_new.loc[:, ['left', 'steering']].rename(columns={'left':'image'}).assign(camera_pos = 1), 
                         labels_new.loc[:, ['right', 'steering']].rename(columns={'right':'image'}).assign(camera_pos = -1)],
                        axis=0)
    

    # define train and test generator
    train_samples, validation_samples = train_test_split(samples, random_state=0)
    
    # angle adjustment for left and right image
    camera_pos_adjustment_coef = options.adjustment_factor
    shift_factor =  options.shift_factor
    shift_limit =  options.shift_limit
    rotate_factor =  options.rotate_factor
    rotate_limit =  options.rotate_limit   
    shadow = options.shadow
    # number of random images
    n_augmentation = 10
    train_generator = generator(train_samples, 
                                camera_pos_adjustment_coef=camera_pos_adjustment_coef,
                                n_augmentation=n_augmentation,
                                shift_factor=shift_factor, shift_limit =shift_limit,
                                rotate_factor=rotate_factor, rotate_limit = rotate_limit,
                                shadow = shadow,
                                batch_size=32, flipimage=True)
    validation_generator = generator(validation_samples, 
                                     camera_pos_adjustment_coef=camera_pos_adjustment_coef,
                                     n_augmentation=n_augmentation,
                                     shift_factor=shift_factor, shift_limit =shift_limit,
                                     rotate_factor=rotate_factor, rotate_limit = rotate_limit,
                                     shadow = shadow,
                                     batch_size=32, flipimage=True)


    # CNN model                                                      
    model = Sequential()
    model.add(Cropping2D(cropping=((60,25), (0,0)), input_shape=(160,320,3)))
    model.add(Lambda(lambda x: x/255.0 -0.5))
    model.add(Convolution2D(8,3,3, activation='relu'))
    model.add(MaxPooling2D((2,2)))
    model.add(Convolution2D(16,3,3, activation='relu'))
    model.add(MaxPooling2D((2,2)))
    model.add(Convolution2D(32,3,3, activation='relu'))   
    model.add(MaxPooling2D((2,2)))   
    model.add(Convolution2D(64,3,3, activation='relu'))   
    model.add(MaxPooling2D((2,2)))         
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))   
    model.add(Dropout(0.5))
    model.add(Dense(1))    
    model.compile(loss='mse', optimizer='adam')

    # train the model and record history
    tag = "%s_%0.3f_%0.3f_%0.3f_%0.3f_%s" % (options.tag, camera_pos_adjustment_coef, shift_factor, rotate_factor, prob_remove, str(shadow))
    checkpoint = ModelCheckpoint("cp" + tag + "checkpoint-{epoch:02d}.h5",
                                 monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    model_history = model.fit_generator(train_generator,
                                        samples_per_epoch = train_samples.shape[0]*2*(1+n_augmentation),
                                        nb_epoch=15,
                                        validation_data=validation_generator,
                                        nb_val_samples = validation_samples.shape[0]*2*(1+n_augmentation),
                                        verbose=1,
                                        callbacks = [checkpoint])    
    model.save('model_f'+tag+'.h5')
    loss_hist = model_history.history
    with open( "history_" + tag + ".p", "wb") as picklefile:
        pickle.dump(loss_hist, picklefile)
