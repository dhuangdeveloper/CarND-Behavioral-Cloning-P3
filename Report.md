# Behaviorial Cloning


---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/running_off_road.jpg "Runningoff road"
[image2]: ./examples/model.png "Grayscaling"
[image3]: ./examples/steering_angle_distribution.png "Angle distribution"
[image4]: ./examples/example_of_augmented_images.png "Augmented images"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* Report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works. model.py contains command line parser that configures the behavior of data augmentation. The default values of these options are modified to reflect the submitted models. 

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 4 convolutional layers with 3x3 filters and depths (8, 16, 32, 64), each followed by a max-pooling layer of size (2,2), and 4 fully-connected layers with number of output nodes equal to (256, 128, 32, 1) where the first free layers using the Rectified Linear Unit (ReLU) as the activation function. (model.py lines 227-246).

The data is normalized in the model using a Keras Lambda layer. Each fully connected layer uses a 0.5 drop out. 


#### 2. Attempts to reduce overfitting in the model

The model tries to minimize the number of model parameters by using multiple convolutional layers with small filter sizes instead of convolutional layers with large filter sizes. 

The model uses dropout in the fully connected layers in order to reduce overfitting (line 252 - 258)

The model was trained and validated on different data sets to ensure that the model was not overfitting (model.py lines 252 for training execution and 217-223 for validation data generator). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 246).

#### 4. Appropriate training data

I use a subset of the training data provided by Udacity. To obtain the training data, I keep all the original data with which the associated steering angle is larger than 0.02 and only keep 30% of the data with which the associated steering angle is no larger than 0.02. The data augmentation appraoch is described the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The main strategy to drive this model is find an appropriate model size, dervied from appropriately augmented training and validation data set. 

My first step was to use a convolution neural network model with 3 layers, where the first layers have filter width 5x5 and last year has filter width 3x3. The filter depths are (6, 16, 64). It is then followed by fully connected layers. However, after training the model with the data provided by Udacity, I find out that while the error on the training set decreases to a small MSE, the error on the validation set remains flat and much larger than that of the traiing error. 

To reduce over fitting, I tried to add more data set by adding data obtained from my own driving simulation. I have also tried to add dropouts to the fully connected network. I then changed the architecture of the convolutional layers to reduce the number of parameters by changing it from two layers of 5x5 and one layer of 3x3, with 35266 parameters (5x5x3x6 + 5x5x16x64 + 3x3x16x64) to 4 layers of 3x3 convlutional layers with 24408 parameters.  

To augment the data, I initally use the following steps:
1. I first use the left and right image in addition to the center image where I shift the steering angle by a certain amount $a$. For example, if the original sample has steering angle $x$, then the steering angle associated with left image would be $x+a$ and the one associated with the right image would be $x-a$.
2. I also experimented apply an affine transformation to the steering angle, i.e., $bx+a$ for the left and right image. I eventuall dropped this approach due to the difficulty to optimize two parameters and lack of evidence that the coefficient $b$ brings significant advantage. 
3. For each image added to the data, I also flip the image and add the flipped image to the data.
There are some more augmentation I took after running this model, and those will be described later. The augmentation idea of using random shadow is borrowed from http://navoshta.com/end-to-end-deep-learning/

I run the model on the simulation. The model would then veers off the road at the first left turn after the bridge where there is a openning. See below

![alt text][image1]

This suggests that the model might be overfitting to the common type of edges seen in the training data. Therefore, I added augmentation with brigthness variation in later steps.

I experimented with various $a$ and added some of my own training image. In addition, I start to drop a random subset of images whose steering angle is close to 0. While changing $a$ would have a significant impact how straight the car drives it self, (a higher $a$ sometimes lead to unstable driving behavior where the car would have large steering angles and begins to bounce between two sides of the road before it finally go off road. These do not resolve the issue when the car would veer off the road at the previously mentioned location.

I started to experiment with more augmentation techniques, including (in the order I experimented with)
1. Image translation, where I would translate the image in x and y by a random amount. The steering angle is shifted by an amount proportional to the amount of translation in x. 
2. Image rotation, where I would rotate the image by a random angle and also add a proportional amount to the steering angle. 
3. Random brightness, where I would multiply the value of the V channel in HSV representation by a random amount
4. Random shadow, where I first randomly choose a line across the image to divide the image into two parts separaand then change the brightness of one part of the image. 

By experimenting with adding these augmentations, I eventually decided to use option 1 and option 4. After these two additional augmentaions, the vehicle is able to drive around the track. In particular, option 4 helps stablize the car and also get over the previous location where the car veers off the road. On the other hand, using option 1, 2 does help get through the previous location but the car is not very stable.

#### 2. Final Model Architecture

The final model architecture (line 227-246) consisted of a convolution neural network with the following layers and filter sizes:
1. cropping layer
2. Lambda layer to normalize the input
3. Convolutional layer (3x3x8) followed by maxpooling layer (2x2)
4. Convolutional layer (3x3x16) followed by maxpooling layer (2x2)
5. Convolutional layer (3x3x32) followed by maxpooling layer (2x2)
6. Convolutional layer (3x3x64) followed by maxpooling layer (2x2)
7. Fully connected layer (output 256) with 0.5 Dropout
8. Fully connected layer (output 128)with 0.5 Dropout
9. Fully connected layer (output 32) with 0.5 Dropout
10. Fully connected layer output 1 without activation

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image2]

#### 3. Creation of the Training Set & Training Process

I use and augment the data set provided by Udacity. As described in the previous section, the following preprocessing and augmentation is done on the data set:

1) Drop a random subset (70%) of image triples (center, left, right) whose associated steering angle is near 0.
![alt text][image3]
2) Use center, left, and right image to form the data set, and add a constant value to the steering angle for the left and right image. The resulting set is the new collection of images.
3) The training and validation generator will each time read one image from the new collection of images, and perform the following augmentations 4)-7)
4) Random translation the image and add to the steering angle by a number proportional to the translation in x
5) Random change the brighness of the image
6) Random add shadow to the image
7) Add the obtained image and also the flipped version of this obtained image with a flipped angle.  

As described above, I use a generator to augment the image. I reset the seed of the random number generator after each epoch to ensure repeatibility. The augmented images from one original image are shown below:

![alt text][image4]
