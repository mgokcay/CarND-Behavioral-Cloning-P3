# **Behavioral Cloning** 


---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./WriteUpImages/NvidiaNet.png "Model Visualization"
[image2]: ./WriteUpImages/Training.png "Training"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* nvidianet.py containing the script to create the model and main.py to load images, train the model and visualize some training results. 
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup.md (this file) summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The main.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I used a model based on the convolution model from Nvidia that was introduced in the class. The original Nvidia Net is described in the following image.
I added a cropping layer at the beginning and added 3 dropout layers between the fully connected layers.

The resultant model is created in nvidianet.py file.


#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 32, 34, 36). 

The model was trained with %20 of data used as validation set to ensure that the model was not overfitting (main.py line 45). 

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track. Which the video run1.mp4 shows the test run on the whole lap.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (main.py line 43).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. 
I used a combination of center lane driving, recovering from the left and right sides of the road with a total of 3 laps. 
One full lap center lane driving, one reverse lap center lane driving and one more lap for recovering from left and right sides on the road.
I also used all three camera images.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

I used the model architecutre recomended in the class.
At first trials with only forward pass at center lane driving model did not performed well on the test.
So i added 2 more laps as i described in the training data section.
But the model still did not perform well on the test run.
Than i added the left and right camera images with a correction factor on steering angle.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

I did not use generators since my PC performed well with direct loading data. And generators slows down the training process too much.
I did not prefer to use generators.

#### 2. Final Model Architecture

I used a model based on the convolution model from Nvidia that was introduced in the class. The original Nvidia Net is described in the following image.
I added a cropping layer at the beginning and added 3 dropout layers between the fully connected layers.

The resultant model is created in nvidianet.py file.

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I recorded three laps on track:
- one forward center lane driving lap
- one backward center lane driving lap
- one forward recovering from left and right sides on the road

I used the images from all three cameras and also used flipped version of all images.

I used %20 of this data set for validation and train the model for 3 epochs.

Train and validation set errors can be seen in the following figure during epochs.

![alt text][image2]
