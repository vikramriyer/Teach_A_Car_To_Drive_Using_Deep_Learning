# Behavioral Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
This repository contains starting files for the Behavioral Cloning Project.

In this project, we will use deep neural networks and convolutional neural networks to clone driving behavior. The model we train using Keras will output a steering angle to an autonomous vehicle.

The Project
---
The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior 
* Design, train and validate a model that predicts a steering angle from image data
* Use the model to drive the vehicle autonomously around the first track in the simulator. The vehicle should remain on the road for an entire loop around the track.
* Summarize the results with a written report

Administrative Stuff
---
Below are the files as required in the rubric
1. [drive.py](https://github.com/vikramriyer/Teach_A_Car_To_Drive_Using_Deep_Learning/blob/master/drive.py)
2. [model.py](https://github.com/vikramriyer/Teach_A_Car_To_Drive_Using_Deep_Learning/blob/master/model.py)
3. [model.h5](https://github.com/vikramriyer/Teach_A_Car_To_Drive_Using_Deep_Learning/blob/master/model.h5) **Not viewable. Please download to run the simulator.**
4. [video](https://github.com/vikramriyer/Teach_A_Car_To_Drive_Using_Deep_Learning/blob/master/run1.mp4) of the car driving itself. **Not viewable. Please download to view.** OR [YoutubeLink](https://youtu.be/DC2Br_Sq0P4) to view (driver's view) directly. __PS: The video quality is not the greatest but I am submitting the project anyways. Will update if I create a better one after data augmentation.__
5. README => This is the github readme.

## Steps

### Dataset summary

### Dataset exploration

### Preprocessing Data
The preprocessing steps we are going to follow are listed below <br>


### Model Architecture and Training

Below is the architecture of the Model, a modified version as inspired from LeNet.

|  Layer  | Output_Shape | Total_Parameters |
|----------|-----------|-----------
||||
||||
||||
||||

The activation functions:

|  Activation  | Comments |
|----------|-----------|
| RELU | Used at the conv layers |
| SOFTMAX | Used at the last to get the probabilities of the classes |

Let's find out in short what each of the layers do: <br>
**Conv layer** <br>
We (rather the library) use a kernel or a filter that is a matrix of values and we do simple matrix multiplication and get values that are passed on as inputs to the next layers. This operation finds out certain details about the image like edges, vertices, circles, faces, etc. These kernels are chosen at random by the library and each of these produce some form of results about the features. These kernels are

**Max Pool** <br>
We simply reduce the dimentionality of the images. This methods uses the knowledge about the fact that the adjacent pixels have almost the similar contribution in terms of view of an image and hence can be removed. There are 2 famous types of pooling methods, namely, max-pooling and average pooling. In our architecture, we use the max-pooling where from a 2x2 matrix, the max value is used to construct a mapping with single pixel value.

**Fully connected layer** <br>
The convolutional layers learn some low level features and to make most of the non-linearities, we use the FC layers that perform combinations of these features and find the best of these to use. This process is again done by using back propogation which learns the best of combinations.

**Dropout** <br>
We randomly drop some information from the network. Though there is a experimental proof about this working well, we can in short say that this method reduces over fitting. It is a form of Regularization.

#### Training
Finally that our architecture is decided, we use the __Adam optimizer__ for training the model. We would not go into details of how the Adam optimizer works but in short we can say that, "**An optimizer in general minimizes the loss in the network.**"

The hyperparameters: 

|  Name  | Value |
|----------|-----------|
| EPOCHS |  |
| LEARNING RATE |  |
| BATCH SIZE |  |

## Discussion
---

### Potential Shortcomings in the Project 

### Possible Improvements
