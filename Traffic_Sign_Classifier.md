# **Traffic Sign Recognition Classifier**
***

![alt text][image1]

---

## Build a Traffic Sign Recognition Project

### Introduction

In this project, we adapt and train a convolutional neural network to classify [German traffic signs](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). Use validation sets, pooling, and dropout to choose a network architecture and improve performance.

I implement our Convolutional Neural Network (CNN), LeNet using Python and the Tensorflow deep learning package to recognize traffic signs which is designed for handwritten and machine-printed character recognition. 

In order to get better results, I need to pre-process images(grayscale, normaliztion), augment data(random crop, random brightness, random contrast), generate fake data, and tweak the hyperparameters(number of filters, filter shape, max pooling shape), the LeNet-like ConvNN could achieve 97.4% accuracy.

The goals / steps of this project are the following:

* Load the data set (see below for links to the project data set)  
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report




[//]: # (Image References)

[image1]: ./resources/preface.png "Preface"
[image2]: ./resources/visualization.png "Visualization"
[image3]: ./resources/traffic_signs.png "Traffic signs"
[image4]: ./resources/histogram1.png "Histogram 1"
[image5]: ./resources/histogram2.png "Histogram 2"
[image6]: ./resources/new_images_original.png "New images original"
[image7]: ./resources/new_images_resized.png "New images resized"
[image8]: ./resources/softmax_prob.png "Softmax Probabilities"
[image9]: ./resources/feature_map.png "Feature map"

[image10]: ./resources/grayscale.png "Grayscale"
[image11]: ./resources/augmented.png "Augmented image"
[image12]: ./resources/bumpy_road.png "Bumpy road"


[image20]: ./examples/grayscale.jpg "Grayscaling"
[image21]: ./examples/random_noise.jpg "Random Noise"
[image22]: ./examples/placeholder.png "Traffic Sign 1"
[image23]: ./examples/placeholder.png "Traffic Sign 2"
[image13]: ./examples/placeholder.png "Traffic Sign 3"
[image14]: ./examples/placeholder.png "Traffic Sign 4"
[image15]: ./examples/placeholder.png "Traffic Sign 5"

### Download the data
[Download the dataset](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5898cd6f_traffic-signs-data/traffic-signs-data.zip). This is a pickled dataset in which we've already resized the images to 32x32.

The datasets we examine are consists of 35k training, 4k validation, and 12k test images of dimensions 32x32x3. There are a total of 800 images per class with 43 distinct classes.

### Data Set Summary & Exploration

**Summary dataset**

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

**Explorate visualization of the dataset**

Here is a random image from training dataset

![alt text][image2]

The following figure shows a sign for each class

![alt text][image3]

Here is an exploratory visualization of the training, validation, and test data set. It is a bar chart showing the number of signs for each classes.


![alt text][image4]

Apparently dataset is very unbalanced, and some classes are represented significantly better than the others.


### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

**Preprocessing**

As a first step, I decided to convert the images to grayscale because it helps to reduce training time without sacrificing the training accuracy.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image10]

As a last step, I normalized the image data because in the process of training the neural network, by normalizing the data, it can accelerate the convergence of the weight parameters and eliminate the differences between features.


I decided to generate additional data because It is common knowledge that the more data an ML algorithm has access to, the more effective it can be. Even when the data is of lower quality, algorithms can actually perform better, as long as useful data can
be extracted by the model from the original data set.

![alt text][image5]


**Data augmentation**

Data augmentation can act as a regularizer in preventing overfitting in neural networks and improve performance in imbalanced class problems, where we increase the amount of training data using information only in our training data. The main techniques fall under the category of data warping, which is an approach which seeks to directly augment the input data to the model in data space. 

A very generic and accepted current practice for augmenting image data is to perform geometric and color augmentations, such as reflecting the image, cropping and translating the image, and changing the color palette of the image. 

Here is an example of an original image and an augmented image:

![alt text][image11]



#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		    |     Description                               | 
|:---------------------:|:---------------------------------------------:| 
| Input         		    | 32x32x1 RGB image                             | 
| Convolution 5x5       | 1x1 stride, same padding, outputs 32x32x12    |
| RELU					    | 	                                             |
| Convolution 5x5       | 1x1 stride, valid padding, outputs 28x28x16   |
| RELU					    |	                                             |
| Max pooling	      	    | 2x2 stride,  outputs 14x14x16                 |
| Convolution 3x3       | 1x1 stride, valid padding, outputs 12x12x32   |
| RELU					    |					                                |
| Convolution 3x3       | 1x1 stride, valid padding, outputs 10x10x64   |
| RELU					    |					                                |
| Max pooling	      	    | 2x2 stride,  outputs 5x5x64                   |
| Flatten	              | 1600   			                                |
| Fully connected	    | 400                                           |  
| RELU					    |					                                |
| Dropout				    | 0.5				                                |
| Fully connected	    | 120                                           |
| RELU					    |					                                |
| Dropout				    | 0.5				                                |
| Fully connected	    | 84                                            |
| RELU					    |					                                |
| Dropout				    | 0.5				                                |
| Fully connected	    | 43                                            |
| Softmax				    | 43     			                                |

 

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used SGD to train the model and chose AdamOptimizer as the optimizer. 

The parameters I tuned were EPOCHS, BATCH_SIZE, learning rate and dropout. BATCH_SIZE, learning rate were tuned to find a good leanrning pace, so that gradient desenct would not take too long or be stuck in a local optimal point or not converge. dropout was used to prevent overfitting.

Here are the parameters: 

* EPOCHS: 20 
* BATCH_SIZE: 256 
* learning rate: 0.001
* dropout = 0.5

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:

* training set accuracy of 0.999
* validation set accuracy of 0.992 
* test set accuracy of 0.974

The improved LeNet-5 network architecture consists of 8 layers, including 4 convolutional layers, and 4 fully connected layers.

ConvNet for traffic sign classification could have the architecture [INPUT - CONV - RELU - POOL - FC]. In more detail:

* INPUT [32x32x3] will hold the raw pixel values of the image, in this case an image of width 32, height 32, and with three color channels R,G,B.
* CONV layer will compute the output of neurons that are connected to local regions in the input, each computing a dot product between their weights and a small region they are connected to in the input volume. This may result in volume such as [32x32x12] if we decided to use 12 filters.
* RELU layer will apply an elementwise activation function, such as the max(0,x)max(0,x) thresholding at zero. This leaves the size of the volume unchanged ([32x32x12]).
* POOL layer will perform a downsampling operation along the spatial dimensions (width, height), resulting in volume such as [16x16x12].
* FC (i.e. fully-connected) layer will compute the class scores, resulting in volume of size [1x1x43], where each of the 10 numbers correspond to a class score, such as among the 43 categories of traffic sing. As with ordinary Neural Networks and as the name implies, each neuron in this layer will be connected to all the numbers in the previous volume.

After data augmentation with default LeNet, i get a high accuracy on the training set but low accuracy on the validation set, which implies overfitting, dropout techniques added to reduce overfitting, dropout works by probabilistically removing an neuron from designated layers during training or by dropping certain connection.

One more convolutional layer is added to perform much more feature extraction.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are ten German traffic signs that I found on the web:

![alt text][image6]

resize image to 32x32x3

![alt text][image7]


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Bumpy road      		| Bumpy road   									| 
| Traffic signals     			| Traffic signals 										|
| Children crossing					| Children crossing											|
| Stop sign	      		| Stop sign					 				|
| Turn right ahead			| Turn right ahead  |
| Go straight or left   | Go straight or left |
| Roundabout mandatory  | Roundabout mandatory |
| Speed limit (60km/h)  | Speed limit (60km/h) |
| End of all speed and passing limits  | End of all speed and passing limits |
| Priority road         | Priority road |


The model was able to correctly guess 10 of the 10 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 97.4%

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a bumpy road sign (probability of 1.0), and the image does contain a bumpy road sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Bumpy road   									| 
| .00     				| Bicycles crossing 										|
| .00					| Wild animals crossing											|
| .00	      			| Traffic signals					 				|
| .00				    | 	Road work      							|


Plot the probabilities as below

![alt text][image8]



### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

![alt text][image12]  
Bumpy Road

![alt text][image9]

The figure above show the activations of the feature map layers for bumpy road traffic sign. The feature map activations clearly show the outline of the sign


