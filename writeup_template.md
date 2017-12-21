# **Traffic Sign Recognition** 

## Robert Moss Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./Writeup/train_distribution.jpg "Training distribution"
[image2]: ./Writeup/valid_distribution.jpg "Validation distribution"
[image3]: ./Writeup/test_distribution.jpg "Test distribution"
[image4]: ./Writeup/augmentation_examples.jpg "Random Noise"
[image5]: ./examples/placeholder.png "Traffic Sign 1"
[image6]: ./examples/placeholder.png "Traffic Sign 2"
[image7]: ./examples/placeholder.png "Traffic Sign 3"
[image8]: ./examples/placeholder.png "Traffic Sign 4"
[image9]: ./examples/placeholder.png "Traffic Sign 5"

Here is a link to my [project code](https://github.com/rjmoss/CarND-Traffic-Sign-Classifier-Project/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration
I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

#### Distribution
The following bar chart demonstrates the significant skew of data across the different sign types, with the most common sign type of 'Speed limit (50km/h)' having 2010 examples and the least common 'Go straight or left' with just 180 examples.

![alt text][image1]

It might be that the signs with few examples would benefit more from data augmentation in order to generate enough examples for the network to learn the features.

The training data distribution matches the validation data and testing data distribution as can be seen in the following graphs.

![alt text][image2]
![alt text][image3]

Here is an example of each one of the sign types.
![alt text][image4]

### Preprocessing

#### Grayscale
The first step was to add a gray channel to the images. The reasoning behind adding a gray channel is that many of the features of the sign (e.g. letters, shapes) are colour independent. However not all features are colour independent, the red edge of warning based signs and the blue background of signs such as 'Ahead only' or 'Keep right' are highly relevant to identifying the sign. By keeping all 4 channels (RGB + gray) we keep the relevant colour information while assisting the network to focus on shapes.

Here is an example of a traffic sign image before and after grayscaling.

#### Histogram equalization
As can be seen from the signs above, there is a large variation of brightness between signs based on the local lighting conditions. As the type of sign shouldn't depend on the lighting condition, as we don't want the network to incorrectly learn an association, `cv2.equaliseHist()` is used on each channel (including gray) to equalize the histogram. Here are a few images before and after histogram equalization:

#### Augmentation
I decided to generate additional data in order to reduce overfitting and increase the validation accuracy. The signs should be recognisable even when slightly rotated, shifted or zoomed so a combination of these (the extent of each operation was small and random) was chosen to create the fake data.

Here is an original image with some examples of augmentation:

![alt text][image4]

Only 2 augmentation examples per original image were necessary to increase the accuracy a few % points. Adding more examples might increase the accuracy yet further, also trying a few other augmentation methods such as skewing the perspective or shifting the colours slightly might also increase the accuracy of the network and decrease the chance of overfitting. Also adding augmented examples for the signs underrepresented in the training set (for example 'Go straight or left') could balance the set and make sure the neural network has sufficient examples to learn from.

Note that the augmented data underwent the same preprocessing as the original training data.

#### Normalisation
Finally I normalised the data using `pixels/255.0 -0.5` (see Gotchas below) as neural networks train best when using data centered around 0 with a small magnitude.

### Model Architecture

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:----------------------|:----------------------------------------------| 
| Input         		| 32x32x4 RGB+Gray image   						| 
| 1. Convolution 5x5    | 1x1 stride, VALID padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  VALID padding, outputs 14x14x6 	|
| 2. Convolution 5x5	| 1x1 stride, VALID padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  VALID padding, outputs 5x5x16 	|
| Flatten				| output size 400								|
| 3. Fully connected 400| output size 120								|
| RELU					|												|
| Dropout				| keep probability 0.5							|
| 4. Fully connected 120| output size 84								|
| RELU					|												|
| Dropout				| keep probability 0.5							|
| 5. Fully connected 84 | output size 43								|
| Softmax				| 												|

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an ....

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


### Possible improvements
* Increased augmentation - more augmentation examples, wider ranges of augmentation parameters, skew and other distortions
* Tailored augmentation (e.g. more augmentation for the signs which had fewer training examples)
* More training layers

### Gotchas:
I spent quite a while stuck on this problem, with my accuracy not increasing above 60%, during which I combed through the forums trying to find answers and along the way found and fixed a few "gotchas" which can have a significant effect on the model:
#### `cv2.COLOR_RGB2GRAY` vs `cv2.COLOR_BGR2GRAY`
I'm using `cv2.cvtColor()` to create the gray channel. The default for cv2 with `cv2.imread()` is to load as BGR. However, the pickled data was in the format RGB, and was loaded directly using numpy/pandas, so even though I was using cv2 to convert the color I was converting from RGB to gray. I verified this by comparing to `np.dot(img[...,:3], [0.299, 0.587, 0.114])` (which I could just have used originally instead!).
#### Normalisation
The normalisation recommended in the project of `(pixel - 128)/ 128` is actually erronous as using integers results in an overflow error (see [this](https://discussions.udacity.com/t/accuracy-is-not-going-over-75-80/314938/23)). Instead using `(pixel - 128.0)/ 128.0` (or alternatively `pixel/255.0 - 0.5` for range -0.5 to 0.5) should work correctly.
#### AWS GPU
On the AWS GPU instance tensorflow-gpu was installed however the network wasn't running particularly quickly. I suspected it might not be using the GPU, a bit of research brought me to the python command:
```
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
```
which logs information about which devices the session is assigned to. This demonstrated to me that the GPU wasn't being used. However running `$ nvidia-smi` showed me that a device existed and was recognised by the nvidia drivers (if not there was an interesting forum post [here](https://discussions.udacity.com/t/latest-nvidia-drivers-broken-on-g2-instances/231377)). It turns out that I was accidentally still using tensorflow instead of tensorflow-gpu as `pip install tensorflow-gpu` was installing `tensorflow` however `pip install tensorflow-gpu==1.2.1` doesn't do this (see [here](https://github.com/tensorflow/tensorflow/issues/12388#issuecomment-323460826)) so reinstalling a specific version of tensorflow-gpu did the job and my model trained nearly 10x faster.

#### Logits
Finally the big one which I didn't find for ages! In my architecture I had a helper function for fully connected layers:
```
def fully_connected(x, W, b):
    x = tf.add(tf.matmul(x, W), b)
    return tf.nn.relu(x)
```
which I was using for all the fully connected layers, including the final layer:
```
logits = fully_connected(x, weights['layer5'], biases['layer5'])
```
This meant that the final layer had a RELU activation, which it should not have had. This error was particularly difficult to notice as the architecture of the network looked fine, and the network was successfully training (but up to 60% max accuracy), however I found this by training with just 5 images and as I was unable to overfit on 5 images I knew it was likely there was a problem with the architecture. The correct line, without the RELU, is:
```
logits = tf.matmul(x, weights['layer5']) + biases['layer5']
```
Once corrected the accuracy immediately sprung up to >85%.


Finally worth saying that the advice in the forums and 
