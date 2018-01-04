# Road segmentation with upsampled neural networks

- [Introduction](https://github.com/Goschjann/road_segmentation_project#intro)
- [Part 1: Explanation](https://github.com/Goschjann/road_segmentation_project#part-1-explanation-ot-the-model-architecture)
	- [Max-Pooling](https://github.com/Goschjann/road_segmentation_project#max-pooling) 
	- [Upsampling](https://github.com/Goschjann/road_segmentation_project#upsampling)
	- [Skip Connections](https://github.com/Goschjann/road_segmentation_project#skip-connections)
	- [Training](https://github.com/Goschjann/road_segmentation_project#training)
	- [Results](https://github.com/Goschjann/road_segmentation_project#skip-connections)
- [Part 2: Implementation](https://github.com/Goschjann/road_segmentation_project#part-2-implementation)

----------

__Reminder:__ This is still work in progress with eventually preliminary results. 

----------
## Intro

Within this university project we, [Jann Goschenhofer](https://github.com/Goschjann) and [Niklas Klein](https://github.com/NiklasDL), implemented a neural net for pixelwise detection of roads on aerial imagery. The project was executed in cooperation with an industry partner on the partner's private data. Therefore, this public version is applied on the [massachusets road data set](https://www.cs.toronto.edu/~vmnih/data/) kindly provided by Volodymir Mnih.

This readme is structured in two parts: in part 1, we explain our model architecture and try to give some intuition for our model. Also we visualize some of the results and compare them with Mnih's approach from his [PhD-Thesis](https://www.cs.toronto.edu/~vmnih/docs/Mnih_Volodymyr_PhD_Thesis.pdf). In the second part, we describe the usage of our code to reproduce the results. 

------------------

## Part 1: Explanation ot the model architecture

The task at hand was to create a model that outputs a probability mask in the same dimension as the input rgb image. In this mask, a probability score $P(pixel_{ij} = road | input image)$ for each input pixel from the input image is contained. This is scetched in  
![scetch of the task](/figures/architecture_1.png). 

How do we do that? As this is an image processing task and we want to include the context information of each pixel for the prediction, we want to use Convolutional Neural Networks (CNNs). This model class is the state of the art in image processing and extracts feature maps from the input image that are later used for the classification via a convolution operation. Our specific architecture roughly follows the [U-Net architecture](https://arxiv.org/abs/1505.04597) that was used for cell detection on biomedical images. 

------------------

### Max-pooling

Max-pooling is one of the key operations that allows the training of very deep CNNs. It efficiently reduces the dimensionality of the subsequent layers and therefore dramatically reduces the amount of matrix mutiplications during the train process. This is illustrated in   
![max-pooling](/figures/unet_1b_6.png).

In addition, max-pooling leads to an __translational invariance__: the extracted features are independent of their specific location. Practically speaking: a facial recognition classifier does not care if my nose is in the upper left or the lower right corner of an input image. In the past, this effect was interpreted as a positive side property of CNNs and it is the core of the current research in [capsule networks](https://hackernoon.com/what-is-a-capsnet-or-capsule-network-2bfbe48769cc). 

In our case, we are indeed interested in this spatial information as we want to make spatial predictions in the dimensionality of our input images. Thus, we need two operations: 1) a method to re-increase the dimensionality of our network and 2) a way to re-integrate the spatial information into our model. 

------------------

### Upsampling

Within this step, we use a nearest neighbor interpolation to double the dimensionality of the feature maps in the the second half of our architecture. Check this figure for an illustration
![uspsampling](unet_2c.png)

There exists a wide range of [upsampling techniques](LINK MISSING) and it would be very interesting to evaluate the performance with different methods. 

------------------

### Skip Connections

Now that we made it back to the desired dimensionality we have to re-integrate the spatial information that was lost in the pooling steps. Therefore we use __skip connections__ (also termed __merge layers__) which are illustrated as the brown arrows between the two halfs of our complete U-shaped architecture:  
![complete architecture](/figures/unet_7.png)

Check for instance the most upper skip connection. For the next convolution (white-orange block with depth 128 to blueblock with depth 64), the net can choose between 1) features that were extracted at a very early stage (white) and contain abstract but rich spatial information and 2) features from below (orange) that were extracted through the whole architecture and contain very detailed features with low spatial information. 



---------------

### Training

We implemented the model in keras and used the following training parameters:

- Optimizer: Adam (lr = 0.00001, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-08, decay = 1e-08)
- Activation: relu between convlayers and sigmoid for last dense layer
- Regularization: two dropout layers with p = 0.5 and early stopping with patience = 10
- batch size: 1
- amount of train images: 800 patches in dimension 512x512
- amount of test images: 166 512x512 patches from Minh's test set for comparison
- Thresholding of patches: we only included patches with > 5% road pixels in the training
- Threshold for probability scores: we selected 0.4 as optimal threshold for the F1 measure

Check our code for more details. As always with hyperparameters, they can be tuned and tweeked to the max and we would be super happy to receive your feedback if you toy around with our code. 

------------------

### Results

#### Error measure

As this is a highly unbalanced binary classification task, we use the F1-Score as performance evaluation and also report precision and recall.  

#### Visualization

This is


#### Comparison with Volodymir's results




## Part 2: Implementation

The code is structured as follows:

1. datagathering: execute the script _get\_mnih\_data.py_ to download the data. Before execution, adjust the paths accordingly.  
2. preprocessing: execute the srcipt _preprocess.py_ to crop the data in 512x512 patches and brings them in a format that is readable by the successive scripts. Adjust the paths accordingly. 
3. training: execute the script _train.py_ to train the architecture and store the model as a hdf5 file. Again, adjust the paths accordingly and do not forget to name your model. Also, there is the option to include a log-file. 
4. evaluation: contains two scripts: _test.py_ and _test\_loop.py_. Both use the test data to report performance measures and visualizations for each of the test images. In addition, _test\_loop.py_ tests and stores different threshold values for the probability scores that are outputed by our net. We found a threshold of 0.4 optimal for our performance measure, but this can change in other architectures, precision-recall-weighting etc. 
5. custommodule: this local library contains all necessary helper functions for the other scripts. As this is still work in progress, the library also contains many functions that are not currently in use. We tried to document the several functions and their functionality in a understandable manner.  






























