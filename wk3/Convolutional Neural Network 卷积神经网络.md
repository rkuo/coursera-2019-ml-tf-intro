# Convolutional Neural Network 卷积神经网络
#MachineLearning/DeepLearning/Stanford CS231n#

Stanford CS231n
- [Syllabus | CS 231N](http://cs231n.stanford.edu/syllabus.html) 2018
- Video [Lecture Collection | Convolutional Neural Networks for Visual Recognition (Spring 2017) - YouTube](https://www.youtube.com/playlist?list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv), 
- [Lecture Notes] [CS231n Convolutional Neural Networks for Visual Recognition](http://cs231n.github.io/convolutional-networks/)

- There is a good reference class, [CS231A: Computer Vision, From 3D Reconstruction to Recognition](http://web.stanford.edu/class/cs231a/syllabus.html),  has many visual recognition information.
- [Machine Learning - Convolutional Neural Network](https://www.slideshare.net/rkuo/machine-learning-convolutional-neural-network) my own presentation slides.

## What is Convolution 卷积?
[Convolution - Wikipedia](https://en.wikipedia.org/wiki/Convolution) a function derived from two given functions by integration that expresses how the shape of one is modified by the other.

![](Convolutional%20Neural%20Network%20%E5%8D%B7%E7%A7%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C/unknown.png)

## Convolution Networks
A regular 3-layer Neural Network
![](Convolutional%20Neural%20Network%20%E5%8D%B7%E7%A7%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C/neural_net2.jpeg)

Hyperparameters are setting, variables related to the architecture of a model [Neural Network Hyperparameters http://colinraffel.com/wiki/](http://colinraffel.com/wiki/neural_network_hyperparameters) , they are often set by-hand before training, for example, number of hidden layers, neurons in the hidden layer, initial weights, learning rate, …

A ConvNet arranges its neurons in three dimensions (width, height, depth), as visualized in one of the layers. Every layer of a ConvNet transforms the 3D input volume to a 3D output volume of neuron activations. In this example, the red input layer holds the image, so its width and height would be the dimensions of the image, and the depth would be 3 (Red, Green, Blue channels).

![](Convolutional%20Neural%20Network%20%E5%8D%B7%E7%A7%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C/cnn.jpeg)
From [CS231n Convolutional Neural Networks for Visual Recognition](http://cs231n.github.io/convolutional-networks/)

## Layers in ConvNets 
ConvNet is a sequence of layers, and every layer of a ConvNet transforms one volume of activations to another through a differentiable function. We use three main types of layers to build ConvNet architectures:**Convolutional Layer**,**Pooling Layer**, and**Fully-Connected Layer**(exactly as seen in regular Neural Networks). We will stack these layers plus **Input** and **output layers** to form a full ConvNet**architecture**.

*Example Architecture: Overview*. For a simple ConvNet for CIFAR-10 classification could have the architecture [INPUT - CONV - RELU - POOL - FC]. In more detail:

* INPUT [32x32x3] will hold the raw pixel values of the image, in this case an image of width 32, height 32, and with three color channels R,G,B. 

* CONV layer will compute the output of neurons that are connected to local regions in the input, each computing a dot product between their weights and a small region they are connected to in the input volume. This may result in volume such as [32x32x12] if we decided to use 12 filters.  The purpose of filter is related to discover features by incrementally discovering the similarities. 

See different kernels apply to an image online [Image Kernels explained visually](http://setosa.io/ev/image-kernels/), it also discussed about how to design kernel: for example, decide the patterns we want to discover, take a derivative  in respect to x or y for 2D images.

* RELU layer will apply an element-wise activation function, such as the *max (0, x)* thresholding at zero. This leaves the size of the volume unchanged ([32x32x12]).

More Activation Function options, from[Neuron Activation Function - GM-RKB](https://www.gabormelli.com/RKB/Neuron_Activation_Function)
![](Convolutional%20Neural%20Network%20%E5%8D%B7%E7%A7%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C/1*p_hyqAtyI8pbt2kEl6siOQ.png)

* POOL layer will perform a downsampling operation along the spatial dimensions (width, height), resulting in volume such as [16x16x12].
[What is pooling in a deep architecture? - Quora](https://www.quora.com/What-is-pooling-in-a-deep-architecture) discusses the different pooling approaches and their impacts in the paper [What is the impact of different pooling methods in convolutional neural networks? Are there any papers that compare (justify) different pooling strategies (max-pooling, average, etc.)? - Quora](https://www.quora.com/What-is-the-impact-of-different-pooling-methods-in-convolutional-neural-networks-Are-there-any-papers-that-compare-justify-different-pooling-strategies-max-pooling-average-etc):
	* max pooling
	* min pooling
	* average pooling
	* stochastic pooling
	* wavelet pooling
	* tree pooling
	* max-avg pooling
	* spatial pyramid pooling

* FC (i.e. fully-connected) layer will compute the class scores, resulting in volume of size [1x1x10], where each of the 10 numbers correspond to a class score, such as among the 10 categories of CIFAR-10. As with ordinary Neural Networks and as the name

[The best explanation of Convolutional Neural Networks on the Internet!](https://medium.com/technologymadeeasy/the-best-explanation-of-convolutional-neural-networks-on-the-internet-fbb8b1ad5df8) also describes the evolution of layer’s shape.

A more complicate ConvNets can be described as 
**INPUT -> [[CONV -> RELU]*N -> POOL?]*M -> [FC -> RELU]*K -> FC**

![](Convolutional%20Neural%20Network%20%E5%8D%B7%E7%A7%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C/958EB75A-8D69-4C08-AAED-599F5ACDCEBC.png)

## More Details on Calculation of Convolutional Layer
We don’t really know the filter (value of matrix), and will guess the initial value. Filter’s depth must be equal to input’s (previous layer’s) depth. The filter is learnable thru training. 

**Spatial arrangement**.  
Three hyper-parameters control the size of the output volume: 
- **depth** corresponds to the number of filters we would like to use, 
- **stride** with which we slide the filter.  
- **zero-padding** 

The output volume as a function of the input volume size (W), the receptive field size of the Conv Layer neurons (F), the stride with which they are applied (S), and the amount of zero padding used (P) on the border. You can convince yourself that the correct formula for calculating how many neurons “fit” is given by (W − F +2 P)/ S +1. For example for a 7x7 input and a 3x3 filter with stride 1 and pad 0 we would get a 5x5 output. With stride 2 we would get a 3x3 output. 





