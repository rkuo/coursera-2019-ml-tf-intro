# Machine Learning - Coursera Week 3
#MachineLearning/DeepLearning/TensorFlow/Coursera/Week3#

## Week 3 Convolution Network
Many times, our training data does not provide all useful information, for example, there are many blank spaces in an image. To reduce the required calculation, we will reduce the waste blank and find the similarities among them.  To take advantage of this, we can use various filters (matrix with different values), for example, a filter for vertical edge detection, F [[1,0,1], [1,0,1], [1,0,1]]. 

See [[Convolutional Neural Network 卷积神经网络]]

**Convolution**
![](Machine%20Learning%20-%20Coursera%20Week%203/6F6D1C64-E160-4124-83A2-2F6FFAA5A228.png)
If current value is 0, and its neighbors are non-0, then it could be part of edge. 
In the context of CNN, [Kernel (image processing) - Wikipedia](https://en.wikipedia.org/wiki/Kernel_(image_processing)) is same as filter.

**Activation**
Apply activation function.
![](Machine%20Learning%20-%20Coursera%20Week%203/67CDFD6C-BAD2-418D-9AEB-F1CECF6813FA.png)

**Pooling**
We can apply a threshold to reduce the data.

## Code
**Prepare data**
```
mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()
training_images=training_images.reshape(60000, 28, 28, 1)
training_images=training_images / 255.0.  
```

Fashion data, training data, is part of keras library, so we just need to load it in , which will be shaped to the same shape as filter, thru reshape(60000, 28, 28, 1).  The data will be normalized by dividing 255 (8 bits data).  

We usually need to closely analyze the available data to organize the data, to design filters.   

**Define a model**

```
 tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28, 28, 1)),
 tf.keras.layers.MaxPooling2D(2, 2),
```

See  [Conv2D](https://www.tensorflow.org/versions/r1.8/api_docs/python/tf/keras/layers/Conv2D) layers and [MaxPooling2D](https://www.tensorflow.org/versions/r1.8/api_docs/python/tf/layers/MaxPooling2D) layers in TensorFlow doc for 2D convolution layer (e.g. spatial convolution over images).

`Conv2D` will generate a convolutional layer with parameters below:
	- Conv2D will define a data format, shape (W, H, D), in this example, it is (28, 28, 1). There are 64 **filters**,  the**kernel** size is 3x3. The depth of the filter is 3, same as in training layer, (28, 28, 3). After convolution, it creates a single value, scalar. 
	
After one filter scan thru the whole input data (W, H, D),  which has been reshaped to (28, 28, 1), it creates one layer (W, H, 1). W of input data (`training_image`) can be different from `Conv2D` W. If the  padding is 0, they have the same W and H.  

The follow diagram is developed for CIFAR10, so its shape transformation is, 32x32 -> 28x28; for fashion_mnist dataset, it is 28x28 -> 26x26. 
	
![](Machine%20Learning%20-%20Coursera%20Week%203/1*mcBbGiV8ne9NhF3SlpjAsA.png)

	- We will apply  `relu` activation function to generated convolution layer. 
	- [python - Trouble figuring out how to define the input_shape in the Conv2D layer in Keras for my own dataset - Stack Overflow](https://stackoverflow.com/questions/49843113/trouble-figuring-out-how-to-define-the-input-shape-in-the-conv2d-layer-in-keras). Training or Testing data is not the first layer, they will only be referred in training `fit` method. As long as out training model concern, this `Conv2D` is the first layer.  We need specify input_shape.
	- After stride 邁, padding, the first layer needs to be defined, so it will match training_data. In our case, it is (28, 28,1).  see [[Convolutional Neural Network 卷积神经网络]] for calculation formula. 
	- **data_format**: A string, one of channels_last(default) or channels first. The ordering of the dimensions in the inputs.channels_last corresponds to inputs with shape(batch, height, width, channels) while channels_first corresponds to inputs with shape(batch, channels, height, width). 

`MaxPooling2D` [tf.layers.MaxPooling2D  |  TensorFlow](https://www.tensorflow.org/versions/r1.8/api_docs/python/tf/layers/MaxPooling2D)
Combining `activation` like remove invalid data, and `MaxPooling`, like pick the most meaningful data, these lead to data reduction. 

There are libraries for 3D convolution layer (e.g. spatial convolution over volumes), [tf.keras.layers.Conv3D  |  TensorFlow](https://www.tensorflow.org/versions/r1.8/api_docs/python/tf/keras/layers/Conv3D) and [tf.layers.MaxPooling3D  |  TensorFlow](https://www.tensorflow.org/versions/r1.8/api_docs/python/tf/layers/MaxPooling3D)

**Compile model**
This assembles/computes the computational graph.  
```
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
```

**Define training approach**
This is different from model itself. 
```
model.fit(training_images, training_labels, epochs=5)
```

**Train model**
Use Session.

## More …
ML at edge, Jetson Nano [Nvidia Announces Jetson Nano Dev Kit & Board: X1 for $99](https://www.anandtech.com/show/14101/nvidia-announces-jetson-nano) Nvidia $99 kit , Jetson family feature comparison table [Hardware For Every Situation | NVIDIA Developer](https://developer.nvidia.com/embedded/develop/hardware) #IoT/Development Boards/Jetson# 

Watch lecture 1 - 9 later [Lecture Collection | Convolutional Neural Networks for Visual Recognition (Spring 2017) - YouTube](https://www.youtube.com/playlist?list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv) Convolutional Neural Network
#MachineLearning/DeepLearning/Stanford CS231n#


