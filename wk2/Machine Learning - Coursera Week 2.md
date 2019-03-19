# Machine Learning - Coursera Week 2
#MachineLearning/DeepLearning/TensorFlow/Coursera/Week2#

## Week 2 Model and Training
This is to learn computer to recognize images.

- load dataset
- construct neural network model (a graph)
- define training method
- test accuracy

### Writing code to load training data

Keras comes with some datasets: image classification, IMDB movie sentiment classification,  fashion articles, …  We can just load it from `Keras.dataset` see [Datasets - Keras Documentation](https://keras.io/datasets/)

[Keras Cheatsheet](https://s3.amazonaws.com/assets.datacamp.com/blog_assets/Keras_Cheat_Sheet_Python.pdf)

We use fashion articles dataset in this exercise.  Dataset has 60,000 28x28 grayscale images of 10 fashion categories as training dataset, along with a test set of 10,000 images as test dataset.  For more , see [GitHub - zalandoresearch/fashion-mnist: A MNIST-like fashion product database. Benchmark](https://github.com/zalandoresearch/fashion-mnist)

![](Machine%20Learning%20-%20Coursera%20Week%202/fashion-mnist-sprite.png)

```
mnist=keras.datasets.fashion_mnist
(train_images, train_label), (test_images, test_label)=mnist.load_data()
```

The class labels are:
**Label**	**Description**
0			T-shirt/top
1			Trouser
2			Pullover
3			Dress
4			Coat
5			Sandal
6			Shirt
7			Sneaker
8			Bag
9			Ankle boot

It returns 2 tuples:
		* **train_image, test_image**: uint8 array of grayscale image data with shape (num_samples, 28, 28);  For each image, there is a 28*28 array, the value of each element is 0-255 (**uint8**is unsigned 8 bit integer).  `training_images.shape =>  (60000, 28, 28)`. 
		* **train_label, test_label**: uint8 array of labels (integers in range 0-9) with shape (num_samples,). There are 60000 and 10000 train labels and test labels respectively 分别.

![](Machine%20Learning%20-%20Coursera%20Week%202/DC33EF2B-404B-472B-B21C-164A62D3B522.png)

The first image ,
```
training_images[0] => 
array([[  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   1,
          0,   0,  13,  73,   0,   0,   1,   4,   0,   0,   0,   0,   1,
          1,   0],

... first 4 row, total 28 rows x 28 columns, ... last 4 rows

       [  2,   0,   0,   0,  66, 200, 222, 237, 239, 242, 246, 243, 244,
        221, 220, 193, 191, 179, 182, 182, 181, 176, 166, 168,  99,  58,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0,  40,  61,  44,  72,  41,  35,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0]], dtype=uint8)
```

For label, `training_labels[0] => 9`  , which is an ankle boot.

### Model 

#### Define model
```
model= keras.Sequential([
	keras.layer.Flatten(input_shape(28, 28)),
	keras.layer.Dense(128, activation=tf.nn.relu),
	keras.layer.Dense(10, activation=tf.nn.softmax), 
])
```

See [Core Layers - Keras Documentation](https://keras.io/layers/core/) for keras.layer.Flatten API.
Flattens the input. Does not affect the batch size. 
Important: see more to understand tensor shape: 
	- [Understanding Tensorflow’s tensors shape: static and dynamic – P. Galeone's blog](https://pgaleone.eu/tensorflow/2018/07/28/understanding-tensorflow-tensors-shape-static-dynamic/)
	- [machine learning - Role of “Flatten” in Keras - Stack Overflow](https://stackoverflow.com/questions/43237124/role-of-flatten-in-keras) for shaping.
If inputs are shaped(batch,) without a channel dimension, then flattening adds an extra channel dimension and output shapes are(batch, 1).

Activation Function:
![](Machine%20Learning%20-%20Coursera%20Week%202/1*p_hyqAtyI8pbt2kEl6siOQ.png)

#### Build model
We will decide the training approach; which is like a training algorithm. Compile defines the loss function, the optimizer and the metrics.  There are some default parameters come with different optimizers. You need a compiled model to train (because training uses the loss function and the optimizer).

- optimizer:  
	- sgd (Stochastic gradient descent)
	- RMSprop
	- Adagrad
	- Adadelta
	- Adam
	- Adamax
	- Nesterov Adam

More details, [Optimizers - Keras Documentation](https://keras.io/optimizers/), 
Review and comparison, [An overview of gradient descent optimization algorithms](http://ruder.io/optimizing-gradient-descent/),  [Neural Network Optimization Algorithms – Towards Data Science](https://towardsdatascience.com/neural-network-optimization-algorithms-1a44c282f61d)

```
model.compile(optimizer = tf.train.AdamOptimizer(),
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=5)
```

#### Test 

`model.evaluate(test_images, test_labels)` this gives the **accuracy** of trained model with respects to test data.  This evaluates the model. 

## Others
`model.predict(test_images)` this uses the trained model, returns the **probability** of likely item of the classification. Since the shape of the output is (10,), this is a list of 10 numbers. We need to use `softmax` to get the index of the item with the maximum probability.

The model will run without `metrics` specified, default is `loss` (is not 1- accuracy).
```model.compile(optimizer = tf.train.AdamOptimizer(),
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])
```

The shape of both input and output need to match the data feed. To make them a single dimension to make feeding easier; batch is the another dimension. 

The model architecture (graph) is affected by:
- number of neurons in hidden layer,
- number of hidden layers.

The training approach is defined by:
- initial weights, loss function, optimizer, … (all parameters in `compile` method),
- epochs (`fit` method).

keras provides a `Callback` which can be triggered at end of each epoch, it can be used to re-direct the action between epochs.
```
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('loss')<0.4):
      print("\nReached 60% accuracy so cancelling training!")
      self.model.stop_training = True

callbacks = myCallback()
```

```
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
model.fit(training_images, training_labels, epochs=5, callbacks=[callbacks])
```
Read [Model (functional API) - Keras Documentation](https://keras.io/models/model/) for more options about model (`compile`) and training (`fit`).

## More …
There are many other data sources: 
* [Datasets | Kaggle](https://www.kaggle.com/datasets?utm_medium=paid&utm_source=google.com+search&utm_campaign=datasets&gclid=CjwKCAiAiJPkBRAuEiwAEDXZZY4Hbp4m5eIfQRztFzTldy8mBgI-7uM9GcZc5t2ny6I6xbM6pleEiRoCru0QAvD_BwE),
* [GitHub - awesomedata/awesome-public-datasets: A topic-centric list of HQ open datasets. PR ☛☛☛](https://github.com/awesomedata/awesome-public-datasets), 
* [UCI Machine Learning Repository: Data Sets](http://archive.ics.uci.edu/ml/datasets.html), …

Installed Kaggle with Colab. #MachineLearning/Kaggle, try some datasets later .

Watch  
- [Getting Started with TensorFlow and Deep Learning | SciPy 2018 Tutorial | Josh Gordon - YouTube](https://www.youtube.com/watch?v=tYYVSEHq-io)
- [Tensorflow for Deep Learning Research - Lecture 2 - YouTube](https://www.youtube.com/watch?v=9kC836XhICU) and notes https://cs224d.stanford.edu/lectures/CS224d-Lecture7.pdf
