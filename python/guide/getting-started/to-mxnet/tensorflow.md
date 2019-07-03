
## MXNet/Gluon for users of Tensorflow

[Tensorflow](https://www.tensorflow.org/) is an open source machine learning framework developed by Google. This tutorial is to a how-to guide for new users of Apache MXNet and gluon, who have some familiarity with Tensorflow.

If you primarily use Tensorflow with the keras framework as a frontend, then you can simply switch out your keras backend to MXNet by using `keras-mxnet`. Keras-mxnet is an open-source package developed by the MXNet community that provides an MXNet backend for keras so you can keep your existing keras applications or work within the familiar keras syntax while taking advantage of the heavily optimized mxnet engine in your backend. To get started, visit the with [keras-mxnet github page](https://github.com/awslabs/keras-apache-mxnet) for installation instructions and examples.

However, to get the full experience of developing deep learning models with MXNet and gluon which provides a simple and intuitive imperative API, then this tutorial should prove a quick getting started guide.

### Installation

To install tensorflow it is common to use Python's pip package manager. For example:


```python
#! pip install tensorflow
#! pip install tensorflow-gpu #to install the tensorflow with gpu support using cuda libraries 
```

Installing mxnet can be achieved in a similar fashion. For example:


```python
#! pip install mxnet
#! pip install mxnet-cu92 #to install mxnet with gpu support using cuda libraries
```

There are also a number of other different package options for the mxnet pip installation, including support for cpu acceleration with Intel's Math Kernel Library (MKL). See the mxnet [installation page](http://mxnet.incubator.apache.org/versions/master/install/index.html) for more.

### Importing the libraries

The conventional way to import tensorflow is


```python
import tensorflow as tf
```

We will also import the mxnet library similarly. To avoid namespace clashes we will also import directly a number of sub-libraries that will be useful in our example.


```python
import mxnet as mx
from mxnet import nd, gluon, init, autograd
from mxnet.gluon import nn
```

## Model Training

Now we will go through a side by side example of how to train a model using tensorflow and using mxnet to classify images in the FashionMNIST data set.

### Loading the Data

Both libraries provide a datasets module with commonly used machine learning datasets preloaded and available for users.

Tensorflow:


```python
tf_fashion_mnist = tf.keras.datasets.fashion_mnist
(tf_train_images, tf_train_labels), (tf_test_images, tf_test_labels) = tf_fashion_mnist.load_data()
```

MXNet:


```python
mx_fashion_mnist_train = mx.gluon.data.vision.datasets.FashionMNIST(train=True)
mx_fashion_mnist_test = mx.gluon.data.vision.datasets.FashionMNIST(train=False)
```

The difference between the two is that MXNet returns the FashionMNIST data as a gluon `Dataset`. The gluon `Dataset` contains the data and labels as mxnet `ndarray`s. You also change the `train` keyword argument to return a different gluon `Dataset` for training or for validation. Tensorflow on the other hand returns the data as a numpy array.

### Data preprocessing and transformation

Since our data is already a numpy ndarray with we can apply transformations directly on the data using numpy operations.


```python
tf_train_images = tf_train_images / 255.0
tf_test_images = tf_test_images / 255.0
```

With MXNet, we can define a transformation function and use the `transform_first` method of gluon `Dataset`s to apply the transformation function on just the training data.


```python
def transformer(data):
    return data.astype('float32')/255

mx_fashion_mnist_train = mx_fashion_mnist_train.transform_first(transformer)
mx_fashion_mnist_test = mx_fashion_mnist_test.transform_first(transformer)
```

The main difference between the code snippets is that MXNet uses transform_first method to indicate that the data transformation is done on the first element of the data batch, the MNIST picture, rather than the second element, the label.

### Model definition

Tensorflow:


```python
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
```




MXNet:


```python
net = nn.HybridSequential()
with net.name_scope():
    net.add(
        nn.Flatten(),
        nn.Dense(120, activation="relu"),
        nn.Dense(10)
    )
net.initialize(init=init.Xavier())
net.hybridize()
```

The process of defining a model is similar for both frameworks. We define the model in a `Sequential` container in both, which signifies that we simply want to wire our network like a simple feed forward neural network. Both frameworks also have baked implementations of common neural network layers and are named similarly.

Notice that in the MXNet model definition, we do not included a softmax activation of the output layer. This is because the loss function that we will be using will compute softmax.

### Loss and optimizer

Now we will specify the loss function we want to optimize the model on and what optimization algorithm we intend to use to train the model.

Because Tensorflow uses the symbolic paradigm, we also have to compile the model before it can be trained. However, with MXNet gluon, in the imperative paradigm, we simply have to construct a gluon `Trainer` with the parameters of the network to be improved after every iteration, and the optimizer which implements the update steps that improves the network parameters.

Tensorflow:


```python
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
```

MXNet:


```python
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(),
                        'adam')
```

Note that the loss function we are using is the SoftmaxCrossEntropyLoss. This function computes softmax on its inputs before computing the cross entropy loss.

### Train the model

Now we are ready to train the model.

With Tensorflow/Keras, you can simply call `model.fit` to train the model. This works fine if you have defined your model correctly and you do not need to inspect the model internals. However, this is not a debuggable approach, as you have no flexibility over what happens during the actual training loop.

Tensorflow:


```python
model.fit(tf_train_images, tf_train_labels, epochs=5)
```

    Epoch 1/5
    60000/60000 [==============================] - 4s 71us/sample - loss: 0.5042 - acc: 0.8220
    Epoch 2/5
    60000/60000 [==============================] - 4s 66us/sample - loss: 0.3737 - acc: 0.8659
    Epoch 3/5
    60000/60000 [==============================] - 4s 72us/sample - loss: 0.3362 - acc: 0.87640s - loss: 0.3391 -
    Epoch 4/5
    60000/60000 [==============================] - 4s 64us/sample - loss: 0.3113 - acc: 0.8852
    Epoch 5/5
    60000/60000 [==============================] - 4s 66us/sample - loss: 0.2941 - acc: 0.8913





    <tensorflow.python.keras.callbacks.History at 0x1a3053ee50>



With MXNet/Gluon, we write a training loop. This use pattern gives the flexibility to write custom training code. First, we create a gluon `DataLoader` to load the data in batches during training. Then we define a function to compute the accuracy after every epoch and finally we have the training loop. The training loop computes the loss in the `autograd` record scope and calls backwards on the loss to compute the gradients before finally taking an optimization step with `trainer.step`.

MXNet:


```python
import time
from __future__ import division

batch_size = 64
data_size = len(mx_fashion_mnist_train)
epochs = 5
train_data = gluon.data.DataLoader(mx_fashion_mnist_train, batch_size=batch_size, shuffle=True)
test_data = gluon.data.DataLoader(mx_fashion_mnist_test, batch_size=batch_size)

def acc(output, label):
    # output: (batch, num_output) float32 ndarray
    # label: (batch, ) int32 ndarray
    acc = (output.argmax(axis=1) == label.astype('float32'))
    return acc.mean().asscalar()

for e in xrange(epochs):
    total_loss = 0
    accuracy = 0
    tic = time.time()
    for i, (data, label) in enumerate(train_data):
        with autograd.record():
            output = net(data)
            loss = softmax_cross_entropy(output, label)
        loss.backward()
        trainer.step(batch_size=128)
        total_loss += loss.mean().asscalar()
        accuracy += acc(output, label)
    print 'Epoch %d/%d' % (e+1, epochs)
    print '%ds %dus/sample - loss: %.4f - acc: %.4f'%((time.time()-tic),
                                                      (time.time()-tic)*10**6//data_size,
                                                      total_loss/i,
                                                      accuracy/i)
    
```

    Epoch 1/5
    10s 175us/sample - loss: 0.5330 - acc: 0.8167
    Epoch 2/5
    10s 172us/sample - loss: 0.3939 - acc: 0.8607
    Epoch 3/5
    10s 181us/sample - loss: 0.3563 - acc: 0.8727
    Epoch 4/5
    10s 168us/sample - loss: 0.3295 - acc: 0.8814
    Epoch 5/5
    9s 164us/sample - loss: 0.3124 - acc: 0.8867


Note that the results for each run may vary because the parameters will get different initial values and the data will be read in a different order due to shuffling.

### Validate model on test data

Tensorflow:


```python
test_loss, test_acc = model.evaluate(tf_test_images, tf_test_labels)

print('Test accuracy:', test_acc)
```

    10000/10000 [==============================] - 0s 33us/sample - loss: 0.3620 - acc: 0.8698
    ('Test accuracy:', 0.8698)


MXNet:


```python
total_accuracy = 0
for i, (data, label) in enumerate(test_data):
        with autograd.record():
            output = net(data)
        total_accuracy += acc(output, label)
print('Test accuracy:', test_accuracy/i)
```

    ('Test accuracy:', 0.8931290064102564)

