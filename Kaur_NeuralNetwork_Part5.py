# coding: utf-8


import sys
import gzip
# import shutil
import os
import struct
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
import csv

# *Python Machine Learning 3rd Edition* by [Sebastian Raschka](https://sebastianraschka.com), Packt Publishing Ltd. 2019
#
# Code Repository: https://github.com/rasbt/python-machine-learning-book-3rd-edition
#
# Code License: [MIT License](https://github.com/rasbt/python-machine-learning-book-3rd-edition/blob/master/LICENSE.txt)

# # Python Machine Learning - Code Examples

# # Chapter 12 - Implementing a Multi-layer Artificial Neural Network from Scratch
#


# ## Obtaining and preparing the MNIST dataset

# The MNIST dataset is publicly available at http://yann.lecun.com/exdb/mnist/ and consists of the following four parts:
#
# - Training set images: train-images-idx3-ubyte.gz (9.9 MB, 47 MB unzipped, 60,000 examples)
# - Training set labels: train-labels-idx1-ubyte.gz (29 KB, 60 KB unzipped, 60,000 labels)
# - Test set images: t10k-images-idx3-ubyte.gz (1.6 MB, 7.8 MB, 10,000 examples)
# - Test set labels: t10k-labels-idx1-ubyte.gz (5 KB, 10 KB unzipped, 10,000 labels)
#
# In this section, we will only be working with a subset of MNIST, thus, we only need to download the training set images and training set labels.
#
# After downloading the files, simply run the next code cell to unzip the files.
#
#


# this code cell unzips mnist
'''
if (sys.version_info > (3, 0)):
    writemode = 'wb'
else:
    writemode = 'w'

zipped_mnist = [f for f in os.listdir() if f.endswith('ubyte.gz')]
for z in zipped_mnist:
    with gzip.GzipFile(z, mode='rb') as decompressed, open(z[:-3], writemode) as outfile:
        outfile.write(decompressed.read()) 


# ----
# 
# IGNORE IF THE CODE CELL ABOVE EXECUTED WITHOUT PROBLEMS:
#     
# If you have issues with the code cell above, I recommend unzipping the files using the Unix/Linux gzip tool from the terminal for efficiency, e.g., using the command 
# 
#     gzip *ubyte.gz -d
#  
# in your local MNIST download directory, or, using your favorite unzipping tool if you are working with a machine running on Microsoft Windows. The images are stored in byte form, and using the following function, we will read them into NumPy arrays that we will use to train our MLP.
# 
# Please note that if you are **not** using gzip, please make sure tha the files are named
# 
# - train-images-idx3-ubyte
# - train-labels-idx1-ubyte
# - t10k-images-idx3-ubyte
# - t10k-labels-idx1-ubyte
# 
# If a file is e.g., named `train-images.idx3-ubyte` after unzipping (this is due to the fact that certain tools try to guess a file suffix), please rename it to `train-images-idx3-ubyte` before proceeding. 
# 
# ----




def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path, 
                               '%s-labels-idx1-ubyte' % kind)
    images_path = os.path.join(path, 
                               '%s-images-idx3-ubyte' % kind)

    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', 
                                 lbpath.read(8))
        labels = np.fromfile(lbpath, 
                             dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII", 
                                               imgpath.read(16))
        images = np.fromfile(imgpath, 
                             dtype=np.uint8).reshape(len(labels), 784)
        images = ((images / 255.) - .5) * 2

    return images, labels








X_train, y_train = load_mnist('', kind='train')
print('Rows: %d, columns: %d' % (X_train.shape[0], X_train.shape[1]))




X_test, y_test = load_mnist('', kind='t10k')
print('Rows: %d, columns: %d' % (X_test.shape[0], X_test.shape[1]))


# Visualize the first digit of each class:
fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True)
ax = ax.flatten()
for i in range(10):
    img = X_train[y_train == i][0].reshape(28, 28)
    ax[i].imshow(img, cmap='Greys')

ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
# plt.savefig('images/12_5.png', dpi=300)
plt.show()


# Visualize 25 different versions of "7":
fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True,)
ax = ax.flatten()
for i in range(25):
    img = X_train[y_train == 7][i].reshape(28, 28)
    ax[i].imshow(img, cmap='Greys')

ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
# plt.savefig('images/12_6.png', dpi=300)
plt.show()
'''


# np.savez_compressed('mnist_scaled.npz',
#                     X_train=X_train,
#                     y_train=y_train,
#                     X_test=X_test,
#                     y_test=y_test)


# mnist = np.load('mnist_scaled.npz')
# mnist.files


# X_train, y_train, X_test, y_test = [mnist[f] for f in ['X_train', 'y_train',
#                                     'X_test', 'y_test']]

# del mnist

# X_train.shape


# ## Implementing a multi-layer perceptron
#reading traing data from wine train.csv
wine_training = pd.read_csv('wine-train.csv', header=None)

X_train = wine_training.iloc[:, 1:12].values - 3
y_train = wine_training.iloc[:, 12].values - 3
#kaggle
w_testing = pd.read_csv('wine-kaggle.csv', header=None)
X_test = w_testing.iloc[:, 1:12].values - 3
#y_test = w_testing[:, 12]
'''
wine_testing = pd.read_csv('wine-test.csv', header=None)
X_test = wine_testing.iloc[:, 1:12].values - 3
y_test = wine_testing.iloc[:, 12].values - 3
'''
class NeuralNetMLP(object):
    """ Feedforward neural network / Multi-layer perceptron classifier.

    Parameters
    ------------
    n_hidden : int (default: 30)
        Number of hidden units.
    l2 : float (default: 0.)
        Lambda value for L2-regularization.
        No regularization if l2=0. (default)
    epochs : int (default: 100)
        Number of passes over the training set.
    eta : float (default: 0.001)
        Learning rate.
    shuffle : bool (default: True)
        Shuffles training data every epoch if True to prevent circles.
    minibatch_size : int (default: 1)
        Number of training examples per minibatch.
    seed : int (default: None)
        Random seed for initializing weights and shuffling.

    Attributes
    -----------
    eval_ : dict
      Dictionary collecting the cost, training accuracy,
      and validation accuracy for each epoch during training.

    """

    def __init__(self, l2=0., epochs=100, eta=0.001,
                 shuffle=True, minibatch_size=1, seed=None, n_hidden_layer = []):

        self.random = np.random.RandomState(seed)
        #self.n_hidden = n_hidden
        self.l2 = l2
        self.epochs = epochs
        self.eta = eta
        self.shuffle = shuffle
        self.minibatch_size = minibatch_size
        self.n_hidden_layer = n_hidden_layer
        self.n_hidden_layer_size = len(self.n_hidden_layer)


    def _onehot(self, y, n_classes):
        """Encode labels into one-hot representation

        Parameters
        ------------
        y : array, shape = [n_examples]
            Target values.
        n_classes : int
            Number of classes

        Returns
        -----------
        onehot : array, shape = (n_examples, n_labels)

        """
        onehot = np.zeros((n_classes, y.shape[0]))
        for idx, val in enumerate(y.astype(int)):
            onehot[val, idx] = 1.
        return onehot.T

    def _sigmoid(self, z):
        """Compute logistic function (sigmoid)"""
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))

    def _forward(self, X):
        """Compute forward propagation step"""

        # step 1: net input of hidden layer
        # [n_examples, n_features] dot [n_features, n_hidden]
        # -> [n_examples, n_hidden]

        z_h1= [0] * (self.n_hidden_layer_size + 1)
        a_h1=[0] * (self.n_hidden_layer_size + 1)

        z_h1[0] = np.dot(X, self.w_h[0]) + self.b_h[0]
        a_h1[0] = self._sigmoid(z_h1[0])
        for i in range(1, self.n_hidden_layer_size):

            z_h1[i] = (np.dot(a_h1[i-1], self.w_h[i]) + self.b_h[i])
            a_h1[i] = self._sigmoid(z_h1[i])


        z_h1[self.n_hidden_layer_size] = np.dot(a_h1[self.n_hidden_layer_size-1], self.w_h[self.n_hidden_layer_size]) + self.b_h[self.n_hidden_layer_size]
        # step 4: activation output layer
        a_h1[self.n_hidden_layer_size] = self._sigmoid(z_h1[self.n_hidden_layer_size])

        return z_h1, a_h1

    def _compute_cost(self, y_enc, output):
        """Compute cost function.

        Parameters
        ----------
        y_enc : array, shape = (n_examples, n_labels)
            one-hot encoded class labels.
        output : array, shape = [n_examples, n_output_units]
            Activation of the output layer (forward propagation)

        Returns
        ---------
        cost : float
            Regularized cost

        """
        for i in range(0, self.n_hidden_layer_size):
             L2_term_wh = (np.sum(self.w_h[i] ** 2.))

        L2_term = (self.l2 *
                   (L2_term_wh))

        term1 = -y_enc * (np.log(output))
        term2 = (1. - y_enc) * np.log(1. - output)
        cost = np.sum(term1 - term2) + L2_term

        # If you are applying this cost function to other
        # datasets where activation
        # values maybe become more extreme (closer to zero or 1)
        # you may encounter "ZeroDivisionError"s due to numerical
        # instabilities in Python & NumPy for the current implementation.
        # I.e., the code tries to evaluate log(0), which is undefined.
        # To address this issue, you could add a small constant to the
        # activation values that are passed to the log function.
        #
        # For example:
        #
        # term1 = -y_enc * (np.log(output + 1e-5))
        # term2 = (1. - y_enc) * np.log(1. - output + 1e-5)

        return cost

    def predict(self, X):
        """Predict class labels

        Parameters
        -----------
        X : array, shape = [n_examples, n_features]
            Input layer with original features.

        Returns:
        ----------
        y_pred : array, shape = [n_examples]
            Predicted class labels.

        """
        z_h1, a_h1 = self._forward(X)
        y_pred = np.argmax(z_h1[self.n_hidden_layer_size-1], axis=1)
        return y_pred


    def fit(self, X_train, y_train, X_valid, y_valid):
        """ Learn weights from training data.

        Parameters
        -----------
        X_train : array, shape = [n_examples, n_features]
            Input layer with original features.
        y_train : array, shape = [n_examples]
            Target class labels.
        X_valid : array, shape = [n_examples, n_features]
            Sample features for validation during training
        y_valid : array, shape = [n_examples]
            Sample labels for validation during training

        Returns:
        ----------
        self

        """
        n_output = np.unique(y_train).shape[0]
        n_features = X_train.shape[1]

        ########################
        # Weight initialization
        ########################

        self.b_h = [0]*(self.n_hidden_layer_size+1)
        self.w_h = [0]*(self.n_hidden_layer_size+1)
        self.w_h[0] = self.random.normal(loc=0.0, scale=0.1,
                                         size=(n_features, self.n_hidden_layer[0]))
        self.b_h[0] = np.zeros(self.n_hidden_layer[0])
        for i in range(1, self.n_hidden_layer_size):

            self.b_h[i] = np.zeros(self.n_hidden_layer[i])

            self.w_h[i] = (self.random.normal(loc=0.0, scale=0.1,
                                               size=(self.n_hidden_layer[i-1], self.n_hidden_layer[i])))

        self.b_h[self.n_hidden_layer_size] = np.zeros(n_output)
        self.w_h[self.n_hidden_layer_size] = self.random.normal(loc=0.0, scale=0.1,
                                        size=(self.n_hidden_layer[self.n_hidden_layer_size-1], n_output))

        epoch_strlen = len(str(self.epochs))  # for progress formatting
        self.eval_ = {'cost': [], 'train_acc': [], 'valid_acc': []}

        y_train_enc = self._onehot(y_train, n_output)

        delta_h1 = [0] * (self.n_hidden_layer_size + 1)
        sigmoid_derivative_h1 = [0] * (self.n_hidden_layer_size + 1)
        grad_wh1 = [0] * (self.n_hidden_layer_size + 1)
        grad_bh1 = [0] * (self.n_hidden_layer_size + 1)
        # iterate over training epochs
        for i in range(self.epochs):

            # iterate over minibatches
            indices = np.arange(X_train.shape[0])

            if self.shuffle:
                self.random.shuffle(indices)

            for start_idx in range(0, indices.shape[0] - self.minibatch_size +
                                      1, self.minibatch_size):
                batch_idx = indices[start_idx:start_idx + self.minibatch_size]

                # forward propagation
                z_h1, a_h1 = self._forward(X_train[batch_idx])

                ##################
                # Backpropagation
                ##################
                delta_h1[self.n_hidden_layer_size] = a_h1[self.n_hidden_layer_size] - y_train_enc[batch_idx]

                for i in range(self.n_hidden_layer_size-1, 0, -1):

                        sigmoid_derivative_h1[i] = a_h1[i] * (1. - a_h1[i])
                        delta_h1[i] = (np.dot(delta_h1[i+1], self.w_h[i + 1].T) *
                                                           sigmoid_derivative_h1[i])

                grad_wh1[self.n_hidden_layer_size] = np.dot(a_h1[self.n_hidden_layer_size-1].T, delta_h1[self.n_hidden_layer_size])
                grad_bh1[self.n_hidden_layer_size] = np.sum(delta_h1[self.n_hidden_layer_size], axis=0)

                for i in range(self.n_hidden_layer_size-1, 0):

                    grad_wh1[i] = np.dot(a_h1[i-1].T, delta_h1[i])
                    grad_bh1[i] = np.sum(delta_h1[i], axis=0)

                grad_wh1[0] = np.dot(X_train[batch_idx].T, delta_h1[0])
                grad_bh1[0] = np.sum(delta_h1[0], axis=0)

                for i in range(self.n_hidden_layer_size, 0):
                    delta_w_h1[i] = (grad_wh1[i] + self.l2 * self.w_h[i])
                    delta_b_h1[i] = grad_bh1[i]  # bias is not regularized
                    self.w_h[i] -= self.eta * delta_wh1[i]
                    self.b_h[i] -= self.eta * delta_bh1[i]




            #############
            # Evaluation
            #############

            # Evaluation after each epoch during training
            z_h, a_h = self._forward(X_train)

            cost = self._compute_cost(y_enc=y_train_enc,
                                      output=a_h[self.n_hidden_layer_size])

            y_train_pred = self.predict(X_train)
            y_valid_pred = self.predict(X_valid)

            train_acc = ((np.sum(y_train == y_train_pred)).astype(np.float) /
                         X_train.shape[0])
            valid_acc = ((np.sum(y_valid == y_valid_pred)).astype(np.float) /
                         X_valid.shape[0])

            sys.stderr.write('\r%0*d/%d | Cost: %.2f '
                             '| Train/Valid Acc.: %.2f%%/%.2f%% ' %
                             (epoch_strlen, i + 1, self.epochs, cost,
                              train_acc * 100, valid_acc * 100))
            sys.stderr.flush()

            self.eval_['cost'].append(cost)
            self.eval_['train_acc'].append(train_acc)
            self.eval_['valid_acc'].append(valid_acc)

        return self

# START Main Code


nn = NeuralNetMLP(l2=0.01,
                  epochs=2,
                  eta=0.0005,
                  minibatch_size=50,
                  shuffle=True,
                  seed=1,
                  n_hidden_layer=[10,5])

tic = time.perf_counter()

nn.fit(X_train=X_train[:2600],
       y_train=y_train[:2600],
       X_valid=X_train[2600:],
       y_valid=y_train[2600:])

toc = time.perf_counter()

plt.plot(range(nn.epochs), nn.eval_['cost'])
plt.ylabel('Cost')
plt.xlabel('Epochs')
# plt.savefig('images/12_07.png', dpi=300)
plt.show()

plt.plot(range(nn.epochs), nn.eval_['train_acc'],
         label='Training')
plt.plot(range(nn.epochs), nn.eval_['valid_acc'],
         label='Validation', linestyle='--')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(loc='lower right')
# plt.savefig('images/12_08.png', dpi=300)
plt.show()

y_test_pred = nn.predict(X_test)

#Part  Kaggle
frame = pd.DataFrame(y_test_pred + 3, columns=['class'])
frame.index = frame.index+1
frame.to_csv('q5_kaggle-pred.csv', index_label='id')
'''
acc = (np.sum(y_test == y_test_pred)
       .astype(np.float) / X_test.shape[0])

print('Test accuracy: %.2f%%' % (acc * 100))
'''

print('\n Time for training %0.4f seconds' % (toc - tic))
'''
miscl_img = X_test[y_test != y_test_pred][:25]
correct_lab = y_test[y_test != y_test_pred][:25]
miscl_lab = y_test_pred[y_test != y_test_pred][:25]
fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True)
ax = ax.flatten()
for i in range(25):
    img = miscl_img[i].reshape(28, 28)
    ax[i].imshow(img, cmap='Greys', interpolation='nearest')
    ax[i].set_title('%d) t: %d p: %d' % (i + 1, correct_lab[i], miscl_lab[i]))

ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
# plt.savefig('images/12_09.png', dpi=300)
plt.show()
'''
