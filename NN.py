import numpy as np
import tensorflow as tf
from libs import cifar10

cifar10.maybe_download_and_extract()
train_images, train_labels = cifar10.load_training_data()
test_images, test_labels = cifar10.load_test_data()

batchCounter = 0
def nextTrainBatch(batchSize):
    global batchCounter
    batchX = train_images[batchCounter:batchCounter + batchSize, :, :]
    batchY = train_labels[batchCounter:batchCounter + batchSize, :]
    batchCounter = batchCounter + int(batchSize)
    return (batchX,batchY)

testBatchCounter = 0
def nextTestBatch(batchSize):
    global testBatchCounter
    batchX = test_images[testBatchCounter:testBatchCounter + batchSize, :, :]
    batchY = test_labels[testBatchCounter:testBatchCounter + batchSize, :]
    testBatchCounter = testBatchCounter + int(batchSize)
    return (batchX,batchY)

train_batchX, train_batchY = nextTrainBatch(25000)
test_batchX, test_batchY = nextTestBatch(1000)

X = tf.placeholder(tf.float32, [None, 32, 32, 3])
XX = tf.reshape(X,[-1,3*32*32])
X_test = tf.placeholder(tf.float32,[32,32,3])
XX_test = tf.reshape(X_test,[-1,3*32*32])

distance =  tf.reduce_sum(tf.abs(tf.add(XX, tf.negative(XX_test))), reduction_indices=1)
pred = tf.arg_min(distance, 0)
accuracy = 0.

init = tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(init)

    for i in range(len(test_batchX)):
        nn_index = sess.run(pred, feed_dict={X: train_batchX, X_test: test_batchX[i, :]})
        print( "Test", i, "Prediction:", np.argmax(train_batchY[nn_index]), "True Class:", np.argmax(test_batchY[i]))
        if np.argmax(train_batchY[nn_index]) == np.argmax(test_batchY[i]):
            accuracy += 1./len(test_batchX)
    print ("Done!")
    print ("Accuracy:", accuracy)