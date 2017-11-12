import tensorflow as tf
import math
from libs import cifar10

tf.set_random_seed(0)

cifar10.maybe_download_and_extract()
train_images, train_labels = cifar10.load_training_data()
test_images, test_labels = cifar10.load_test_data()

batchCounter = 0
def nextBatch(batchSize):
    global batchCounter
    batchX = train_images[batchCounter:batchCounter + batchSize, :, :]
    batchY = train_labels[batchCounter:batchCounter + batchSize, :]
    batchCounter = batchCounter + int(batchSize/20)
    return (batchX,batchY)

X = tf.placeholder(tf.float32, [None, 32, 32, 3])

Y_ = tf.placeholder(tf.float32, [None, 10])

lr = tf.placeholder(tf.float32)

pkeep = tf.placeholder(tf.float32)

OG = 32*32*3
L = 900
M = 300
N = 100
O = 60

W1 = tf.Variable(tf.truncated_normal([OG, L], stddev=0.1))
B1 = tf.Variable(tf.ones([L]))
W2 = tf.Variable(tf.truncated_normal([L, M], stddev=0.1))
B2 = tf.Variable(tf.ones([M]))
W3 = tf.Variable(tf.truncated_normal([M, N], stddev=0.1))
B3 = tf.Variable(tf.ones([N]))
W4 = tf.Variable(tf.truncated_normal([N, O], stddev=0.1))
B4 = tf.Variable(tf.ones([O]))
W5 = tf.Variable(tf.truncated_normal([O, 10], stddev=0.1))
B5 = tf.Variable(tf.zeros([10]))

XX = tf.reshape(X, [-1, OG])

Y1 = tf.nn.relu(tf.matmul(XX, W1) + B1)
Y1d = tf.nn.dropout(Y1, pkeep)

Y2 = tf.nn.relu(tf.matmul(Y1d, W2) + B2)
Y2d = tf.nn.dropout(Y2, pkeep)

Y3 = tf.nn.relu(tf.matmul(Y2d, W3) + B3)
Y3d = tf.nn.dropout(Y3, pkeep)

Y4 = tf.nn.relu(tf.matmul(Y3d, W4) + B4)
Y4d = tf.nn.dropout(Y4, pkeep)

Ylogits = tf.matmul(Y4d, W5) + B5
Y = tf.nn.softmax(Ylogits)


cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
cross_entropy = tf.reduce_mean(cross_entropy)

correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

allweights = tf.concat([tf.reshape(W1, [-1]), tf.reshape(W2, [-1]), tf.reshape(W3, [-1]), tf.reshape(W4, [-1]), tf.reshape(W5, [-1])], 0)
allbiases  = tf.concat([tf.reshape(B1, [-1]), tf.reshape(B2, [-1]), tf.reshape(B3, [-1]), tf.reshape(B4, [-1]), tf.reshape(B5, [-1])], 0)

train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

def training_step(i, update_test_data, update_train_data):

    batch_X, batch_Y = nextBatch(100)

    max_learning_rate = 0.003
    min_learning_rate = 0.0001
    decay_speed = 2000.0
    learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-i/decay_speed)

    if update_train_data:
        a, c, w, b = sess.run([accuracy, cross_entropy, allweights, allbiases], {X: batch_X, Y_: batch_Y, pkeep: 1.0})
        print(str(i) + ": accuracy:" + str(a) + " loss: " + str(c) + " (lr:" + str(learning_rate) + ")")

    if update_test_data:
        a, c = sess.run([accuracy, cross_entropy], {X: test_images, Y_:test_labels, pkeep: 1.0})
        print(str(i) + ": ********* epoch " + str(i*100//test_images.shape[0]+1) + " ********* test accuracy:" + str(a) + " test loss: " + str(c))

    sess.run(train_step, {X: batch_X, Y_: batch_Y, pkeep: 0.75, lr: learning_rate})


for i in range(10000+1): training_step(i, i % 100 == 0, i % 20 == 0)



