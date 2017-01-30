import time
import numpy as np

import tensorflow as tf
from tensorflow.contrib.layers import flatten
from sklearn.cross_validation import train_test_split

class TrafficSignClassifierNet(object):

    def __init__(self):

        self.learn_rate    = 0.001
        self.batch_size    = 128
        self.keep_prob     = 0.65
        self.features      = tf.placeholder(tf.float32, (None, 32, 32, 3))
        self.labels        = tf.placeholder(tf.int32, None)
        self.kp            = tf.placeholder(tf.float32)
        self.learning_rate = tf.placeholder(tf.float32)
        self.mu            = 0
        self.sigma         = 0.1
        self.acc_threshold = 0.991
        self.epochs        = 25
        self.save_loc      = "model-convnet-tsc.chkpt"
        self.sess          = tf.Session()

        self.build_convnet()

    def build_convnet(self):

        # Layer 1: Convolutional. Input = 32x32x3. Output = 30x30x32.
        conv1_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 3, 32), mean = self.mu, stddev = self.sigma))
        conv1_b = tf.Variable(tf.zeros(32))
        conv1   = tf.nn.conv2d(self.features, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b
        conv1   = tf.nn.relu(conv1)

        # Layer 2: Convolutional. Output = 28x28x32.
        conv2_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 32, 32), mean = self.mu, stddev = self.sigma))
        conv2_b = tf.Variable(tf.zeros(32))
        conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
        conv2   = tf.nn.relu(conv2)

        # Pooling. Input = 28x28x32. Output = 14x14x32.
        conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        conv2 = tf.nn.dropout(conv2, keep_prob = self.kp)

        # Layer 3: Convolutional. Iutput = 14x14x32. Output = 12x12x64
        conv3_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 32, 64), mean = self.mu, stddev = self.sigma))
        conv3_b = tf.Variable(tf.zeros(64))
        conv3   = tf.nn.conv2d(conv2, conv3_W, strides=[1, 1, 1, 1], padding='VALID') + conv3_b
        conv3   = tf.nn.relu(conv3)

        # Layer 4: Convolutional. Iutput = 12x12x64. Output = 10x10x64
        conv4_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 64, 64), mean = self.mu, stddev = self.sigma))
        conv4_b = tf.Variable(tf.zeros(64))
        conv4   = tf.nn.conv2d(conv3, conv4_W, strides=[1, 1, 1, 1], padding='VALID') + conv4_b
        conv4   = tf.nn.relu(conv4)

        # Pooling. Input = 10x10x64. Output = 5x5x64.
        conv4 = tf.nn.max_pool(conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        conv4 = tf.nn.dropout(conv4, keep_prob = self.kp)

        # Layer 5: Convolutional. Iutput = 5x5x64. Output = 3x3x128
        conv5_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 64, 128), mean = self.mu, stddev = self.sigma))
        conv5_b = tf.Variable(tf.zeros(128))
        conv5   = tf.nn.conv2d(conv4, conv5_W, strides=[1, 1, 1, 1], padding='VALID') + conv5_b
        conv5   = tf.nn.relu(conv5)

        # Flatten. Input = 3x3x128. Output = 1152.
        fc0   = flatten(conv5)

        # Layer 3: Fully Connected. Input = 2048. Output = 1024.
        fc1_W = tf.Variable(tf.truncated_normal(shape=(1152, 1024), mean = self.mu, stddev = self.sigma))
        fc1_b = tf.Variable(tf.zeros(1024))
        fc1   = tf.matmul(fc0, fc1_W) + fc1_b

        # Activation.
        fc1 = tf.nn.relu(fc1)
        fc1 = tf.nn.dropout(fc1, keep_prob = self.kp)

        # Layer 4: Fully Connected. Input = 1024. Output = 1024.
        fc2_W  = tf.Variable(tf.truncated_normal(shape=(1024, 1024), mean = self.mu, stddev = self.sigma))
        fc2_b  = tf.Variable(tf.zeros(1024))
        fc2    = tf.matmul(fc1, fc2_W) + fc2_b

        # Activation.
        fc2 = tf.nn.relu(fc2)
        fc2 = tf.nn.dropout(fc2, keep_prob = self.kp)

        # Layer 5: Fully Connected. Input = 1024. Output = 43.
        fc3_W       = tf.Variable(tf.truncated_normal(shape=(1024, 43), mean = self.mu, stddev = self.sigma))
        fc3_b       = tf.Variable(tf.zeros(43))
        self.logits = tf.matmul(fc2, fc3_W) + fc3_b

    def accuracy_score(self, X_test, y_test, file_path):

        loader = tf.train.import_meta_graph(file_path)
        loader.restore(self.sess, tf.train.latest_checkpoint('./'))

        one_hot_y     = tf.one_hot(self.labels, len(set(y_test)))
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(self.logits, one_hot_y)
        loss          = tf.reduce_mean(cross_entropy)
        optimizer     = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)

        correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(one_hot_y, 1))
        accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        test_accuracy = []
        test_iter = int(len(X_test)/self.batch_size)

        for i in range(test_iter):
            X_test_batch = X_test[i*self.batch_size:(i+1)*self.batch_size]
            y_test_batch = y_test[i*self.batch_size:(i+1)*self.batch_size]
            test_accuracy = self.sess.run(accuracy_operation, feed_dict={
                self.features: X_test_batch,
                self.labels: y_test_batch,
                self.kp: 1.0,
                self.learning_rate: self.learn_rate})

        print("testing accuracy: {:0.3f}".format(test_accuracy))

    def train(self, x_train, y_train):

        X_train, X_val, y_train, y_val = train_test_split(
            x_train, y_train, train_size=0.8, test_size=0.20, random_state=42
        )

        one_hot_y = tf.one_hot(self.labels, len(set(y_train)))

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(self.logits, one_hot_y)

        loss = tf.reduce_mean(cross_entropy)

        optimizer = tf.train.AdamOptimizer(self.learn_rate).minimize(loss)

        correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(one_hot_y, 1))
        accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        self.sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()

        start_time = time.clock()

        print("Start training...")

        try:
            loader = tf.train.import_meta_graph("model-convnet-tsc.chkpt.meta")
            loader.restore(self.sess, tf.train.latest_checkpoint('./'))
            print("Restored Model Successfully.")
        except Exception as e:
            print("No model found...Start building a new one")

        for x in range(self.epochs):
            num_iter = int(len(X_train)/self.batch_size)
            for i in range(num_iter):
                X_batch = X_train[i*self.batch_size:(i+1)*self.batch_size]
                y_batch = y_train[i*self.batch_size:(i+1)*self.batch_size]

                self.sess.run(optimizer, feed_dict={
                    self.features: X_batch,
                    self.labels: y_batch,
                    self.kp: self.keep_prob,
                    self.learning_rate: self.learn_rate})

            val_accuracy    = []
            validation_iter = int(len(X_val)/self.batch_size)
            for i in range(validation_iter):
                X_val_batch = X_val[i*self.batch_size:(i+1)*self.batch_size]
                y_val_batch = y_val[i*self.batch_size:(i+1)*self.batch_size]
                val_accuracy.append(
                    self.sess.run(accuracy_operation, feed_dict={
                        self.features:X_val_batch,
                        self.labels: y_val_batch,
                        self.kp: 1}))

            print("Epoch: {}    Validation accuracy:{:.3f}".format(x, np.mean(np.array(val_accuracy))))


        saver.save(self.sess, self.save_loc)
        print("Train Model saved")

        # Calculate runtime and print out results
        train_time = time.clock() - start_time
        m, s = divmod(train_time, 60)
        h, m = divmod(m, 60)
        print("Optimization Finished!! Training time: %02dh:%02dm:%02ds"% (h, m, s))

    def predict(self, img, saved_model='model-convnet-tsc.chkpt.meta'):
        """
        Predict input
        :param img:
        :param  saved_model:
        :return: labels array
        """

        try:
            loader = tf.train.import_meta_graph(saved_model)
            loader.restore(self.sess, tf.train.latest_checkpoint('./'))
            print("Restored Model Successfully.")
        except Exception as e:
            print("No model found...Start building a new one")

        result = None

        if len(img) > 5000:
            for offset in range(0, len(img), self.batch_size):
                end = offset + self.batch_size
                batch_x = img[offset:end]
                predictions = session.run(model, feed_dict={self.features: batch_x})
                if result is None:
                    result = predictions
                else:
                    result = np.concatenate((result, predictions))
        else:
            predictions = session.run(model, feed_dict={self.features: img})
            if result is None:
                result = predictions
            else:
                result = np.concatenate((result, predictions))

        return result
