import tensorflow as tf
from tensorflow.contrib.layers import flatten

def weights(weight_name, shape=[]):
    """
    Return TensorFlow weights
    :param weight_name: name of weight
    :param shape: Number of features
    :return: TensorFlow weights
    """
    # Xavier Initialization - Linear only
    return tf.get_variable(weight_name, shape, initializer=tf.contrib.layers.xavier_initializer())


def biases(name, n_labels):
    """
    Return TensorFlow bias
    :param n_labels: Number of labels
    :return: TensorFlow bias
    """

    b = tf.Variable(tf.zeros(n_labels), name=name)
    return b


def conv_layer(data, size, bias, stride=1):
    """
    Create a new Convolution Layer
    """
    layer = tf.nn.conv2d(input=data, filter=size, strides=[1, stride, stride, 1], padding='SAME')
    layer = tf.nn.bias_add(layer, bias)
    layer = tf.nn.relu(layer)

    return layer


def max_pool_layer(data, sub_sampling_rate=2):
    """
    Sub-sampling data from Convolution Layer
    """
    k = sub_sampling_rate
    pool_layer = tf.nn.max_pool(value=data,
                                ksize=[1, k, k, 1],
                                strides=[1, k, k, 1],
                                padding='VALID')
    return pool_layer


def fc_layer(data, weight, bias):
    """
    Fully Connected Layer
    """
    fully_connected_layer = tf.matmul(data, weight)
    fully_connected_layer = tf.add(fully_connected_layer, bias)
    fully_connected_layer = tf.nn.relu(fully_connected_layer)

    return fully_connected_layer


def evaluate(x, y_data, features, labels, accuracy_operation, loss_op, BATCH_SIZE):
    num_of_examples = len(x)
    total_accuracy = 0
    val_loss = 0.0
    session = tf.get_default_session()

    for offset in range(0, num_of_examples, BATCH_SIZE):
        batch_features = x[offset:offset + BATCH_SIZE]
        batch_labels = y_data[offset:offset + BATCH_SIZE]

        accuracy, _loss = session.run(
            [accuracy_operation, loss_op],
            feed_dict={
                features: batch_features,
                labels: batch_labels}
        )

        total_accuracy += (accuracy * len(batch_features))
        val_loss += _loss

        return total_accuracy / num_of_examples, val_loss/num_of_examples



class TrafficSignClassifierNet(object):

    def __init__(self):
        self.learn_rate = 0.001
        self.batch_size = 128
        self.keep_prob  = 0.5
        self.features   = tf.placeholder(tf.float32, (None, 32, 32, 3))
        self.labels     = tf.placeholder(tf.int32, None)

        self.w = {
            'conv1_0': weights('conv1_0', [1, 1, 3, 3]),
            'conv1_1': weights('conv1_1', [3, 3, 3, 16]),
            'conv1_2': weights('conv1_2', [3, 3, 16, 16]),

            'conv2_1': weights('conv2_1', [3, 3, 16, 32]),
            'conv2_2': weights('conv2_2', [3, 3, 32, 32]),

            'conv3_1': weights('conv3_1', [3, 3, 32, 64]),
            'conv3_2': weights('conv3_2', [3, 3, 64, 64]),

            'fc1': weights('fc1', [4096, 1024]),
            'fc2': weights('fc2', [1024, 1024]),
            'logit': weights('logits', [1024, 43])
        }

        self.b = {
            'conv1_0': biases('conv1_0', 3),
            'conv1_1': biases('conv1_1', 16),
            'conv1_2': biases('conv1_2', 16),

            'conv2_1': biases('conv2_1', 32),
            'conv2_2': biases('conv2_2', 32),

            'conv3_1': biases('conv3_1', 64),
            'conv3_2': biases('conv3_2', 64),

            'fc1': biases('fc1', 1024),
            'fc2': biases('fc2', 1024),
            'fc3': biases('fc3', 1024),

            'logit': biases('logits', 43)
        }

        self.conv_0  = conv_layer(self.features, self.w['conv1_0'], self.b['conv1_0'])
        self.conv1_1 = conv_layer(self.conv_0, self.w['conv1_1'], self.b['conv1_1'])
        self.conv1_2 = conv_layer(self.conv1_1, self.w['conv1_2'], self.b['conv1_2'])
        self.pool_1  = max_pool_layer(self.conv1_2)
        self.pool_1  = tf.nn.dropout(self.pool_1, self.keep_prob + 0.25)

        self.conv2_1 = conv_layer(self.conv1_2, self.w['conv2_1'], self.b['conv2_1'])
        self.conv2_2 = conv_layer(self.conv2_1, self.w['conv2_2'], self.b['conv2_2'])
        self.pool_2  = max_pool_layer(self.conv2_2)
        self.pool_2  = tf.nn.dropout(self.pool_2, self.keep_prob + 0.2)

        self.conv3_1 = conv_layer(self.pool_2, self.w['conv3_1'], self.b['conv3_1'])
        self.conv3_2 = conv_layer(self.conv3_1, self.w['conv3_2'], self.b['conv3_2'])
        self.pool_3  = max_pool_layer(self.conv3_2)
        self.pool_3  = tf.nn.dropout(self.pool_3, self.keep_prob + 0.2)

        self.flatten_layer = flatten(self.pool_3)

        self.fc1 = tf.matmul(self.flatten_layer, self.w['fc1'])
        self.fc1 = tf.add(self.fc1, self.b['fc1'])
        self.fc1 = tf.nn.relu(self.fc1)
        self.fc1 = tf.nn.dropout(self.fc1, self.keep_prob)

        self.fc2 = tf.add(tf.matmul(self.fc1, self.w['fc2']), self.b['fc2'])
        self.fc2 = tf.nn.relu(self.fc2)
        self.fc2 = tf.nn.dropout(self.fc2, self.keep_prob)

        self.logits = tf.add(tf.matmul(self.fc2, self.w['logit']), self.b['logit'])

    def train(self,
              x_train, y_train,
              save_loc='./model/model-convnet-tsc.chkpt',
              epochs=10, learn_rate=0.001,
              batch_size=128, keep_prob=0.5,
              acc_threshold=0.999):

        # Update Learning Rate and Batch Size
        self.learn_rate = learn_rate
        self.batch_size = batch_size
        self.keep_prob = keep_prob

        one_hot_y = tf.one_hot(self.labels, len(set(y_train)))

        # Soft-Max
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(self.logits, one_hot_y)

        loss = tf.reduce_mean(cross_entropy)

        epoch_step = tf.Variable(0, name='epoch')

        exp_lr = tf.train.exponential_decay(
            self.learn_rate,
            epoch_step, 1000,
            0.96, staircase=True,
            name='expo_rate')

        optimizer    = tf.train.AdamOptimizer(exp_lr, name='adam_optimizer')
        training_ops = optimizer.minimize(loss, global_step=epoch_step)

        # Model Evaluation
        correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(one_hot_y, 1))
        accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        saver = tf.train.Saver()

        # Train Model
        with tf.Session() as sess:
            print("Start training...")
            start_time = time.clock()
            try:
                saver.restore(sess, save_loc)
                print("Restored Model Successfully.")
            except Exception as e:
                print(e)
                print("No model found...Start building a new one")
                sess.run(tf.global_variables_initializer())

            num_examples = len(x_train)

            for i in range(epochs):
                # Separate Training and Validation Set
                train_samples = np.ceil(int(num_examples * 0.8)).astype('uint32')
                x_train, y_train = shuffle(x_train, y_train)

                # Validation set
                x_val = x_train[train_samples:]
                y_val = y_train[train_samples:]

                print("EPOCH {}: ".format(i + 1), end="")

                for offset in range(0, train_samples, self.batch_size):
                    end = offset + self.batch_size
                    batch_x = x_train[offset:end]
                    batch_y = y_train[offset:end]

                    _, lr = sess.run(
                        [training_ops, exp_lr],
                        feed_dict={
                            self.features: batch_x,
                            self.labels: batch_y
                        }
                    )

                validation_accuracy, validation_loss = self.evaluate(
                    x_val, y_val, accuracy_operation, loss
                )

                epoch_time = time.clock() - start_time
                min, sec = divmod(epoch_time, 60)
                print("LR: {:<7.6f} Loss: {:<6.5f} Accuracy = "\
                      "{:.3f} || Time: %02dm:%02ds".format(lr,
                                                           validation_loss,
                                                           validation_accuracy,
                                                           (min, sec)
                                                          )
                     )
                if validation_accuracy > acc_threshold:
                    print("Reached accuracy requirement. Training completed.")
                    break

            saver.save(sess, save_loc)
            print("Train Model saved")

            # Calculate runtime and print out results
            train_time = time.clock() - start_time
            m, s = divmod(train_time, 60)
            h, m = divmod(m, 60)
            print("\nOptimization Finished!! Training time: %02dh:%02dm:%02ds"
                  % (h, m, s))

    def test(self, x_test, y_test, model, batch_size=128):
        # Model Evaluation
        one_hot_y = tf.one_hot(self.labels, len(set(y_test)))
        correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(one_hot_y, 1))
        accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        saver = tf.train.Saver()

        # Train Model
        with tf.Session() as sess:
            print("Start Testing...")
            saver.restore(sess, model)
            print("Restored Model Successfully.")
            num_samples = len(x_test)
            x_test, y_test = shuffle(x_test, y_test)
            print("Testing on {} samples".format(num_samples))
            total_acc = 0.0
            for offset in range(0, num_samples, batch_size):
                end = offset + batch_size
                batch_x, batch_y = x_test[offset:end], y_test[offset:end]

                _acc = sess.run(accuracy_operation, feed_dict={
                        self.features: batch_x, self.labels: batch_y
                    })

                total_acc += _acc * len(batch_x)

            print("Test Accuracy = {:.4f}".format(total_acc/num_samples))
            print("\nFinished Testing. Model is not saved")

    def evaluate(self, features, labels, acc_op, loss_op):
        num_of_examples = len(features)
        total_accuracy = 0
        val_loss = 0.0
        session = tf.get_default_session()

        for offset in range(0, num_of_examples, self.batch_size):
            batch_features = features[offset:offset + self.batch_size]
            batch_labels = labels[offset:offset + self.batch_size]

            accuracy, _loss = session.run([acc_op, loss_op],feed_dict={
                    self.features: batch_features,
                    self.labels: batch_labels
                })

            total_accuracy += (accuracy * len(batch_features))

            val_loss += _loss
        return total_accuracy / num_of_examples, val_loss/num_of_examples
