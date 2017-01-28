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
        batch_features, batch_labels = x[offset:offset + BATCH_SIZE], y_data[offset:offset + BATCH_SIZE]
        accuracy, _loss = session.run([accuracy_operation, loss_op], feed_dict={features: batch_features,
                                                                                labels: batch_labels})
        total_accuracy += (accuracy * len(batch_features))
        val_loss += _loss
    return total_accuracy / num_of_examples, val_loss/num_of_examples


