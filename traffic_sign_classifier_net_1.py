from tensorflow.contrib.layers import flatten


class TrafficSignClassifierNet(object):

    def __init__(self):
        self.learn_rate = 0.001
        self.batch_size = 128
        self.keep_prob  = 0.65
        self.features   = tf.placeholder(tf.float32, (None, 32, 32, 3))
        self.labels     = tf.placeholder(tf.int32, None)
        self.mu         = 0
        self.sigma      = 0.1
        self.accuracy   = 0.991
        self.epocs      = 50

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
        conv2 = tf.nn.dropout(conv2, keep_prob = self.keep_prob)

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
        conv4 = tf.nn.dropout(conv4, keep_prob = self.keep_prob)

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
        fc1 = tf.nn.dropout(fc1, keep_prob = self.keep_prob)

        # Layer 4: Fully Connected. Input = 1024. Output = 1024.
        fc2_W  = tf.Variable(tf.truncated_normal(shape=(1024, 1024), mean = self.mu, stddev = self.sigma))
        fc2_b  = tf.Variable(tf.zeros(1024))
        fc2    = tf.matmul(fc1, fc2_W) + fc2_b

        # Activation.
        fc2 = tf.nn.relu(fc2)
        fc2 = tf.nn.dropout(fc2, keep_prob = self.keep_prob)

        # Layer 5: Fully Connected. Input = 1024. Output = 43.
        fc3_W       = tf.Variable(tf.truncated_normal(shape=(1024, 43), mean = self.mu, stddev = self.sigma))
        fc3_b       = tf.Variable(tf.zeros(43))
        self.logits = tf.matmul(fc2, fc3_W) + fc3_b

