import sys
import numpy as np
import pickle
import matplotlib.pyplot as plt
from image_effect import ImageEffect
import os.path

class ImagePreprocessor():

    OUTPUT_FILE   = '../data/traffic-signs-data/train_preprocessed.p'
    SCALE_FACTOR  = 3.5
    TRAINING_FILE = '../data/traffic-signs-data/train.p'

    def __init__(self):
        plt.interactive(False)

        self.train_data = self.load_data()
        self.image_effect = ImageEffect()

        self.X_train = self.train_data['features']
        self.y_train = self.train_data['labels']

    def call(self):

        self.extended_data, self.extended_labels = self.augment_data(
            self.X_train,
            self.y_train,
            scale=self.SCALE_FACTOR
        )

        self.new_train_data = {
            'features': extended_data,
            'labels': extended_labels
        }

        self.save_data()

    def plot_samples(self):
        item, count = np.unique(self.y_train, return_counts=True)
        freq = np.array((item, count)).T

        item2, count2 = np.unique(self.extended_labels, return_counts=True)
        freq2 = np.array((item2, count2)).T


        print('Before Data Preprocessing: %d samples' % (self.y_train.shape[0]))

        plt.figure(1)
        plt.bar(item, count, alpha=0.2)
        plt.title('Before Data Preprocessing: Unequally Distributed Data')

        print('After Data Preprocessing: %d samples' % (self.extended_labels.shape[0]))

        plt.figure(2)
        plt.bar(item2, count2, alpha=0.2)
        plt.title('After Data Preprocessing: More Equally Distributed Data')

        plt.show()


    def augment_data(self, X_train, y_train, scale=2):
        total_traffic_signs = len(set(y_train))

        ts, imgs_per_sign   = np.unique(y_train, return_counts=True)

        avg_per_sign        = np.ceil(np.mean(imgs_per_sign)).astype('uint32')

        separated_data      = []

        for traffic_sign in range(total_traffic_signs):
            images_in_this_sign = X_train[y_train == traffic_sign, ...]
            separated_data.append(images_in_this_sign)

        expanded_data   = np.array(np.zeros((1, 32, 32, 3)))
        expanded_labels = np.array([0])

        for sign, sign_images in enumerate(separated_data):
            scale_factor = (scale*(avg_per_sign / imgs_per_sign[sign])).astype('uint32')
            print(sign, " ", avg_per_sign / imgs_per_sign[sign], " ", scale_factor)

            new_images = []

            for img in sign_images:
                for _ in range(scale_factor):
                    new_images.append(self.image_effect.random_effect(img))

            if len(new_images) > 0:
                sign_images = np.concatenate((sign_images, new_images), axis=0)

            new_labels      = np.full(len(sign_images), sign, dtype='uint8')

            expanded_data   = np.concatenate((expanded_data, sign_images), axis=0)
            expanded_labels = np.concatenate((expanded_labels, new_labels), axis=0)

        return expanded_data[1:], expanded_labels[1:]


    def save_data(self, output_path=OUTPUT_FILE):
        bytes_out = pickle.dumps(self.new_train_data)
        max_bytes = 2**31 - 1
        n_bytes   = sys.getsizeof(bytes_out)

        with open(output_path, 'wb') as f:
            for idx in range(0, n_bytes, max_bytes):
                f.write(bytes_out[idx:idx+max_bytes])

        return True

    def load_data(self, train_path=TRAINING_FILE):
        bytes_in = bytearray(0)
        input_size = os.path.getsize(train_path)
        max_bytes = 2**31 - 1

        with open(train_path, mode='rb') as f:
            for _ in range(0, input_size, max_bytes):
                bytes_in += f.read(max_bytes)
        train = pickle.loads(bytes_in)

        return train




