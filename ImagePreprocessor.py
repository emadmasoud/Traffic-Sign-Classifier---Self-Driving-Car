import numpy as np
import pickle
import matplotlib.pyplot as plt

class ImagePreprocessor():

    OUTPUT_FILE   = '../data/traffic-signs-data/train_preprocessed.p'
    SCALE_FACTOR  = 3.5
    TRAINING_FILE = '../data/traffic-signs-data/train.p'

    def call():
        plt.interactive(False)

        train_data = load_data()
        X_train, y_train = train_data['features'], train_data['labels']

        extended_data, extended_labels = augment_data(
                X_train,
                y_train,
                scale=SCALE_FACTOR
        )

    def augment_data(X_train, y_train, scale=2):
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
                    new_images.append(random_transform(img))

            if len(new_images) > 0:
                sign_images = np.concatenate((sign_images, new_images), axis=0)

            new_labels = np.full(len(sign_images), sign, dtype='uint8')

            expanded_data = np.concatenate((expanded_data, sign_images), axis=0)
            expanded_labels = np.concatenate((expanded_labels, new_labels), axis=0)

        return expanded_data[1:], expanded_labels[1:]



    def load_data(train_path=TRAINING_FILE):
        with open(train_path, mode='rb') as f:
            train = pickle.load(f)

        return train

    def save_data(file, path):
        with open(path, 'wb') as f:
            pickle.dump(file, f)
