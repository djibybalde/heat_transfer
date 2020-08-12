"""
heat_transfer/src/process.py
"""

import os
import time
import numpy as np
import pandas as pd
import imageio

from termcolor import colored
from keras.preprocessing.image import load_img, img_to_array

from timer import timer


class ProcessData:
    """
    Data processing class.
    """

    def __init__(self,
                 data_path,
                 image_size=(50, 50),
                 verbose=True
                 ):

        """
        Constructor for data processing.

        Args:
            data_path (str): Root directory of the data (where the raw, generated, precessed data live).
            image_size (tuple): Size of the images. All the images are 50x50 RGB channels.
            verbose (bool): Verbosity mode (Whether or not verbose).

        """

        self.data_path = data_path
        self.image_size = image_size
        self.verbose = verbose

    def process_params(self):

        """
        Loads and Process the parameters data.

        Return:
            Index and scaled parameters (array type).

        """

        # Path of the parameters data.
        params_path = os.path.join(self.data_path, 'raw/params.csv')

        # Read data.
        params = pd.read_csv(params_path)

        # Columns names and index.
        columns = params.columns.values
        params_idx = [idx for idx in params.index]

        # Sum all the constants variables.
        constant = np.sum(params[columns[-4:]], axis=1)

        # Ignore the constant variables in order to scale the data.
        params = params[columns[:-4]]

        # Scaling: center and reduce (mean = 0, standard deviation = 1).
        scaled_params = (params - np.mean(params)) / np.std(params)

        # Add constant of one in the data. Note that the sum of all constants is 556.
        # scaled_params['constant'] = constant / 556.0
        scaled_params['constant'] = 1.0

        # From DataFrame to array.
        scaled_params = np.array(scaled_params)

        return params_idx, scaled_params

    def process_images(self, raw=True):

        """
        Loads and process images.

        Return:
            Index and matrix of the images.

        """

        # Paths of the images.
        images_path = os.path.join(self.data_path, 'raw/images/') if raw else self.data_path
        images_names = [x for x in sorted(os.listdir(images_path)) if x.endswith('png') or x.endswith('jpg')]

        # Store the identity of images in oder to distinguish them.
        images_idx = [int(name.split('.')[0][-4:]) for name in images_names]

        # Read and save the images.
        images_list = []
        for idx, file_name in enumerate(sorted(images_names)):
            file_path = os.path.join(images_path, file_name)

            # height*width*channels = 50x50x3
            image = load_img(file_path, target_size=self.image_size, color_mode="rgb")

            # From image to array.
            image = img_to_array(image)
            images_list.append(image)

        # From list to array using a empty matrix. We can also use np.array(images_list).
        image_array = np.empty((len(images_names),) + self.image_size + (3,))
        image_array[np.arange(len(images_names))] = images_list

        # Scale to [0, 1] and convert to float. For scaling to [-1, 1] use (images - 127.5)/127.5.
        image_array = image_array.astype('float32') / 255.0

        return images_idx, image_array

    def split_data(self, test_size=0.3, shuffle=True, save_data=True):

        """
        Splits data into train and test sample.

        Args:
            test_size (float): Between 0 and 1. Fraction of the data to be used as test sample.
            shuffle (bool): Whether or not to shuffle the index before splitting the data.
            save_data (bool): Whether or not to save the data in the data_path.

        Returns:
            Return two tuples (four array samples): (x and y train samples) and (x and y test samples).

        """

        begin = time.time()

        if self.verbose:
            print(colored('\nLoading and processing data...', 'blue'))

        # Load and process data (parameters and images).
        idx_params, params = self.process_params()
        idx_images, images = self.process_images()

        for p, i in zip(idx_params, idx_images):
            if p == i:
                idx = idx_images
            else:
                raise RuntimeError(colored("Input and output index must be same!", 'red'))

        # Shuffle the index in order to shuffle all the data without confusion.
        if shuffle:
            np.random.shuffle(idx)

        # Split data: (1-test_size)*100% for the train sample and test_size*100% for the test sample.
        len_train = int(len(idx) * (1 - test_size))
        idx_train, idx_test = idx[:len_train], idx[len_train:]

        x_train, x_test = params[idx_train, :], params[idx_test, :]
        y_train, y_test = images[idx_train, :, :], images[idx_test, :, :]

        """# Split data using Scikit-Learn library.
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(params, images,
                                                            test_size=test_size,
                                                            shuffle=False,
                                                            random_state=0)"""

        if self.verbose:
            print(colored(f'Params shape: {params.shape} ==> [train: {x_train.shape}, test: {x_test.shape}].', 'cyan'))
            print(colored(f'Images shape: {images.shape} ==> [train: {y_train.shape}, test: {y_test.shape}].', 'cyan'))

        # Create directory to save the data.
        if save_data:
            # Directory for train data.
            train_path = os.path.join(self.data_path, 'train/')
            if not os.path.exists(train_path):
                os.makedirs(train_path)
            else:
                raise RuntimeError(colored('\nThis "train/" folder already exists. You have to delete it.', 'red'))

            img_train = os.path.join(train_path, 'y_train/')
            if not os.path.exists(img_train):
                os.makedirs(img_train)

            # Directory for test data.
            test_path = os.path.join(self.data_path, 'test/')
            if not os.path.exists(test_path):
                os.makedirs(test_path)
            else:
                raise RuntimeError(colored('This "test/" folder already exists. You have to delete it.', 'red'))

            img_test = os.path.join(test_path, 'y_test/')
            if not os.path.exists(img_test):
                os.makedirs(img_test)

            # Specialize the columns of the data.
            col = ['density', 'conduct', 'capacity', 't_init', 't_top', 't_bot', 't_left', 't_right', 'const']

            # Save train data to csv.
            pd.DataFrame(x_train, columns=col, index=idx_train).to_csv(train_path + 'x_train.csv', sep=';')

            # Save train images.
            for i, j in zip(range(y_train.shape[0]), idx_train):
                img = y_train[i, :, :]
                img = np.array(255 * np.clip(img, 0, 1), dtype=np.uint8)
                imageio.imwrite(img_train + 'img{:04d}.jpg'.format(j), img)

            # Save test data to csv.
            pd.DataFrame(x_test, columns=col, index=idx_test).to_csv(test_path + 'x_test.csv', sep=';')

            # Save test images.
            for i, j in zip(range(y_test.shape[0]), idx_test):
                img = y_test[i, :, :]
                img = np.array(255 * np.clip(img, 0, 1), dtype=np.uint8)
                imageio.imwrite(img_test + 'img{:04d}.jpg'.format(j), img)

            print(colored(f'Train and test data are saved in "{self.data_path}" folder.', 'green'))
            print(colored(f'Hey! All processing are done in {timer(begin, time.time())}.', 'green'))

        return (x_train, x_test), (y_train, y_test)
