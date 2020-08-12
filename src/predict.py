"""
heat_transfer/src/predict.py
"""

import os
import time
import imageio
import numpy as np
import pandas as pd
from termcolor import colored
from keras.models import load_model

from timer import timer


def generate(data_path,
             model_path,
             batch_size=100,
             extension='jpg'):

    # Create an output directory for the predicted data.
    out_images = os.path.join(data_path, 'predicted/')
    if not os.path.exists(out_images):
        os.makedirs(out_images)

    # Return an error if there are many models available in the same folder.
    model_name = [m for m in os.listdir(model_path) if 'TCNN' in m]
    if len(model_name) > 1:
        raise RuntimeError(colored(f'There are many models in "{model_path}". '
                                   f'Unable to load one without specification.', 'red'))
    elif len(model_name) == 0:
        raise RuntimeError(colored(f'No model to load from "{model_path}" folder.'))
    else:
        model_name = model_name[0]

    # Load model.
    print(colored('Loading "{}" model...'.format(model_name), 'blue'))
    model = load_model(model_path + model_name)

    # Load test data to make prediction.
    test_path = os.path.join(data_path, 'test/')
    x_test = pd.read_csv(test_path + 'x_test.csv', sep=';', index_col=0)

    i_text = x_test.index.values
    x_test = np.array(x_test)

    print(colored('\nPrediction starting...', 'blue'))
    start = time.time()

    # Predicting images with 100 batch size.
    predict = model.predict(x_test, batch_size=batch_size)

    # Save the predicted images. Note that the name of the predicted images is related to the index of the real one.
    for i, idx in zip(range(predict.shape[0]), i_text):
        image = predict[i, :, :]
        image = np.array(255 * np.clip(image, 0, 1), dtype=np.uint8)
        out_img = os.path.join(out_images, 'img{:04d}.{}'.format(idx, extension))
        imageio.imwrite(out_img, image)

    print(colored(f'Good news! {predict.shape[0]} images are generated in {timer(start, time.time())} '
                  f'and saved in "{out_images}" folder.', 'cyan'))


generate('../data/', '../models/', extension='jpg')
