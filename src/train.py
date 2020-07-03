"""
heat_transfer/src/training.py

"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from termcolor import colored

from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from keras.models import load_model

from timer import timer
from process import ProcessData
from model import TCNN_model


class Intermediate(Callback):

    """
    Callback to generate intermediate images during training.

    """

    def __init__(self,
                 x_test,
                 out_inter,
                 batch_size=100,
                 n_epochs=2000,
                 show_inter=False
                 ):

        """
        Constructor for a Intermediate object.

        Args:
            out_inter (str): Directory where outputs data and model should be saved.
            batch_size (int): Size of the batch to use. It's the number of samples per gradient update.
            n_epochs (int): Number of epochs - iteration over the entire x and y data - to train the model.
            show_inter (bool): Whether or not to show the intermediate generated images.

        Return:
            Record and show images during training.

        """

        super(Callback, self).__init__()

        self.x_test = x_test
        self.out_inter = out_inter
        self.show_inter = show_inter
        self.batch_size = batch_size
        self.n_epochs = n_epochs

    def on_epoch_end(self, epoch, logs=None):

        """
        Generate and save results to the output directory.

        """

        # Generate images during training depending for each 10 epochs.
        if (epoch % 10 == 0) or epoch == self.n_epochs - 1:

            # Number of images to generate using the test sample.
            r, c = 3, 3  # r*c images to predict.

            # Make predicting.
            predict = self.model.predict(self.x_test[:r*c, :], self.batch_size)

            # Visualize (save) the r*c predicted images.
            fig = plt.figure(figsize=(4, 4))
            for i in range(0, predict.shape[0]):
                img = predict[i, :, :]
                img = np.array(255 * np.clip(img, 0, 1), dtype=np.uint8)
                plt.subplot(r, c, i + 1)
                plt.imshow(img)
                plt.title('img{:04d}'.format(i), fontsize=10)
                plt.axis('off')
                plt.savefig(self.out_inter + 'img_E{:04d}.png'.format(epoch), bbox_inches='tight', pad_inches=0)
                plt.rcParams.update({'figure.max_open_warning': 0})

            if self.show_inter:
                plt.show()

            plt.close(fig)


def train_model(data_root,
                model_file='',
                loss='mse',
                opt_adam=None,
                batch_size=100,
                n_epochs=2000,
                val_split=0.2,
                gen_inter=True,
                show_inter=False,
                verbose=True):

    """
    Trains the TCNN model.

    Args:
        data_root (str): Root directory of the data (where the raw, generated, precessed data live).
        model_file (str): Model file to load. If none specified, a new model will be created.
        opt_adam (str): Name of optimizers or optimizer instance. If None, ADAM optimizer will be use.
        loss (str): Name of objective function or objective function or Loss instance. default: Mean Square Error
        batch_size (int): Size of the batch to use. It's the number of samples per gradient update.
        n_epochs (int): Number of epochs - iteration over the entire x and y data - to train the model.
        val_split (float): Between 0.0 and 1.0. Fraction of the training data to be used as validation data.
        verbose (bool): Whether or not to show desired the progression process.
        gen_inter (bool): Whether or not to generate intermediate images.
        show_inter (bool): Whether or not to show the intermediate generated images.

    Return:
        Return and save the trained model.

    """

    # Load an existence model of train the constructed model.
    # Load an existent model.
    if model_file:
        model = load_model(model_file)
        print("Loaded model %d from {}".format(model.model_file))

    # Train the TCNN model.
    else:
        model = TCNN_model(opt_adam=opt_adam, verbose=verbose)

    # Load data and begin training
    # Size of the images.
    img_size = model.output_shape[1:3]

    # Apply the ProcessData function to process and save the train and test data.
    processed_data = ProcessData(data_path=data_root, image_size=img_size, verbose=verbose)
    (x_train, x_test), (y_train, y_test) = processed_data.split_data(test_size=0.2, shuffle=True, save_data=True)

    # Create training callbacks list.
    callbacks = list()

    out_inter = os.path.join(data_root, 'intermediate/')
    if not os.path.exists(out_inter):
        os.makedirs(out_inter)

    if gen_inter or show_inter:
        callbacks.append(
            Intermediate(
                x_test=x_test,
                out_inter=out_inter
            )
        )

    # Create a folder for the model, if not already exist.
    out_model = '../models/'
    if not os.path.exists(out_model):
        os.makedirs(out_model)

    out_model = os.path.join(out_model, 'TCNN_model.{}'.format(str(loss).upper()))

    # Save the model at checked point.
    callbacks.append(
        ModelCheckpoint(
            out_model,
            monitor='loss',
            verbose=0,
            save_best_only=True,
            mode="auto"
        )
    )

    # Stop training when the performance of the model does not increase during training.
    callbacks.append(
        EarlyStopping(
            monitor='loss',
            min_delta=0,
            patience=10,
            mode="auto",
            restore_best_weights=True,
        )
    )

    # Train and validate model.
    begin = time.time()
    print(colored(f"\nTrain and validate TCNN model on {n_epochs} epochs with {batch_size} batch...", 'blue'))

    history = model.fit(x_train,
                        y_train,
                        batch_size=batch_size,
                        epochs=n_epochs,
                        callbacks=callbacks,
                        validation_split=val_split,
                        shuffle=True,
                        verbose=2)

    print(colored(f"\nTraining process successfully completed in {timer(begin, time.time())}!", 'green'))
    print(colored(f'The trained model is saved in the "{out_model}" folder.', 'cyan'))

    # Plot the values of the metrics in a 1x2 plots.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history.epoch, history.history['acc'])
    ax1.plot(history.epoch, history.history['val_acc'])
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Validation')
    ax1.set_title('TCNN Model Accuracy')
    ax1.legend(['train', 'validation'], loc='lower right')

    ax2.plot(history.epoch, history.history['loss'])
    ax2.plot(history.epoch, history.history['val_loss'])
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.set_title('TCNN Model Loss')
    ax2.legend(['train', 'validation'], loc='upper right')

    # Create a folder, if not exist, to save the graphs.
    reports_path = '../reports/'
    if not os.path.exists(reports_path):
        os.makedirs(reports_path)

    # Save and show the curves.
    plt.savefig(reports_path + 'EvaluationCurves.png')
    print(colored(f'The history output graph is saved in "{reports_path}" folder.', 'green'))
    plt.show()
    plt.close(fig)


# Run
root_path = '../data/'
train_model(data_root=root_path, batch_size=100, n_epochs=2000, val_split=0.2)
