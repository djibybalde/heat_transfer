"""
heat_transfer/src/model.py

"""

from keras import Model, optimizers
from keras.layers import *
from termcolor import colored


def TCNN_model(len_params=9,
               init_shape=(4, 4, 100),
               opt_adam=None,
               loss='mse',
               ldr=0.25,
               verbose=True,
               ):

    """
    Build a TCNN model.

    Keywords in comment lines:
        TCNN2D: 2D Transposed convolution layer (Deconvolution).
        CNN2D: 2D convolution layer (spatial convolution).
        ReLU: Rectified Linear Unit  activation function.

    Args:
        len_params (int): Number of parameters to use.
        init_shape (tuple): Initial shape. A tuple of integers: height, width, channels.
        opt_adam (str): Name of optimizers or optimizer instance. If None, ADAM optimizer will be use.
        loss (str): Name of objective function or objective function or Loss instance. default: Mean Square Error
        ldr (int): Fraction of the input units to drop.
        verbose (bool): Whether or not verbose.

    Return:
        TCNN constructed model.

    """

    # Use ADAM optimizer as default.
    if opt_adam is None:
        lr, b1, b2 = 0.0001, 0.5, 0.99
        opt_adam = optimizers.Adam(learning_rate=lr, beta_1=b1, beta_2=b2)
        print(colored(f'\nThis model uses ADAM optimizer with {lr} learning rate and beta1={b1}, beta2={b2}.', 'green'))

    # Initial size.
    height, width, channels = init_shape

    # Input parameters. Specialize the number of parameters to use.
    input_layer = Input(shape=(len_params,), name='params', )

    # First fully connected layer with RelU.
    layer = Dense(units=625, use_bias=True, name='Dense_1', )(input_layer)
    layer = LeakyReLU(alpha=0.3, name='ReLu_d1', )(layer)

    # Apply Dropout to the input.
    layer = Dropout(rate=ldr, seed=32, trainable=False, name='Dropout', )(layer)

    # Second fully connected layer with ReLU.
    layer = Dense(units=1250, use_bias=True, name='Dense_2', )(layer)
    layer = LeakyReLU(alpha=0.3, name='ReLu_d2', )(layer)

    # Third fully connected layer with ReLU.
    layer = Dense(units=height * width * channels, use_bias=True, name='Dense_3', )(layer)
    layer = LeakyReLU(alpha=0.3, name='ReLu_3', )(layer)

    # Reshape from 1600 to 4x4x100.
    layer = Reshape(target_shape=(height, width, channels), name='Reshape_', )(layer)

    # Apply 5x5 TCNN2D and CNN2D: Up-sample from 4x4x100, to 11x11x48, 25x25x24 and 50x50x3 with 3 de-convolutions.

    # First TCNN2D with ReLU: Up-simple to 11x11x48.
    layer = Conv2DTranspose(filters=48, kernel_size=(5, 5), strides=(2, 2), padding='valid', name='Transp_1', )(layer)
    layer = LeakyReLU(alpha=0.3, name='ReLu_t1', )(layer)

    # First CNN2D with ReLU: Up-sample to 11x11x48.
    layer = Conv2D(filters=48, kernel_size=(5, 5), strides=(1, 1), padding='same', name='Conv2D_1', )(layer)
    layer = LeakyReLU(alpha=0.3, name='ReLu_c1', )(layer)

    # Second TCNN2D with ReLU: Up-simple to 25x25x24.
    layer = Conv2DTranspose(filters=24, kernel_size=(5, 5), strides=(2, 2), padding='valid', name='Transp_2', )(layer)
    layer = LeakyReLU(alpha=0.3, name='ReLu_t2', )(layer)

    # Second CNN2D with ReLU: Up-sample to 25x25x24.
    layer = Conv2D(filters=24, kernel_size=(5, 5), strides=(1, 1), padding='same', name='Conv2D_2', )(layer)
    layer = LeakyReLU(alpha=0.3, name='ReLu_c2', )(layer)

    # Third TCNN2D with ReLU: Up-simple to 50x50x3.
    layer = Conv2DTranspose(filters=3, kernel_size=(5, 5), strides=(2, 2), padding='same', name='Transp_3', )(layer)
    layer = LeakyReLU(alpha=0.3, name='ReLu_t3', )(layer)

    # Third CNN2D with ReLU: Up-sample to 50x50x3.
    layer = Conv2D(filters=3, kernel_size=(5, 5), strides=(1, 1), padding='same', name='Conv2D_3', )(layer)
    layer = LeakyReLU(alpha=0.3, name='ReLu_c3', )(layer)

    # Compile the model.
    model = Model(inputs=input_layer, outputs=layer, name='TCNN model')
    model.compile(optimizer=opt_adam, loss=loss, metrics=['acc', 'mae', 'mse'])

    # Show some information.
    if verbose:
        for layer in model.layers:
            if 'ReLu' not in layer.name and 'params' not in layer.name:
                print(f'{layer.name}:, [output shape: {layer.output_shape}, trainable? {layer.trainable}]')

        print(colored('\nTCNN model has {} trainable parameters.', 'green').format(model.count_params()))

        _, df = model.layers[0].input_shape
        _, h, w, c = model.output_shape
        print(colored(f'The degrees of freedom of the system increases from {df} to {h * w * c}.', 'green'))

    return model
