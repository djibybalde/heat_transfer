"""
heat_transfer/src/get_data.py

"""

import os
import pandas as pd
import numpy as np
from termcolor import colored
import time
import matplotlib.pyplot as plt

from heat_dist import uniform_dist
from heat_dist import transfer2D
from timer import timer


def get_bounds(data_path):

    # Read data
    bdd_feu_path = os.path.join(data_path, 'bdd_feu/BddFeu - v2014-02-26.xls')
    df = pd.read_excel(bdd_feu_path, sheet_name='Non_combustibles')

    # Select the interested columns and rows.
    columns = [
        'Famille',
        'Densité (kg/m3)',
        'Conductivité (W/m.K)',
        'Capacité thermique (J/kg.K)'
    ]

    name = ['density', 'conduct', 'capacity']

    df = df[columns]
    df = df[df.Famille.str.contains('Métal') | df.Famille.str.contains('construction')]
    df[columns[2]] = df[columns[2]].str.replace(',', '.').astype(float)

    # Get the minimum and maximum values of each variable.
    bounds = {
        name[i]: [df[col].dropna().min(), df[col].dropna().max()] for i, col in enumerate(columns[1:])
    }

    return bounds


# Define a function that helps to generate and save the desired data.
def save_data(data_path,
              input_size=3_000,
              output_size=3_000,
              extension='png',
              use_bounds=True,
              verbose=True,
              show=False
              ):

    """
    Save generated images and the corresponding parameters.

    Args:
        data_path (str): Root directory of the data (where the row, generated, precessed data live).
        input_size (int): Size of the sample to be generated using a uniform distribution.
        output_size (int): Number of possible values for each parameter.
        use_bounds (bool): Whether or not to use the min (low) and max (high) of the Excel file data.
        extension (str): Extension of the file (type of the file). Default is 'png'.
        verbose (bool): Whether or not verbose.
        show (bool): Whether or not to show images.

    Return:
        The output data are directly saved.

    """

    # Create a path where to save the data.
    out_data = os.path.join(data_path, 'row/')
    if not os.path.exists(out_data):
        os.makedirs(out_data)
    else:
        raise RuntimeError(
            colored(f'Directory "{out_data}" exists. To not overwrite existing data, check your path.', 'red'))

    if use_bounds:
        # Use the Excel file to get the minimum (low) and maximum (high) values.
        bounds = get_bounds(data_path)

        # In order to reduce the computation time, we modify some values (min/max).
        density = uniform_dist(bounds['density'][0]*100, bounds['density'][1]/2, input_size)
        conduct = uniform_dist(bounds['conduct'][0]*100, bounds['conduct'][1]*100, input_size)
        capacity = uniform_dist(bounds['capacity'][0]*10, bounds['capacity'][1]/4, input_size)

    else:
        density = uniform_dist(7600 / 2, 7600 * 2, input_size)
        conduct = uniform_dist(47 / 2, 47 * 2, input_size)
        capacity = uniform_dist(480 / 2, 480 * 2, input_size)

    # The following variables are not available in the Excel file.
    init_temp = uniform_dist(400 / 2, 400 * 2, input_size)
    top_temp = uniform_dist(800 / 2, 800 * 2, input_size)
    bot_temp = uniform_dist(300 / 2, 300 * 2, input_size)
    left_temp = uniform_dist(200 / 2, 200 * 2, input_size)
    right_temp = uniform_dist(500 / 2, 500 * 2, input_size)

    # The following variables are constant (not vary).
    module, length, n_points, final_time = 1, 5, 50, 500

    # To DataFrame
    df = pd.DataFrame(np.c_[density, conduct, capacity, init_temp, top_temp, bot_temp, left_temp, right_temp],
                      columns=['density', 'conduct', 'capacity',
                               'init_temp', 'top_temp', 'bot_temp', 'left_temp', 'right_temp'])

    df['module'], df['length'] = module, length
    df['n_points'], df['final_time'] = n_points, final_time

    # Drop duplicated values.
    df.drop_duplicates(ignore_index=True, subset=['density', 'conduct'], inplace=True)
    df.drop_duplicates(ignore_index=True, subset=['density', 'capacity'], inplace=True)
    df.drop_duplicates(ignore_index=True, subset=['conduct', 'capacity'], inplace=True)
    df.drop_duplicates(ignore_index=True, subset=['bot_temp', 'right_temp'], inplace=True)
    df.drop_duplicates(ignore_index=True, subset=['bot_temp', 'left_temp'], inplace=True)
    df.drop_duplicates(ignore_index=True, subset=['bot_temp', 'top_temp'], inplace=True)
    df.drop_duplicates(ignore_index=True, subset=['left_temp', 'right_temp'], inplace=True)
    df.drop_duplicates(ignore_index=True, subset=['left_temp', 'top_temp'], inplace=True)
    df.drop_duplicates(ignore_index=True, subset=['right_temp', 'top_temp'], inplace=True)

    # Save to "CSV" file named "params".
    df.to_csv(out_data + 'params.csv', index=False)

    # Create a folder for the images, if not exist.
    out_img = os.path.join(out_data, 'images/')
    if not os.path.exists(out_img):
        os.makedirs(out_img)
    else:
        raise RuntimeError(
            colored(f'Directory "{out_img}" exists. To not overwrite existing data, check your path.', 'red'))

    # Applying the the transfer2D function, plot and save the images.
    print(colored(f'\nStart generating {df.shape[0]} images...', 'blue'))

    # Store the started time.
    begin = time.time()

    # For each value of the parameters,
    for i in range(df.shape[0]):

        trans = transfer2D(density=df['density'][i],
                           conduct=df['conduct'][i],
                           capacity=df['capacity'][i],
                           t0=df['init_temp'][i],
                           tt=df['top_temp'][i],
                           tb=df['bot_temp'][i],
                           tl=df['left_temp'][i],
                           tr=df['right_temp'][i],
                           # The following are unchanged.
                           m=module,
                           L=length,
                           n=n_points,
                           ft=final_time
                           )

        # We want cropped and smoothed images with 50x50 pixels, without axis, ...
        plt.figure(figsize=(.65, .65))
        plt.pcolormesh(trans, cmap='jet', shading='gouraud')
        plt.axis('off')
        plt.savefig(out_img + 'img{:04}.{}'.format(i, extension),
                    dpi=100, bbox_inches='tight', pad_inches=0, transparent=True)
        plt.rcParams.update({'figure.max_open_warning': 0})

        if show:
            plt.show()

        # Show progression.
        if verbose and i % (df.shape[0] // 10) == 0:
            print('Processing index {:04d} of {:04d}: ==> {:03d}% ; progress time: {}'
                  .format(i, df.shape[0], round(100 * i / df.shape[0]), timer(begin, time.time())))

    print(colored('\nProcess finished! %d images are generated in %s.' % (df.shape[0], timer(begin, time.time())),
                  'green'))
    print(colored(f'All data are saved in "{out_data}" folder with separated files.', 'cyan'))


# Run
root_path = '../data/'
save_data(root_path, input_size=3_000, use_bounds=False, verbose=True)
