"""
heat_transfer/src/visualize.py
"""

import os
import glob
import imageio
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from termcolor import colored
from process import ProcessData
from keras.preprocessing.image import load_img
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sns.set(color_codes=True)

# Make a gif animation.
print(colored('\nMaking a GIF animation...', 'blue'))
inter_images = []
inter_path = glob.glob('../data/intermediate/*.png')
for p in inter_path:
    inter_images.append(imageio.imread(p))

imageio.mimsave('../reports/intermediate_predicted.gif', inter_images, fps=5)


# Visual compare between the fakes (predicted) and true images using grids.
# Make grids of images.
def make_grid(images_path, out_grid, real=True, K=10):
    n = 0
    rows, cols = 3, 4
    nn = int(rows * cols)
    idx, images = ProcessData(images_path).process_images(raw=False)

    if not os.path.exists(out_grid):
        os.makedirs(out_grid)

    for k in range(K):
        for i, j in enumerate(idx[n:n + nn]):
            plt.subplot(rows, cols, i + 1)
            plt.imshow(images[i + n, :, :])
            plt.title('img{:04d}'.format(j), fontsize=10)
            plt.axis('off')

        name = 'real' if real else 'fake'
        plt.savefig(out_grid + name + '_img{:04d}.png'.format(k), bbox_inches='tight', pad_inches=0)
        plt.rcParams.update({'figure.max_open_warning': 0})

        n += nn
        plt.close()


print(colored('Making grids for comparing the real and fake images...', 'blue'))
grid_output_path = '../reports/grids_images/'
real_images_path = '../data/test/y_test/'
fake_images_path = '../data/predicted/'

# For real images (y_test).
make_grid(real_images_path, grid_output_path, real=True)

# For predicted images.
make_grid(fake_images_path, grid_output_path, real=False)

# Compare the images.
grid_names = [x for x in sorted(os.listdir(grid_output_path)) if not x.endswith('tiff')]
idx, grids = ProcessData(grid_output_path).process_images(raw=False)

for i in range(int(len(grid_names) / 2)):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.imshow(load_img(grid_output_path + 'real_img{:04d}.png'.format(i), color_mode="rgb"))
    ax1.set_title('True Images ({:02d})'.format(i))
    ax1.axis('off')

    ax2.imshow(load_img(grid_output_path + 'fake_img{:04d}.png'.format(i), color_mode="rgb"))
    ax2.set_title('Fake Images ({:02d})'.format(i))
    ax2.axis('off')

    plt.savefig(os.path.join(grid_output_path + 'grids_{:04d}.tiff'.format(i)))
plt.close()

print(colored(f'Grids are saved in "{grid_output_path}" folder.', 'green'))
# Remove the no needed grids.
for file in os.listdir(grid_output_path):
    file_p = os.path.join(grid_output_path, file)
    if os.path.exists(file_p) and not file_p.endswith('tiff'):
        os.remove(file_p)

# Read the images data.
idx_real, images_real = ProcessData('../data/test/y_test/').process_images(raw=False)
idx_fake, images_fake = ProcessData('../data/predicted/').process_images(raw=False)

idx = [i for i, j in zip(idx_real, idx_fake) if i == j]
mse = []
mae = []
for k in range(images_real.shape[0]):
    if np.all(idx == sorted(idx)):  # images are sorted by index.
        mn = images_real[k, :, :].shape[0] * images_real[k, :, :].shape[1]

        # TODO: https://www.pyimagesearch.com/2014/09/15/python-compare-two-images/
        se = (np.sum((images_real[k, :, :] - images_fake[k, :, :]) ** 2)) / mn
        ae = (np.sum(abs(images_real[k, :, :] - images_fake[k, :, :]))) / mn
        mse.append(round(se, 5))
        mae.append(round(ae, 5))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
sns.distplot(mse, bins=200, hist=True, color='green', ax=ax1, label='MSE: Histogram')
sns.distplot(mse, bins=200, hist=False, color='blue', ax=ax1, label='MSE: Density curve')
sns.distplot(mae, bins=200, hist=True, color='green', ax=ax2, label='MAE: Histogram')
sns.distplot(mae, bins=200, hist=False, color='blue', ax=ax2, label='MAE: Density curve')

ax1.text(0.125, 150., r'$\mu = {:0.3f}$'.format(np.mean(mse)))
ax1.text(0.125, 140., r'$\sigma = {:0.3f}$'.format(np.std(mse)))
ax2.text(0.350, 17.5, r'$\mu = {:0.3f}$'.format(np.mean(mae)))
ax2.text(0.350, 16.2, r'$\sigma = {:0.3f}$'.format(np.std(mae)))

ax1.set_ylabel('Probability Density'), ax1.set_xlabel('Mean Squared Error')
ax2.set_ylabel('Probability Density'), ax2.set_xlabel('Mean Absolute Error')
ax1.legend(), ax2.legend()

plt.savefig('../reports/' + 'mse_mae.png')

# Using Scikit-Learn libraries to compute MSE, MAE and R-square.
# Reshape the pixels matrix to one column.
real_reshaped = images_real.reshape(-1, )
fake_reshaped = images_fake.reshape(-1, )

# Compute the MSE, MAE and R-square.
mse = np.round(mean_squared_error(real_reshaped, fake_reshaped), 3)
mae = np.round(mean_absolute_error(real_reshaped, fake_reshaped), 3)
r_2 = np.round(r2_score(real_reshaped, fake_reshaped), 3)

print(colored('Model metrics: MSE={:03f}, MAE={:03f}, R2={:03f}.'.format(mse, mae, r_2), 'cyan'))

# Compute the difference between the predicted and true pixels (the errors).
error = fake_reshaped - real_reshaped

# Plot an histogram and density of the errors.
plt.figure(figsize=(10, 6))
sns.distplot(error, bins=130, hist=True, color='cyan', vertical=True)
sns.distplot(error, bins=130, hist=False, color='red', vertical=True)
plt.title('Histogram and density of the Errors', fontsize=17)
plt.xlabel('Errors counted')
plt.ylabel('Errors values')
plt.ylim([-0.5, 0.5])
plt.text(15.0, 0.40, r'$\mu_\epsilon = {:0.3f}$'.format(error.mean()))
plt.text(15.0, 0.35, r'$\sigma_\epsilon = {:0.3f}$'.format(error.std()))

plt.savefig('../reports/' + 'ErrorsHistogram.png')
print(colored('Histogram of the errors is saved in "../reports/" folder.', 'green'))
plt.show()

# TODO: look at a metric for the performance of the model. ideas: Confusion matrix
# TODO: How to modify the input size of the data such that images from the model architecture ?
# Comparing two images: https://www.pyimagesearch.com/2014/09/15/python-compare-two-images/
