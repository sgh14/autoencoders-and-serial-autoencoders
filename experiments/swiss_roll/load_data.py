import numpy as np
from sklearn.datasets import make_swiss_roll
from sklearn.model_selection import train_test_split


def normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))


def my_colormap1D(x, c1=(0.75, 0, 0.75), c2=(0, 0.75, 0.75)):
    # Calculate the RGB values based on interpolation
    color = np.array(c1) * (1 - x) + np.array(c2) * x

    return color


def my_colormap2D(x, y):
    # Define colors in RGB
    bottom_left = (0.5, 0, 0.5) # dark magenta
    bottom_right = (0, 0.5, 0.5) # dark cyan
    top_left = (1, 0, 1) # magenta
    top_right = (0, 1, 1) # cyan

    # Calculate the RGB values based on interpolation
    top_color = np.array(top_left) * (1 - x) + np.array(top_right) * x
    bottom_color = np.array(bottom_left) * (1 - x) + np.array(bottom_right) * x

    return top_color * (1 - y) + bottom_color * y


def get_data(npoints=2000, test_size=0.5, seed=123, noise=0):
    X, color = make_swiss_roll(npoints, random_state=seed, noise=noise)
    dimension_1 = normalize(color)
    dimension_2 = normalize(X[:, 1])
    y = np.array([my_colormap2D(x, y) for x, y in zip(dimension_1, dimension_2)])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
    X_train_small = X_train[::2]
    y_train_small = y_train[::2]

    data_train = (X_train, y_train)
    data_test = (X_test, y_test)
    data_train_small = (X_train_small, y_train_small)

    return data_train, data_test, data_train_small


def get_datasets(npoints=2000, test_size=0.5, seed=123, noise=1):
    data_clean_train, data_clean_test, data_clean_train_small = get_data(npoints, test_size, seed, 0)
    data_noisy_train, data_noisy_test, data_noisy_train_small = get_data(npoints, test_size, seed, noise)
    # Train-test datasets
    datasets_train = [
        data_clean_train_small, data_clean_train,
        data_noisy_train_small, data_noisy_train
    ]
    datasets_test = [
        data_clean_test, data_clean_test,
        data_noisy_test, data_noisy_test
    ]

    return datasets_train, datasets_test
