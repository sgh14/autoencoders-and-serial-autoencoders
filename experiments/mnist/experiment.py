import time
import numpy as np
import tensorflow as tf
from os import path

from Autoencoder import Autoencoder
from experiments.mnist.plot_results import plot_original, plot_projection, plot_interpolations, plot_history
from experiments.mnist.load_data import get_datasets
from experiments.mnist.metrics import compute_metrics
from experiments.utils import build_conv_encoder, build_conv_decoder


seed = 123
tf.random.set_seed(seed)
root = 'experiments/mnist/results'
titles = [
    'Few samples without noise',
    'Many samples without noise',
    'Few samples with noise',
    'Many samples with noise'
]

datasets_train, datasets_test = get_datasets(npoints=10000, test_size=0.1, seed=seed, noise=0.25)

for (X_train, y_train), (X_test, y_test), title in zip(datasets_train, datasets_test, titles):
    # Añadimos la dimensión de canal
    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)

    encoder = build_conv_encoder(input_shape=X_train.shape[1:], filters=8, n_components=2, zero_padding=(2, 2))
    decoder = build_conv_decoder(output_shape=X_train.shape[1:], filters=8, n_components=2, cropping=(2, 2))
    autoencoder = Autoencoder(encoder, decoder)
    tic = time.perf_counter()
    autoencoder.compile(optimizer='adam', loss='mse')
    history = autoencoder.fit(X_train, epochs=2, validation_split=0.1, shuffle=True, batch_size=64, verbose=0)

    X_train_red = autoencoder.encode(X_train)
    tac = time.perf_counter()
    X_test_red = autoencoder.encode(X_test)
    toc = time.perf_counter()
    X_train_rec = autoencoder.decode(X_train_red).numpy()
    X_test_rec = autoencoder.decode(X_test_red).numpy()

    time_in_sample = tac - tic
    time_out_of_sample = toc - tac
    
    plot_original(
        X_train, y_train,
        path.join(root, title), 'train_orig',
        images_per_class=2, grid_shape=(3, 4)
    )
    plot_original(
        X_test, y_test,
        path.join(root, title), 'test_orig',
        images_per_class=2, grid_shape=(3, 4)
    )
    plot_projection(X_train_red, y_train, path.join(root, title), 'train_red',)
    plot_original(
        X_train_rec, y_train,
        path.join(root, title), 'train_rec',
        images_per_class=2, grid_shape=(3, 4)
    )
    plot_projection(X_test_red, y_test, path.join(root, title), 'test_red',)
    plot_original(
        X_test_rec, y_test,
        path.join(root, title), 'test_rec',
        images_per_class=2, grid_shape=(3, 4)
    )
    plot_interpolations(
        X_train_red,
        y_train,
        autoencoder.decoder,
        path.join(root, title),
        'train_interp',
        class_pairs = [(i, i+1) for i in range(0, 5)],
        n_interpolations=6
    )
    plot_interpolations(
        X_test_red,
        y_test,
        autoencoder.decoder,
        path.join(root, title),
        'test_interp',
        class_pairs = [(i, i+1) for i in range(0, 5)],
        n_interpolations=6
    )
    plot_history(history, path.join(root, 'histories', title), log_scale=True)

    compute_metrics(
        X_train,
        y_train,
        X_train_red,
        X_train_rec,
        X_test,
        y_test,
        X_test_red,
        X_test_rec,
        time_in_sample,
        time_out_of_sample,
        title,
        output_dir=path.join(root, title)
    )