import time
import tensorflow as tf
from os import path

from Autoencoder import Autoencoder
from experiments.helix.plot_results import plot_original, plot_projection, plot_history
from experiments.helix.load_data import get_datasets
from experiments.utils import build_encoder, build_decoder
from experiments.helix.metrics import compute_metrics

seed = 123
tf.random.set_seed(seed)
root = 'experiments/helix/results'
titles = [
    'Few samples without noise',
    'Many samples without noise',
    'Few samples with noise',
    'Many samples with noise'
]
datasets_train, datasets_test = get_datasets(npoints=2000, test_size=0.5, seed=seed, noise=0.1)

for (X_train, y_train), (X_test, y_test), title in zip(datasets_train, datasets_test, titles):
    encoder = build_encoder(input_shape=(X_train.shape[-1],), units=128, n_components=2)
    decoder = build_decoder(output_shape=(X_train.shape[-1],), units=128, n_components=2)
    autoencoder = Autoencoder(encoder, decoder)
    tic = time.perf_counter()
    autoencoder.compile(optimizer='adam', loss='mse')
    history = autoencoder.fit(X_train, epochs=500, validation_split=0.1, shuffle=False, batch_size=64, verbose=0)

    X_train_red = autoencoder.encode(X_train)
    tac = time.perf_counter()
    X_test_red = autoencoder.encode(X_test)
    toc = time.perf_counter()
    X_train_rec = autoencoder.decode(X_train_red)
    X_test_rec = autoencoder.decode(X_test_red)

    plot_original(X_train, y_train, path.join(root, title), 'train_orig')
    plot_original(X_test, y_test, path.join(root, title), 'test_orig')
    plot_projection(X_train_red, y_train, path.join(root, title), 'train_red')
    plot_original(X_train_rec, y_train, path.join(root, title), 'train_rec')
    plot_projection(X_test_red, y_test, path.join(root, title), 'test_red')
    plot_original(X_test_rec, y_test, path.join(root, title), 'test_rec')
    plot_history(history, path.join(root, 'histories', title), log_scale=True)

    time_in_sample = tac - tic
    time_out_of_sample = toc - tac
    compute_metrics(
        X_train,
        X_train_red,
        X_train_rec,
        X_test,
        X_test_red,
        X_test_rec,
        time_in_sample,
        time_out_of_sample,
        title,
        output_dir=path.join(root, title)
    )

   