import time
import tensorflow as tf
from os import path

from Autoencoder import Autoencoder
from experiments.phoneme.plot_results import plot_original, plot_projection, plot_history
from experiments.phoneme.load_data import get_datasets
from experiments.phoneme.metrics import compute_metrics
from experiments.utils import build_seq_encoder, build_seq_decoder


seed = 123
tf.random.set_seed(seed)
root = 'experiments/phoneme/results'
titles = [
    'Few samples without noise',
    'Many samples without noise',
    'Few samples with noise',
    'Many samples with noise'
]

datasets_train, datasets_test = get_datasets(test_size=0.2, seed=seed, noise=0.25)

for (X_train, y_train), (X_test, y_test), title in zip(datasets_train, datasets_test, titles):
    encoder = build_seq_encoder(input_shape=X_train.shape[1:], filters=8, n_components=2, zero_padding=0)
    decoder = build_seq_decoder(output_shape=X_train.shape[1:], filters=8, n_components=2, cropping=0)
    autoencoder = Autoencoder(encoder, decoder)
    tic = time.perf_counter()
    autoencoder.compile(optimizer='adam', loss='mse')
    history = autoencoder.fit(X_train, epochs=50, validation_split=0.1, shuffle=True, batch_size=64, verbose=0)

    X_train_red = autoencoder.encode(X_train)
    tac = time.perf_counter()
    X_test_red = autoencoder.encode(X_test)
    toc = time.perf_counter()
    X_train_rec = autoencoder.decode(X_train_red).numpy()
    X_test_rec = autoencoder.decode(X_test_red).numpy()

    time_in_sample = tac - tic
    time_out_of_sample = toc - tac
    
    plot_original(X_train, y_train, path.join(root, title), 'train_orig')
    plot_original(X_test, y_test, path.join(root, title), 'test_orig')
    plot_projection(X_train_red, y_train, path.join(root, title), 'train_red',)
    plot_original(X_train_rec, y_train, path.join(root, title), 'train_rec')
    plot_projection(X_test_red, y_test, path.join(root, title), 'test_red',)
    plot_original(X_test_rec, y_test, path.join(root, title), 'test_rec')
    
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