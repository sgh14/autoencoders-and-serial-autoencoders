from os import path

from Autoencoder import Autoencoder
from experiments.helix.plot_results import plot_original, plot_projection, plot_history
from experiments.helix.load_data import get_datasets
from experiments.utils import build_encoder, build_decoder
from experiments.helix.metrics import compute_metrics


root = 'experiments/helix/results'
titles = [
    'Few samples without noise',
    'Many samples without noise',
    'Few samples with noise',
    'Many samples with noise'
]
datasets_train, datasets_test = get_datasets(npoints=2000, test_size=0.5, seed=123, noise=0.1)
for (X, y), title in zip(datasets_train, titles):
    plot_original(X, y, title, path.join(root, 'train_orig'))

for (X, y), title in zip(datasets_test, titles):
    plot_original(X, y, title, path.join(root, 'test_orig'))

for (X_train, y_train), (X_test, y_test), title in zip(datasets_train, datasets_test, titles):
    encoder = build_encoder(input_shape=(X_train.shape[-1],), units=128, n_components=2)
    decoder = build_decoder(output_shape=(X_train.shape[-1],), units=128, n_components=2)
    autoencoder = Autoencoder(encoder, decoder)
    autoencoder.compile(optimizer='adam', loss='mse')
    history = autoencoder.fit(X_train, epochs=300, validation_split=0.1, shuffle=True, batch_size=64, verbose=0)

    X_train_red = autoencoder.encode(X_train)
    X_test_red = autoencoder.encode(X_test)
    X_train_rec = autoencoder.decode(X_train_red)
    X_test_rec = autoencoder.decode(X_test_red)

    plot_projection(X_train_red, y_train, title, path.join(root, 'train_red'))
    plot_original(X_train_rec, y_train, title, path.join(root, 'train_rec'))
    plot_projection(X_test_red, y_test, title, path.join(root, 'test_red'))
    plot_original(X_test_rec, y_test, title, path.join(root, 'test_rec'))
    plot_history(history, path.join(root, 'histories', title), log_scale=True)

    compute_metrics(X_test, X_test_rec, title, root)        